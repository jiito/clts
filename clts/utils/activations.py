# run a model and save the activations
import os
from pathlib import PosixPath

import datasets
import einops
import h5py
import nnsight
import torch
from jaxtyping import Float
import tqdm
from pydantic import BaseModel
from torch.utils.data import DataLoader

from clts.modal.logging import timeout_trace


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class ActivationCacheConfig(BaseModel):
    model_name: str
    dataset_path: str
    save_path: str
    batch_size: int
    hg_token: str


def get_nnsight_model(model_name: str):
    return nnsight.LanguageModel(model_name, device_map="auto", dispatch=True)


def get_dataset_and_loader(cfg: ActivationCacheConfig):
    try:
        token_dataset = datasets.load_dataset(
            cfg.dataset_path,
            token=cfg.hg_token,
            split="train",
        )
        loader = DataLoader(
            token_dataset,
            batch_size=cfg.batch_size * 4,
            pin_memory=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e
    return token_dataset, loader


@torch.no_grad()
def extract_activations(
    llm: nnsight.LanguageModel, tokens, n_layers
) -> Float[torch.Tensor, "batch_size seq_len 2 n_layers d_model"]:
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model is using device: {next(llm.parameters()).device}")
    llm.requires_grad_(False)
    tokens = tokens.to(device)
    print(f"Tokens are on device: {tokens.get_device()}")

    llm.eval()

    mlp_acts: Float[torch.Tensor, "batch_size seq_len 2 n_layers d_model"] = None
    with llm.trace(tokens):
        mlp_ins = []
        mlp_outs = []
        for layer in range(n_layers):
            # Note: we NEED to save the layer norm input because this is necessary for the linear combination
            mlp_in = llm.transformer.h[
                layer
            ].ln_2.input.save()  # shape: [batch_size, seq_len, d_model]
            mlp_out = llm.transformer.h[
                layer
            ].mlp.output.save()  # shape: [batch_size, seq_len, d_model]
            mlp_ins.append(mlp_in)
            mlp_outs.append(mlp_out)
            del mlp_in, mlp_out

        mlp_ins = mlp_ins.save()
        mlp_outs = mlp_outs.save()

    mlp_ins = einops.rearrange(
        mlp_ins,
        " n b s e -> b s n e ",
    )
    # mlp_ins shape: [batch_size, seq_len, n_layers, d_model]
    mlp_outs = einops.rearrange(
        mlp_outs,
        " n b s e -> b s n e ",
    )

    # shape: [batch_size, seq_len, 2 (mlp-in, mlp-out), n_layers, d_model]
    mlp_acts = einops.rearrange([mlp_ins, mlp_outs], "l b s n e -> b s l n e")

    del mlp_ins, mlp_outs

    return mlp_acts


def compute_and_save_activations(
    loader: DataLoader,
    config: ActivationCacheConfig,
    llm: nnsight.LanguageModel,
    max_dataset_size: int = 1000000,
):
    with h5py.File(config.save_path, "w") as f:
        seq_len = min(llm.config.max_position_embeddings, 511)  # pick your window

        h5_dataset = f.create_dataset(
            "activations",
            # [dataset_size, 2 (in, out), n_layers, d_model]
            shape=(
                min(len(loader) * config.batch_size * seq_len, max_dataset_size),
                2,  # (in, out)
                llm.config.num_layers,
                llm.config.hidden_size,
            ),
            dtype="float32",
        )

        h5_pointer = 0
        dataset_size = h5_dataset.shape[0]

        for batch in tqdm.tqdm(loader):
            print(f"{h5_pointer / dataset_size * 100:.1f}% done")

            tokenizer = llm.tokenizer

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token  # GPT-3-style models

            enc = tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(get_device())

            timeout_trace(llm, input_ids)

            activations = extract_activations(llm, input_ids, llm.config.num_layers)

            token_activations = einops.rearrange(
                activations,
                "b s l n e -> (b s) l n e",
            )

            n_tokens = token_activations.shape[0]

            if h5_pointer + n_tokens > dataset_size:
                remaining_space = dataset_size - h5_pointer
                h5_dataset[h5_pointer:] = (
                    token_activations[:remaining_space].cpu().numpy()
                )
                print(f"Dataset full. Saved {dataset_size} token activations.")
                break
            else:
                h5_dataset[h5_pointer : h5_pointer + n_tokens] = (
                    token_activations.cpu().numpy()
                )
                h5_pointer += n_tokens
