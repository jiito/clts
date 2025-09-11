from huggingface_hub import HfFolder
from clts.utils.activations import (
    ActivationCacheConfig,
    extract_activations,
    get_dataset_and_loader,
    get_device,
    get_nnsight_model,
)


def test_activation_extraction():
    cfg = ActivationCacheConfig(
        model_name="roneneldan/TinyStories-1Layer-21M",
        dataset_path="roneneldan/TinyStories",
        save_path="test_activations",
        batch_size=1,
        hf_token=HfFolder.get_token(),
    )
    _, loader = get_dataset_and_loader(cfg)
    model = get_nnsight_model(cfg.model_name)
    seq_len = min(model.config.max_position_embeddings, 511)  # pick your window

    i = 0

    for batch in loader:
        if i > 0:
            break
        tokenizer = model.tokenizer

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

        activations = extract_activations(model, input_ids, model.config.num_layers)

        assert activations.shape == (
            input_ids.shape[0],
            seq_len,
            2,
            model.config.num_layers,
            model.config.hidden_size,
        )

        i += 1
