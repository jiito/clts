from jaxtyping import Float
import modal
import wandb
import torch as t
import tqdm

from clts.modal.app import app
from clts.models.clt import CrossLayerTranscoder
from clts.models.georg_clt import CrosslayerDecoder
from clts.utils.activations import get_device
from clts.utils.buffer import DiscBuffer
from clts.modal.app import app, gpu, HOURS, torch_image, volume_path, volume


@app.function(
    gpu=gpu,
    timeout=1 * HOURS,
    image=torch_image,
    volumes={volume_path: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train():
    device = get_device()

    activations_path_h5_modal = volume_path / "activations" / "openai-community_gpt2.h5"
    batch_size = 32
    epochs = 3
    lr = 1e-3
    lambda_ = 0.0002
    c = 0.1

    config = {
        "batch_size": 32,
        "activations_path_h5_modal": activations_path_h5_modal,
        "epochs": epochs,
        "lr": lr,
        "lambda": lambda_,
        "c": c,
    }

    wandb.init(
        project="gpt2-clts-modal",
        config=config,
        settings=wandb.Settings(init_timeout=120),
    )

    buffer = DiscBuffer(activations_path_h5_modal, accessor="activations")
    loader = t.utils.data.DataLoader(
        buffer,
        num_workers=15,
        prefetch_factor=2,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )

    model = CrossLayerTranscoder(d_activations=768, d_features=768 * 8, n_layers=12)
    wandb.watch(model, log="all", log_freq=15)

    model = model.to(device)

    optimizer = t.optim.AdamW(model.parameters(), lr=lr)

    loss_list = []

    examples_seen = 0

    output_standardizer = model.output_standardizer
    current_sparsity_penalty = model.current_sparsity_penalty
    use_tanh = True

    for epoch in range(epochs):
        pbar = tqdm.tqdm(loader)

        model.train()

        for batch_idx, batch in enumerate(pbar):
            loss = training_step(
                batch_idx,
                batch,
                model,
                output_standardizer,
                c,
                current_sparsity_penalty,
                use_tanh,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            examples_seen += batch.shape[0]

            pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}")

    wandb.finish()


def current_sparsity_penalty(max_steps, global_step, _lambda):
    n_steps = max_steps
    current_step = global_step  # use global step instead of batch idx to work with gradient accumulation
    cur_lambda = _lambda * (current_step / n_steps)
    wandb.log(
        {
            "training/sparsity_penalty": cur_lambda,
        },
        step=global_step,
    )
    return cur_lambda


def training_step(
    batch_idx, batch, model, output_standardizer, c, current_sparsity_penalty, use_tanh
):
    # Initialize standardizers
    if batch_idx == 0:
        model.initialize_standardizers(batch)

    # Forward pass
    resid, mlp_out = batch[:, 0], batch[:, 1]
    pre_actvs, features, recons_norm, recons = model.forward(resid)

    # self.update_dead_features(features)
    # Compute MSE loss
    mse = (recons_norm - output_standardizer.standardize(mlp_out)) ** 2

    # Compute Sparsity Loss
    # with torch.no_grad():
    if isinstance(model.decoder, CrosslayerDecoder):
        dec_norms = t.zeros_like(features[:1])
        for l in range(model.decoder.n_layers):
            W = model.decoder.get_parameter(
                f"W_{l}"
            )  # (from_layer, d_features, d_acts)
            dec_norms[:, : l + 1] = dec_norms[:, : l + 1] + (W**2).sum(dim=-1)
        dec_norms = dec_norms.sqrt()

    weighted_features = features * dec_norms * c
    wandb.log(
        {
            "model/weighted_features_mean": weighted_features.detach().mean().cpu(),
        },
        step=batch_idx,
    )

    if use_tanh:
        weighted_features = t.tanh(
            weighted_features
        )  # (batch_size, n_layers, d_features)
    sparsity = current_sparsity_penalty() * weighted_features.sum(dim=[1, 2]).mean()
    wandb.log(
        {
            "training/sparsity_loss": sparsity,
        },
        step=batch_idx,
    )

    loss = mse.mean() + sparsity
    wandb.log(
        {
            "training/loss": loss,
        },
        step=batch_idx,
    )

    # # Log training metrics
    # if batch_idx % log_metrics_every == 0:
    #     log_training_metrics(features, recons_norm, recons, mlp_out, batch_idx)

    return loss


@t.inference_mode()
def evaluate(
    model: CrossLayerTranscoder,
):
    model.eval()

    pass


def sparsity_loss(
    mlp_out: Float[t.Tensor, "batch_size n_layers d_activations"],
    encoder_out: Float[t.Tensor, "batch_size n_layers d_features"],
    logits: Float[t.Tensor, "batch_size n_layers d_activations"],
    decoder_weights_per_layer: Float[t.Tensor, "n_layers d_features d_activations"],
    lambda_: float,
    c: float,
) -> float:
    """
    The loss function is MSE of the logits vs the actual post-mlp activations
    """

    mse_term = t.norm(t.subtract(mlp_out, logits), p=2, dim=-1).square().sum(dim=1)
    # The regularization is the  sum over every feature for every layer tanh(c * norm of W_dec(layer, feature) * post-mlp acts)
    regularization_term = 0
    for layer_l, decoder_weights in enumerate(decoder_weights_per_layer):
        layer_reg = (
            (t.mul(t.norm(decoder_weights, dim=-1), encoder_out) * c)
            .tanh()
            .sum(dim=[-2, -1])
        )
        regularization_term += layer_reg
    loss = mse_term + (regularization_term * lambda_)

    return loss.mean()


@app.local_entrypoint()
def main():
    train.remote()
