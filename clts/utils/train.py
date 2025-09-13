from jaxtyping import Float
import modal
import wandb
import torch as t
import tqdm

from clts.modal.app import app
from clts.models.clt import CrossLayerTranscoder
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

    wandb.init(project="gpt2-clts-modal", config=config)

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

    for epoch in range(epochs):
        pbar = tqdm.tqdm(loader)

        model.train()

        for batch in pbar:
            assert batch.shape == (batch_size, 2, 12, 768)
            mlp_in = batch[:, 0, :, :]
            mlp_in = mlp_in.to(device)
            assert mlp_in.shape == (batch_size, 12, 768)
            mlp_out = batch[:, 1, :, :]
            mlp_out = mlp_out.to(device)
            assert mlp_out.shape == (batch_size, 12, 768)
            logits, encoder_out, concat_w_dec = model(mlp_in)

            loss = sparsity_loss(
                mlp_out=mlp_out,
                encoder_out=encoder_out,
                logits=logits,
                decoder_weights_per_layer=concat_w_dec,
                lambda_=lambda_,
                c=c,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            examples_seen += batch.shape[0]

            wandb.log(
                {
                    "loss": loss.item(),
                },
                step=examples_seen,
            )

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}")

    wandb.finish()


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
