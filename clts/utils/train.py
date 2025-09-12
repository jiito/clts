from jaxtyping import Float
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
)
def train():
    device = get_device()

    # TODO: fix this path
    activations_path_h5_modal = "/vol/data/activations/openai-community_gpt2.h5"
    batch_size = 32

    buffer = DiscBuffer(activations_path_h5_modal)
    loader = t.utils.data.DataLoader(
        buffer,
        num_workers=20,
        prefetch_factor=2,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )

    model = CrossLayerTranscoder(d_activations=768, d_features=768 * 8, n_layers=12)

    optimizer = t.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 3

    loss_list = []

    for epoch in range(epochs):
        pbar = tqdm.tqdm(loader)

        for batch in pbar:
            # Move data to device, perform forward pass
            batch = batch.to(device)
            # TODO: check shape
            logits = model(batch)

            # Calculate loss
            loss = sparsity_loss(batch, logits)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}")


def sparsity_loss(
    mlp_out: Float[t.Tensor, "batch_size n_layers d_activations"],
    encoder_out: Float[t.Tensor, "batch_size n_layers d_features"],
    logits: Float[t.Tensor, "batch_size n_layers d_activations"],
    decoder_weights: Float[t.Tensor, "n_layers d_features d_activations"],
    lambda_: float,
    c: float,
) -> float:
    """
    The loss function is MSE of the logits vs the actual post-mlp activations
    """

    mse_term = t.norm(t.subtract(mlp_out, logits), p=2, dim=-1).square().sum(dim=1)
    # The regularization is the  sum over every feature for every layer tanh(c * norm of W_dec(layer, feature) * post-mlp acts)
    regularization_term = (
        (t.mul(t.norm(decoder_weights, dim=-1), encoder_out) * c)
        .tanh()
        .sum(dim=[-2, -1])
    ) * lambda_

    loss = mse_term + regularization_term

    return loss.mean()


@app.local_entrypoint()
def main():
    train.remote()
