import torch as t
from jaxtyping import Float

from clts.utils.train import sparsity_loss

batch_size = 32
n_layers = 12
d_activations = 768
d_features = 1023


def test_sparsity_loss_sanity_check():
    mlp_out: Float[t.Tensor, "batch_size n_layers d_activations"]
    encoder_out: Float[t.Tensor, "batch_size n_layers d_features"]
    logits: Float[t.Tensor, "batch_size n_layers d_activations"]
    decoder_weights: Float[t.Tensor, "n_layers d_features d_activations"]
    lambda_: float = 1
    c: float = 1

    mlp_out = t.ones((batch_size, n_layers, d_activations))
    encoder_out = t.ones((batch_size, n_layers, d_features))
    logits = t.randn((batch_size, n_layers, d_activations))
    decoder_weights = t.randn((n_layers, d_features, d_activations))

    loss: t.Tensor = sparsity_loss(
        mlp_out,
        encoder_out,
        logits,
        decoder_weights,
        lambda_,
        c,
    )

    print(loss)

    assert loss.dim() == 0


def test_sparsity_loss_comparison():
    # compare loss for dense vs sparse encoder outs
    lambda_: float = 1
    c: float = 1

    mlp_out = t.ones((batch_size, n_layers, d_activations))
    dense_encoder_out = t.ones((batch_size, n_layers, d_features))
    sparse_encoder_out = t.zeros((batch_size, n_layers, d_features))
    logits = t.randn((batch_size, n_layers, d_activations))
    decoder_weights = t.randn((n_layers, d_features, d_activations))

    dense_loss: t.Tensor = sparsity_loss(
        mlp_out,
        dense_encoder_out,
        logits,
        decoder_weights,
        lambda_,
        c,
    )
    sparse_loss: t.Tensor = sparsity_loss(
        mlp_out,
        sparse_encoder_out,
        logits,
        decoder_weights,
        lambda_,
        c,
    )

    print(dense_loss, sparse_loss)

    assert dense_loss > sparse_loss
