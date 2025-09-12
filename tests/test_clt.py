import torch
from clts.models.clt import CrossLayerTranscoder


def test_clt():
    batch_size = 100
    d_activations = 768
    d_features = 1023
    n_layers = 12

    clt = CrossLayerTranscoder(
        d_activations=d_activations, d_features=d_features, n_layers=n_layers
    )
    x = torch.randn((batch_size, n_layers, d_activations))
    logits, encoder_out, concat_w_dec = clt.forward(x)

    assert logits.shape == x.shape

    assert encoder_out.shape == (batch_size, n_layers, d_features)

    assert len(concat_w_dec) == n_layers
