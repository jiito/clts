import torch
from clts.models.clt import Decoder


def test_decoder():
    decoder = Decoder(d_activations=12, d_features=10, n_layers=2)
    test_features = torch.randn(100, 2, 10)
    act = decoder(test_features)
    # expected shape: (100, 2, 12)
    assert act.shape == (100, 2, 12)
