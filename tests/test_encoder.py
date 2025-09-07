import torch
from models.clt import Encoder


def test_encoder():
    encoder = Encoder(n_layers=2, d_activations=12, d_features=10)
    x = torch.randn(100, 2, 12)
    act = encoder(x)
    assert act.shape == (100, 2, 10)
