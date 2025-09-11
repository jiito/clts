import torch
from clts.models.clt import CrossLayerTranscoder


def test_clt():
    clt = CrossLayerTranscoder(d_activations=12, d_features=10, n_layers=2)
    x = torch.randn(100, 2, 12)
    act = clt(x)
    assert act.shape == (100, 2, 12)
