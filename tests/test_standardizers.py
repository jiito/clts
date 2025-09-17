import torch
from clts.models.clt import InputStandardizer, OutputStandardizer


def test_standardizers():
    batch_size = 100
    d_activations = 768
    d_features = 1023
    n_layers = 12

    batch = torch.randn((batch_size, 2, n_layers, d_activations))

    input_standarizer = InputStandardizer(
        n_layers=n_layers, d_activations=d_activations
    )
    input_standarizer.initialize_from_batch(batch)

    output_standarizer = OutputStandardizer(
        n_layers=n_layers, d_activations=d_activations
    )

    output_standarizer.initialize_from_batch(batch)

    s_in = input_standarizer(batch)

    s_out = output_standarizer(s_in)

    assert s_out.shape == batch.shape
