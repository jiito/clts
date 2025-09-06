import torch.nn as nn

from models.jump_relu import JumpReLU


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation_func = JumpReLU()

    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return x


class CrossLayerTranscoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        assert self.input_dim == self.output_dim

        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, x):
        """
        The forward pass will sum the combination of each of the decoder weight mats with the layer activations for all the previous layers

        The role of the decoder is to project the encoders activations back to residual stream space (post-MLP activation space). Takes into account all previous layers' outputs.
        """
        x = self.encoder(x)
        return self.decoder(x)
