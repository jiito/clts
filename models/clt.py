import torch.nn as nn
from einops import einsum
import torch as t
from jaxtyping import Float

from models.jump_relu import JumpReLU


class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_activations: int, d_features: int):
        super().__init__()
        self.n_layers = n_layers
        self.d_activations = d_activations
        self.d_features = d_features

        # set activation function
        # TODO: in other implementations, the non-linearity is in the main module not the encoder
        self.activation_func = JumpReLU()

        #  setup the encoder weights
        # shape: [n_layers, d_activations, d_features]
        self.W_enc = nn.Parameter(t.randn(n_layers, d_activations, d_features))

        # TODO: we can include a bias term as well

        self.reset_parameters()

    def reset_parameters(self):
        # This is a standard pytorch method that inits the parameters
        # TODO: here we might optimize how we init the weight mats
        pass

    def forward(
        self, x: Float[t.Tensor, "batch_size n_layers d_activations"]
    ) -> Float[t.Tensor, "batch_size n_layers d_features"]:
        """
        The foward pass of the encoder takes each layers pre-MLP activations and then multiplies them by the encoder matrix

        Then it passes the output through a non-linear activation function jumpReLU
        """

        activations = einsum(
            x,
            self.W_enc,
            "batch layers activations, layers activations features -> batch layers features",
        )

        # Note: this is for pytorch memory management
        activations = activations.contiguous()

        # apply non-linearity
        activations = self.activation_func(activations)

        return activations


class Decoder(nn.Module):
    def __init__(self, d_activations: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_activations = d_activations
        self.d_features = d_features
        self.n_layers = n_layers

        self.W_dec = nn.Parameter()

    def forward(
        self, x: Float[t.Tensor, "batch_size d_activations"]
    ) -> Float[t.Tensor, "batch_size d_features"]:
        """
        The forward pass will sum the combination of each of the decoder weight mats with the layer activations for all the previous layers

        The role of the decoder is to project the encoders activations back to residual stream space (post-MLP activation space). Takes into account all previous layers' outputs.
        """
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
        """ """
        x = self.encoder(x)
        return self.decoder(x)
