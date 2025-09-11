import torch.nn as nn
from einops import einsum
import torch as t
from jaxtyping import Float

from clts.models.jump_relu import JumpReLU


class Encoder(nn.Module):
    def __init__(self, d_activations: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_activations = d_activations
        self.d_features = d_features
        self.n_layers = n_layers

        # set activation function
        # TODO: in other implementations, the non-linearity is in the main module not the encoder
        self.activation_func = JumpReLU()

        #  setup the encoder weights
        # shape: [n_layers, d_activations, d_features]
        self.W_enc = nn.Parameter(t.randn((n_layers, d_activations, d_features)))

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

        activations: Float[t.Tensor, "batch_size n_layers d_features"] = einsum(
            x,
            self.W_enc,
            "batch layers activations, layers activations features -> batch layers features",
        )

        # Note: this is for pytorch memory management
        activations = activations.contiguous()

        # apply non-linearity

        activations: Float[t.Tensor, "batch_size n_layers d_features"] = (
            self.activation_func(activations)
        )

        return activations


class Decoder(nn.Module):
    def __init__(self, d_activations: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_activations = d_activations
        self.d_features = d_features
        self.n_layers = n_layers

        # register W_dec for each layer
        for i in range(n_layers):
            # for each layer we have to create a decoder weight with shape [i+1, d_features, d_activations]
            self.register_parameter(
                f"W_{i}", nn.Parameter(t.randn((i + 1, d_features, d_activations)))
            )

    def forward(
        self, x: Float[t.Tensor, "batch_size n_layers d_features"]
    ) -> Float[t.Tensor, "batch_size n_layers d_activations"]:
        """
        The forward pass will sum the combination of each of the decoder weight mats with the layer activations for all the previous layers

        The role of the decoder is to project the encoders activations back to residual stream space (post-MLP activation space). Takes into account all previous layers' outputs.
        """

        # run over the layers and compute predictions for each

        batch_size = x.shape[0]
        preds = t.empty((batch_size, self.n_layers, self.d_activations))

        for i in range(self.n_layers):
            # get the weight matrix
            W_dec = self.get_parameter(f"W_{i}")

            prev_layer_features = x[:, : i + 1, :]

            layer_predictions = einsum(
                prev_layer_features,
                W_dec,
                f"batch_size n_layers d_features, n_layers d_features d_activations -> batch_size d_activations",
            )

            preds[:, i, :] = layer_predictions

            # TODO: add bias here

        return preds


class CrossLayerTranscoder(nn.Module):
    def __init__(self, d_activations, d_features, n_layers):
        super().__init__()
        self.d_activations = d_activations
        self.d_features = d_features
        self.n_layers = n_layers

        self.encoder = Encoder(d_activations, d_features, n_layers)
        self.decoder = Decoder(d_activations, d_features, n_layers)

    def forward(self, x):
        """ """
        x = self.encoder(x)
        return self.decoder(x)
