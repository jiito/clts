from typing import List, Tuple
import einops
import torch.nn as nn
from einops import einsum
import torch as t
from jaxtyping import Float

# from clts.models.jump_relu import JumpReLU
from clts.models.georg_jumprelu import JumpReLU
from clts.models.standardizer import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


class InputStandardizer(nn.Module):
    def __init__(self, n_layers: int, d_activations: int):
        super().__init__()
        self.register_buffer("mean", t.zeros((n_layers, d_activations)))
        self.register_buffer("std", t.zeros((n_layers, d_activations)))

    def initialize_from_batch(
        self, batch: Float[t.Tensor, "batch_size io n_layers actv_dim"]
    ):
        batch_in = batch[:, 0]
        # Note we take the average over the batch dimension for every layer and activation
        self.mean = batch_in.mean(dim=0)
        self.std = batch_in.std(dim=0)
        # TODO: add clamping

    def forward(self, batch: Float[t.Tensor, "batch_size io n_layers actv_dim"]):
        return (batch - self.mean) / self.std


class OutputStandardizer(nn.Module):
    def __init__(self, n_layers: int, d_activations: int):
        super().__init__()
        self.register_buffer("mean", t.zeros((n_layers, d_activations)))
        self.register_buffer("std", t.zeros((n_layers, d_activations)))

    def initialize_from_batch(
        self, batch: Float[t.Tensor, "batch_size io n_layers actv_dim"]
    ):
        batch_in = batch[:, 1]
        # Note we take the average over the batch dimension for every layer and activation
        self.mean = batch_in.mean(dim=0)
        self.std = batch_in.std(dim=0)

    def forward(self, batch: Float[t.Tensor, "batch_size io n_layers actv_dim"]):
        return (batch * self.std) + self.mean


class Encoder(nn.Module):
    def __init__(self, d_activations: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_activations = d_activations
        self.d_features = d_features
        self.n_layers = n_layers

        #  setup the encoder weights
        # shape: [n_layers, d_activations, d_features]
        self.W_enc = nn.Parameter(t.empty((n_layers, d_activations, d_features)))

        self.b = nn.Parameter(t.empty((n_layers, d_features)))

        self.reset_parameters()

    def reset_parameters(self):
        # Taken from: https://github.com/Goreg12345/crosslayer-transcoder/blob/master/crosslayer_transcoder/model/clt.py
        enc_uniform_thresh = 1 / (self.d_features**0.5)
        self.W_enc.data.uniform_(-enc_uniform_thresh, enc_uniform_thresh)
        self.b.data.zero_()

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

        return activations


class Decoder(nn.Module):
    def __init__(self, d_activations: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_activations = d_activations
        self.d_features = d_features
        self.n_layers = n_layers

        self.b = nn.Parameter(t.empty((n_layers, d_activations)))

        # register W_dec for each layer
        for i in range(n_layers):
            # for each layer we have to create a decoder weight with shape [i+1, d_features, d_activations]
            self.register_parameter(
                f"W_{i}", nn.Parameter(t.empty((i + 1, d_features, d_activations)))
            )

        self.reset_parameters()

    def reset_parameters(self):
        # copied from: https://github.com/Goreg12345/crosslayer-transcoder/blob/master/crosslayer_transcoder/model/clt.py
        dec_uniform_thresh = 1 / ((self.d_activations * self.n_layers) ** 0.5)
        for i in range(self.n_layers):
            self.get_parameter(f"W_{i}").data.uniform_(
                -dec_uniform_thresh, dec_uniform_thresh
            )

        self.b.data.zero_()

    def decoder_weights_for_layer(self, layer_index: int):
        return self.get_parameter(f"W_{layer_index}")

    def forward(
        self, x: Float[t.Tensor, "batch_size n_layers d_features"]
    ) -> Float[t.Tensor, "batch_size n_layers d_activations"]:
        """
        The forward pass will sum the combination of each of the decoder weight mats with the layer activations for all the previous layers

        The role of the decoder is to project the encoders activations back to residual stream space (post-MLP activation space). Takes into account all previous layers' outputs.
        """

        # run over the layers and compute predictions for each

        batch_size = x.shape[0]
        preds = t.empty(
            (batch_size, self.n_layers, self.d_activations), device=x.device
        )

        for i in range(self.n_layers):
            # get the weight matrix
            W_dec = self.get_parameter(f"W_{i}")

            prev_layer_features = x[:, : i + 1, :]

            layer_predictions = einsum(
                prev_layer_features,
                W_dec,
                "batch_size n_layers d_features, n_layers d_features d_activations -> batch_size d_activations",
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

        self.input_standardizer = InputStandardizer()
        self.output_standardizer = OutputStandardizer()
        self.encoder = Encoder(d_activations, d_features, n_layers)
        self.decoder = Decoder(d_activations, d_features, n_layers)
        self.activation_fun = JumpReLU(
            theta=0.03, bandwidth=1.0, n_layers=n_layers, d_features=d_features
        )

        # following standardization technique of georg
        self.input_standardizer = DimensionwiseInputStandardizer(
            n_layers=n_layers, activation_dim=d_activations
        )
        self.output_standardizer = DimensionwiseOutputStandardizer(
            n_layers=n_layers, activation_dim=d_activations
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def initialize_standardizers(
        self, batch: Float[t.Tensor, "batch_size io n_layers d_acts"]
    ):
        self.input_standardizer.initialize_from_batch(batch)
        self.output_standardizer.initialize_from_batch(batch)

    def forward(
        self, x: Float[t.Tensor, "batch_size n_layers d_activations"]
    ) -> Tuple[
        Float[t.Tensor, "batch_size n_layers d_activations"],  # logits
        Float[t.Tensor, "batch_size n_layers d_features"],  # encoder_out
        List[Float[t.Tensor, "d_features concat_activations"]],  # concat_w_dec
    ]:
        """ """
        # TODO: standardize input

        standardized_x = self.input_standardizer(x)

        encoder_out = self.encoder(standardized_x)

        act_out = self.activation_fun(encoder_out)

        logits = self.decoder(act_out)

        # TODO: do we standardize here or before the decoder?
        standardized_logits = self.output_standardizer(logits)

        concat_w_dec = []
        for i in range(self.n_layers):
            layer_w_dec = self.decoder.decoder_weights_for_layer(i)

            concat_for_feat = einops.rearrange(layer_w_dec, "i f a -> f (a i)")

            concat_w_dec.append(concat_for_feat)

        return logits, encoder_out, concat_w_dec


class Encoder(nn.Module):
    def __init__(self, d_activations: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_activations = d_activations
        self.d_features = d_features
        self.n_layers = n_layers

        #  setup the encoder weights
        # shape: [n_layers, d_activations, d_features]
        self.W_enc = nn.Parameter(t.empty((n_layers, d_activations, d_features)))

        self.b = nn.Parameter(t.empty((n_layers, d_features)))

        self.reset_parameters()

    def reset_parameters(self):
        # Taken from: https://github.com/Goreg12345/crosslayer-transcoder/blob/master/crosslayer_transcoder/model/clt.py
        enc_uniform_thresh = 1 / (self.d_features**0.5)
        self.W_enc.data.uniform_(-enc_uniform_thresh, enc_uniform_thresh)
        self.b.data.zero_()

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

        return activations


class Decoder(nn.Module):
    def __init__(self, d_activations: int, d_features: int, n_layers: int):
        super().__init__()
        self.d_activations = d_activations
        self.d_features = d_features
        self.n_layers = n_layers

        self.b = nn.Parameter(t.empty((n_layers, d_activations)))

        # register W_dec for each layer
        for i in range(n_layers):
            # for each layer we have to create a decoder weight with shape [i+1, d_features, d_activations]
            self.register_parameter(
                f"W_{i}", nn.Parameter(t.empty((i + 1, d_features, d_activations)))
            )

        self.reset_parameters()

    def reset_parameters(self):
        # copied from: https://github.com/Goreg12345/crosslayer-transcoder/blob/master/crosslayer_transcoder/model/clt.py
        dec_uniform_thresh = 1 / ((self.d_activations * self.n_layers) ** 0.5)
        for i in range(self.n_layers):
            self.get_parameter(f"W_{i}").data.uniform_(
                -dec_uniform_thresh, dec_uniform_thresh
            )

        self.b.data.zero_()

    def decoder_weights_for_layer(self, layer_index: int):
        return self.get_parameter(f"W_{layer_index}")

    def forward(
        self, x: Float[t.Tensor, "batch_size n_layers d_features"]
    ) -> Float[t.Tensor, "batch_size n_layers d_activations"]:
        """
        The forward pass will sum the combination of each of the decoder weight mats with the layer activations for all the previous layers

        The role of the decoder is to project the encoders activations back to residual stream space (post-MLP activation space). Takes into account all previous layers' outputs.
        """

        # run over the layers and compute predictions for each

        batch_size = x.shape[0]
        preds = t.empty(
            (batch_size, self.n_layers, self.d_activations), device=x.device
        )

        for i in range(self.n_layers):
            # get the weight matrix
            W_dec = self.get_parameter(f"W_{i}")

            prev_layer_features = x[:, : i + 1, :]

            layer_predictions = einsum(
                prev_layer_features,
                W_dec,
                "batch_size n_layers d_features, n_layers d_features d_activations -> batch_size d_activations",
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
        self.activation_fun = JumpReLU(
            theta=0.03, bandwidth=1.0, n_layers=n_layers, d_features=d_features
        )

        # following standardization technique of georg
        self.input_standardizer = DimensionwiseInputStandardizer(
            n_layers=n_layers, activation_dim=d_activations
        )
        self.output_standardizer = DimensionwiseOutputStandardizer(
            n_layers=n_layers, activation_dim=d_activations
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def initialize_standardizers(
        self, batch: Float[t.Tensor, "batch_size io n_layers d_acts"]
    ):
        self.input_standardizer.initialize_from_batch(batch)
        self.output_standardizer.initialize_from_batch(batch)

    def forward(
        self, x: Float[t.Tensor, "batch_size n_layers d_activations"]
    ) -> Tuple[
        Float[t.Tensor, "batch_size n_layers d_activations"],  # logits
        Float[t.Tensor, "batch_size n_layers d_features"],  # encoder_out
        List[Float[t.Tensor, "d_features concat_activations"]],  # concat_w_dec
    ]:
        """ """
        # TODO: standardize input

        standardized_x = self.input_standardizer(x)

        encoder_out = self.encoder(standardized_x)

        act_out = self.activation_fun(encoder_out)

        logits = self.decoder(act_out)

        # TODO: do we standardize here or before the decoder?
        standardized_logits = self.output_standardizer(logits)

        concat_w_dec = []
        for i in range(self.n_layers):
            layer_w_dec = self.decoder.decoder_weights_for_layer(i)

            concat_for_feat = einops.rearrange(layer_w_dec, "i f a -> f (a i)")

            concat_w_dec.append(concat_for_feat)

        return logits, encoder_out, concat_w_dec
