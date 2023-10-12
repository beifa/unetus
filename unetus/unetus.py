import torch.nn as nn
from unetus.encoding import Encoder
from unetus.decoding import Decoder
from unetus.convblock import convBlock


class Unet3D(nn.Module):
    r"""
    in_chanels: image chanels
        - 1, gray
        - 3, RGB
    out_chanels: num labels
    num_block: count encoder & decoder block (depth model)
    """

    def __init__(
        self,
        in_chanels: int = 1,
        labels: int = 1,
        out_channels_first: int = 8,
        num_block: int = 3,
        residual: bool = False,
        unsample_type: str = 'conv',
        pool="Max",
        activation: str = "ReLU",
        normolization: str = "BatchNorm3d",
    ):
        super().__init__()
        self.encoder = Encoder(
            in_chanels=in_chanels,
            out_channels_first=out_channels_first,
            num_block=num_block,
            residual=residual,
            pool=pool,
            activation=activation,
            normolization=normolization,
        )
        in_chanels = self.encoder.out_channels
        self.decoder = Decoder(
            in_chanels=in_chanels,
            num_block=num_block,
            residual=residual,
            unsample_type=unsample_type,
            pool=None,
            activation=activation,
            normolization=normolization,
        )
        self.bottom = convBlock(
            in_chanels,
            in_chanels * 2,
            residual=residual,
            pool=None,
            activation=activation,
            normolization=normolization,
        )
        in_chanels_decoder = self.decoder.out_channels
        self.final_layer = nn.Conv3d(in_chanels_decoder, labels, 1)

    def forward(self, x):
        out, connections = self.encoder(x)
        out, _ = self.bottom(out)
        out = self.decoder(out, connections)
        return self.final_layer(out)
