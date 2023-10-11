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
        pool="Max",
        activation: str = "ReLU",
        normolization: str = "BatchNorm3d",
    ):
        super().__init__()
        self.encoder = Encoder(
            in_chanels=in_chanels,
            out_channels_first=out_channels_first,
            num_block=num_block,
            pool=pool,
            activation=activation,
            normolization=normolization,
        )
        in_chanels = self.encoder.out_channels
        self.decoder = Decoder(
            in_chanels=in_chanels,
            num_block=num_block,
            pool=None,
            activation=activation,
            normolization=normolization,
        )
        self.bottom = convBlock(
            in_chanels,
            in_chanels * 2,
            pool=None,
            activation=activation,
            normolization=normolization,
        )
        in_chanels_decoder = self.decoder.out_channels
        self.final_layer = nn.Conv3d(in_chanels_decoder, labels, 1)

    def forward(self, x):
        print("encoder")
        out, connections = self.encoder(x)
        out = self.bottom(out)
        print("decoder")
        out = self.decoder(out, connections)
        return self.final_layer(out)
