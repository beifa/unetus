import torch
import torch.nn as nn
from .convblock import convBlock


class Decoder(nn.Module):
    def __init__(
        self,
        in_chanels: int,
        num_block: int,
        residual: bool,
        pool,
        activation: str,
        normolization: str,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.unsample = nn.Upsample(
            scale_factor=2, align_corners=False, mode="trilinear"
        )
        for _ in range(num_block):
            self.convblock = convBlock(
                in_chanels=in_chanels * 3,
                out_chanels=in_chanels,
                residual=residual,
                pool=pool,
                activation=activation,
                normolization=normolization,
            )
            self.blocks.append(self.convblock)
            in_chanels //= 2

    def forward(self, x, connection):
        for encoding_block, skip_connection in zip(
            self.blocks, connection[::-1]
        ):  # noqa 501
            x = self.unsample(x)
            x = torch.concat((skip_connection, x), axis=1)
            x, _ = encoding_block(x)
        return x

    @property
    def out_channels(self):
        return self.blocks[-1].out_channels
