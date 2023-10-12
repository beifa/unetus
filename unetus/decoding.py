import torch
import torch.nn as nn
from .convblock import convBlock


class Decoder(nn.Module):
    def __init__(
        self,
        in_chanels: int,
        num_block: int,
        residual: bool,
        unsample_type: str,
        pool,
        activation: str,
        normolization: str,
    ):
        super().__init__()
        self.unsample_type = unsample_type
        self.blocks = nn.ModuleList()
        self.ct_blocks = nn.ModuleList()
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
            self.ct_blocks.append(
                nn.ConvTranspose3d(in_chanels*2, in_chanels*2, kernel_size=2, stride=2)  # noqa 501
            )
            in_chanels //= 2

    def forward(self, x, connection):
        for idx, (encoding_block, skip_connection) in enumerate(
            zip(self.blocks, connection[::-1])
        ):
            if not self.unsample_type == 'transpose':
                x = self.unsample(x)
            else:
                x = self.ct_blocks[idx](x)
            x = torch.concat((skip_connection, x), axis=1)
            x, _ = encoding_block(x)
        return x

    @property
    def out_channels(self):
        return self.blocks[-1].out_channels
