import torch.nn as nn
from functools import reduce
from .convblock import convBlock


class Encoder(nn.Module):
    def __init__(
        self,
        in_chanels: int = 1,
        out_channels_first: int = 8,
        num_block: int = 3,
        pool: str = 'Max',
        activation: str = 'ReLU',
        normolization: str = 'BatchNorm3d'
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_block):
            self.convblock = convBlock(
                in_chanels,
                out_channels_first,
                pool=pool,
                activation=activation,
                normolization=normolization
            )
            self.blocks.append(self.convblock)
            in_chanels = out_channels_first
            out_channels_first *= 2

    def forward(self, x):
        _, _, x1, y1, z1 = x.shape
        assert x1 == y1 == z1, f'error all axis should be eaul size get x: {x1}, y: {y1}, z: {z1}'  # noqa 501
        chk_correct_depth = reduce(
            lambda x, y: x//y, [x1] + [2] * len(self.blocks)
        )
        if chk_correct_depth == 0:
            raise ValueError(
                f"Depth model not correct reduce size depth, current value: {len(self.blocks)}"  # noqa 501
            )
        skip_connections = []
        for encoding_block in self.blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return x, skip_connections

    @property
    def out_channels(self):
        return self.blocks[-1].out_channels
