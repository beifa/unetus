import torch.nn as nn
from functools import reduce
from .convblock import convBlock


class Encoder(nn.Module):
    def __init__(
        self,
        in_chanels: int = 1,
        out_channels_first: int = 8,
        num_block: int = 3,
        residual: bool = False,
        pool="Max",
        activation: str = "ReLU",
        normolization: str = "BatchNorm3d",
    ):
        super().__init__()
        self.in_chanels = in_chanels
        self.blocks = nn.ModuleList()
        for _ in range(num_block):
            self.convblock = convBlock(
                in_chanels,
                out_channels_first,
                residual=residual,
                pool=pool,
                activation=activation,
                normolization=normolization,
            )
            self.blocks.append(self.convblock)
            in_chanels = out_channels_first
            out_channels_first *= 2

    def forward(self, x):
        if not len(x.shape) == 5:
            raise ValueError(
                f"model data dim should be equal (b, c, x, y, z), get: {len(x.shape)}"  # noqa 501
            )
        _, ch, x1, y1, z1 = x.shape
        assert ch == self.in_chanels, \
            f"error set chanels and data chanels not equal, get: {ch} != input_ch: {self.in_chanels}"  # noqa 501
        assert x1 == y1 == z1, \
            f'error all axis should be eaul size get x: {x1}, y: {y1}, z: {z1}'  # noqa 501
        chk_correct_depth = reduce(lambda x, y: x//y, [x1] + [2] * len(self.blocks))  # noqa 501
        if chk_correct_depth <= 1:
            raise ValueError(
                f"Depth model not correct reduce size depth, current value: {len(self.blocks)} or up size data"  # noqa 501
            )
        skip_connections = []
        for encoding_block in self.blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return x, skip_connections

    @property
    def out_channels(self):
        return self.blocks[-1].out_channels
