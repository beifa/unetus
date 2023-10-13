import torch
import torch.nn as nn
from torch.nn import functional as F
from .convblock import convBlock


class Decoder(nn.Module):
    def __init__(
        self,
        in_chanels: int,
        num_block: int,
        residual: bool,
        unsample_type: str,
        crop_conn: bool,
        pool,
        activation: str,
        normolization: str,
    ):
        super().__init__()
        self.unsample_type = unsample_type
        self.crop_conn = crop_conn
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
            xori = x.clone()
            if self.crop_conn:
                print('crop_conn, crop')
                skip_connection = self.crop_connection(x, skip_connection)
            if not self.unsample_type == 'transpose':
                x = self.unsample(x)
            else:
                x = self.ct_blocks[idx](x)
            print(x.shape, skip_connection.shape)
            x = torch.concat(
                (skip_connection, x if not self.crop_conn else xori),
                axis=1
            )
            if self.crop_conn:
                print('crop_conn uns')
                print(x.shape, skip_connection.shape, 'afte conv')
                x = self.unsample(x)
            x, _ = encoding_block(x)
            print(x.shape, skip_connection.shape, 'afte conv')
        return x

    def crop_connection(self, x, connection):
        r"""
        x==y==z = we need last
        #  to pad the last 3 dimensions,
            use padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            padding_front,
            padding_back
            == 6
        """
        zx, zconnection = x.shape[-1], connection.shape[-1]
        assert zx <= zconnection, \
            f'error wrong data crop zx: {zx}, zconnection: {zconnection}'
        skip_value = (zconnection - zx) // 2
        skip_value = [-skip_value] * 6
        connection = F.pad(connection, skip_value)
        znew = connection.shape[-1]
        assert zx == znew, \
            f'error, after negative pad zx: {zx} not equal znew: {znew}'
        return connection

    @property
    def out_channels(self):
        return self.blocks[-1].out_channels
