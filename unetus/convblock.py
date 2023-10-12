import torch.nn as nn


class convBlock(nn.Module):
    def __init__(
        self,
        in_chanels: int,
        out_chanels: int,
        residual: bool,
        pool,
        activation: str,
        normolization: str,
    ):
        super().__init__()
        self.residual = residual
        # pool
        if pool:
            try:
                self.pool = getattr(nn, f"{pool}Pool3d")(kernel_size=2)
            except AttributeError:
                raise AttributeError(
                    f"Module with name {f'{pool}Pool3d'} not find use: "
                    "Max, FractionalMax, Avg, AdaptiveMax, AdaptiveAvg or None when skip"  # noqa 501
                )
        else:
            self.pool = None  # noqa 701
        # activation
        self.activation = getattr(nn, activation)()  # (inplace=True)
        # normolization
        self.normolization = getattr(nn, normolization)
        # block
        self.block = nn.Sequential(
            nn.Conv3d(in_chanels, out_chanels, kernel_size=3, padding=1),
            self.normolization(out_chanels),
            self.activation,
            nn.Conv3d(out_chanels, out_chanels, kernel_size=3, padding=1),
            self.normolization(out_chanels),
            self.activation,
        )
        self.residual_block = nn.Conv3d(
            in_chanels, out_chanels, kernel_size=1, padding=0
        )

    def forward(self, x):
        to_connect = []
        if self.residual:
            conv_residual = self.residual_block(x)
        if not self.pool:
            out = self.block(x)
        else:
            conv_residual = 0
            to_connect = self.block(x)
            out = self.pool(to_connect)
        out = out if not self.residual else out + conv_residual
        return out, to_connect

    @property
    def out_channels(self):
        return self.block[-3].out_channels
