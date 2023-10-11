import torch.nn as nn


class convBlock(nn.Module):
    def __init__(
        self,
        in_chanels: int,
        out_chanels: int,
        pool,
        activation: str,
        normolization: str,
    ):
        super().__init__()
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

    def forward(self, x):
        if not self.pool:
            return self.block(x)
        to_connect = self.block(x)
        out = self.pool(to_connect)
        return out, to_connect

    @property
    def out_channels(self):
        return self.block[-3].out_channels
