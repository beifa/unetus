import torch
from unetus.encoding import Encoder


def encoding(
       in_chanels: int,
       out_channels_first: int,
       num_block: int,
       data: torch.tensor
) -> tuple:
    encoder = Encoder(
        in_chanels=in_chanels,
        out_channels_first=out_channels_first,
        num_block=num_block,
        pool='Max',
        activation='ReLU',
        normolization='BatchNorm3d'
    )
    return encoder(data)


def test_count_block():
    x = torch.empty(1, 1, 64, 64, 64)
    num_block = 3
    _, connect = encoding(
        in_chanels=1,
        out_channels_first=8,
        num_block=num_block,
        data=x
    )
    assert len(connect) == num_block


def test_correct_dim_and_type():
    x = torch.empty(1, 1, 64, 64, 64)
    out, _ = encoding(
        in_chanels=1,
        out_channels_first=8,
        num_block=3,
        data=x
    )
    assert torch.is_tensor(out)
    assert out.shape.numel() == 32 * 8 * 8 * 8


def test_out_channels():
    block = Encoder(
        in_chanels=1,
        out_channels_first=64,
        num_block=3,
        pool='Max',
        activation='ReLU',
        normolization='BatchNorm3d'
    )
    assert block.out_channels == 256
