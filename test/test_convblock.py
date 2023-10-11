import torch
from torch.nn import (
    MaxPool3d,
    AvgPool3d,
    PReLU,
    ReLU,
    InstanceNorm3d,
    BatchNorm3d
)
from unetus.convblock import convBlock


def cblock(x: list, out_chanels: int) -> list:
    block = convBlock(
        in_chanels=1,
        out_chanels=out_chanels,
        pool="Max",
        activation="ReLU",
        normolization="BatchNorm3d",
    )
    return block(x)[0]


def inside_block(
    in_chanels: int,
    out_chanels: int,
    pool: str,
    activation: str,
    normolization: str
):
    return convBlock(in_chanels, out_chanels, pool, activation, normolization)


def test_block_12():
    r""" test case """
    x = torch.empty(1, 1, 64, 64, 64)
    result = cblock(x, 12)
    assert torch.is_tensor(result)
    assert result.shape.numel() == 12 * 32 * 32 * 32


def test_block_32():
    r""" test case """
    x = torch.empty(1, 1, 64, 64, 64)
    result = cblock(x, 32)
    assert torch.is_tensor(result)
    assert result.shape.numel() == 32 * 32 * 32 * 32


def test_out_channels():
    r""" test case """
    block = inside_block(1, 12, "Max", "ReLU", "BatchNorm3d")
    assert block.out_channels == 12


def test_pool():
    r""" test case """
    block_max = inside_block(1, 12, "Max", "ReLU", "BatchNorm3d")
    block_avg = inside_block(1, 12, "Avg", "ReLU", "BatchNorm3d")
    assert isinstance(block_max.pool, MaxPool3d)
    assert isinstance(block_avg.pool, AvgPool3d)


def test_activation():
    r""" test case """
    block_ReLU = inside_block(1, 12, "Max", "ReLU", "BatchNorm3d")
    block_PReLU = inside_block(1, 12, "Max", "PReLU", "BatchNorm3d")
    assert isinstance(block_PReLU.activation, PReLU)
    assert isinstance(block_ReLU.activation, ReLU)


def test_normolization():
    r""" test case """
    block_batch = inside_block(1, 12, "Max", "ReLU", "BatchNorm3d")
    block_inst = inside_block(1, 12, "Max", "PReLU", "InstanceNorm3d")
    assert isinstance(block_batch.block[1], BatchNorm3d)
    assert isinstance(block_inst.block[1], InstanceNorm3d)


def test_cout_block():
    r""" test case """
    block = inside_block(1, 12, "Max", "ReLU", "BatchNorm3d")
    assert len(block.block) == 6
