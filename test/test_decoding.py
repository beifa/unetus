import torch
from unetus.encoding import Encoder
from unetus.decoding import Decoder
from unetus.convblock import convBlock

POOL, ACTIVATION, NORMALIZATION = "Max", "ReLU", "BatchNorm3d"


def encoding(
    in_chanels: int,
    out_channels_first: int,
    num_block: int
):
    encoder = Encoder(
        in_chanels=in_chanels,
        out_channels_first=out_channels_first,
        num_block=num_block,
        pool=POOL,
        activation=ACTIVATION,
        normolization=NORMALIZATION,
    )
    return encoder


def decoding(
    in_chanels: int,
    num_block: int,
    pool: str,
):
    decoder = Decoder(
        in_chanels=in_chanels,
        num_block=num_block,
        pool=pool,
        activation=ACTIVATION,
        normolization=NORMALIZATION,
    )
    return decoder


def cblock(
    x: list,
    in_chanels: int,
    out_chanels: int,
    pool: str
) -> list:
    block = convBlock(
        in_chanels=in_chanels,
        out_chanels=out_chanels,
        pool=pool,
        activation=ACTIVATION,
        normolization=NORMALIZATION,
    )
    return block(x)


def test_overall_decoder():
    r""" test case """
    x = torch.empty(1, 1, 64, 64, 64)
    num_block = 3
    out_channels_first = 8
    encoder = encoding(1, out_channels_first, num_block)
    in_chanels = encoder.out_channels
    out, connect = encoder(x)
    out = cblock(out, in_chanels, in_chanels * 2, None)
    assert out.shape.numel() == 64 * 8 * 8 * 8
    decoder = decoding(in_chanels, num_block=len(connect), pool=None)
    x = decoder(out, connect)
    assert x.shape.numel() == out_channels_first * (x.shape[-1] ** 3)


def test_decoder_1():
    r""" test case """
    x = torch.empty(1, 32, 8, 8, 8)
    connect = [
        torch.empty(1, 8, 64, 64, 64),
        torch.empty(1, 16, 32, 32, 32),
        torch.empty(1, 32, 16, 16, 16),
    ]
    in_chanels = x.shape[1]
    decoder = decoding(in_chanels, num_block=len(connect), pool=None)
    x = cblock(x, in_chanels, in_chanels * 2, None)
    x = decoder(x, connect)
    assert x.shape.numel() == decoder.out_channels * (x.shape[-1] ** 3)


def test_decoder_2():
    r""" test case """
    x = torch.empty(1, 64, 4, 4, 4)
    connect = [
        torch.empty(1, 8, 64, 64, 64),
        torch.empty(1, 16, 32, 32, 32),
        torch.empty(1, 32, 16, 16, 16),
        torch.empty(1, 64, 8, 8, 8),
    ]
    in_chanels = x.shape[1]
    decoder = decoding(in_chanels, num_block=len(connect), pool=None)
    x = cblock(x, in_chanels, in_chanels * 2, None)
    x = decoder(x, connect)
    assert x.shape.numel() == decoder.out_channels * (x.shape[-1] ** 3)


def test_decoder_3():
    r""" test case """
    x = torch.empty(1, 256, 4, 4, 4)
    connect = [
        torch.empty(1, 32, 64, 64, 64),
        torch.empty(1, 64, 32, 32, 32),
        torch.empty(1, 128, 16, 16, 16),
        torch.empty(1, 256, 8, 8, 8),
    ]
    in_chanels = x.shape[1]
    decoder = decoding(in_chanels, num_block=len(connect), pool=None)
    x = cblock(x, in_chanels, in_chanels * 2, None)
    x = decoder(x, connect)
    assert x.shape.numel() == decoder.out_channels * (x.shape[-1] ** 3)


def test_decoder_out_channels():
    r""" test case """
    x = torch.empty(1, 32, 8, 8, 8)
    connect = [
        torch.empty(1, 8, 64, 64, 64),
        torch.empty(1, 16, 32, 32, 32),
        torch.empty(1, 32, 16, 16, 16),
    ]
    in_chanels = x.shape[1]
    decoder = decoding(in_chanels, num_block=len(connect), pool=None)
    assert decoder.out_channels == 8
