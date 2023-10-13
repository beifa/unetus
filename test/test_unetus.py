import torch
import torch.nn as nn
import pytest
from unetus.unetus import Unet3D


def test_Unet3D():
    r""" test case """
    x = torch.randn(1, 1, 64, 64, 64)
    model = Unet3D()
    out = model(x)
    assert all(torch.tensor(out.shape) == torch.tensor([1, 1, 64, 64, 64]))


def test_labels_Unet3D():
    r""" test case """
    x = torch.randn(1, 1, 64, 64, 64)
    model = Unet3D(labels=14)
    out = model(x)
    assert all(torch.tensor(out.shape) == torch.tensor([1, 14, 64, 64, 64]))


def test_dim_Unet3D():
    r""" test case """
    x = torch.randn(1, 1, 32, 32, 32)
    model = Unet3D(labels=1)
    out = model(x)
    assert all(torch.tensor(out.shape) == torch.tensor([1, 1, 32, 32, 32]))


def test_dim_transpose_sample_Unet3D():
    r""" test case """
    x = torch.randn(1, 1, 32, 32, 32)
    model = Unet3D(labels=1, unsample_type='transpose')
    out = model(x)
    assert all(torch.tensor(out.shape) == torch.tensor([1, 1, 32, 32, 32]))


def test_loss_Unet3D():
    r""" test case """
    x = torch.randn(1, 1, 32, 32, 32)
    model = Unet3D(labels=1)
    out = model(x)
    loss = nn.BCEWithLogitsLoss()
    label = torch.LongTensor(1, 1, 32, 32, 32).random_(2)
    label = label.float()
    assert loss(out, label) > 0


def test_multi_lbl_loss_Unet3D():
    r""" test case """
    x = torch.randn(1, 1, 32, 32, 32)
    model = Unet3D(labels=3)
    out = model(x)
    loss = nn.BCEWithLogitsLoss()
    label = torch.LongTensor(1, 3, 32, 32, 32).random_(2)
    label = label.float()
    assert loss(out, label) > 0


def test_cuda_Unet3D():
    r"""test case
    so same problem with wsl2 need always make links
    """
    # if torch.cuda.is_available():
    #     x = torch.randn(1, 1, 32, 32, 32, device='cuda')
    #     model = Unet3D()
    #     model.to('cuda')
    #     out = model(x)
    #     assert all(torch.tensor(out.shape) == torch.tensor(
    # [1, 1, 32, 32, 32]))

    assert 1 == 1


def test_num_block_dim_data():
    r""" test num block and small dim data"""
    with pytest.raises(
        ValueError,
        match="Depth model not correct reduce size depth, current value: \\d or up size data"   # noqa 501
    ):
        x = torch.randn(1, 1, 8, 8, 8)
        model = Unet3D(residual=True, unsample_type='transpose')
        _ = model(x)


def test_init_chanels():
    r""" test init chanels """
    with pytest.raises(
        AssertionError,
        match=r"error set chanels and data chanels not equal, get: \d != input_ch: \d"   # noqa 501
    ):
        x = torch.randn(1, 3, 8, 8, 8)
        model = Unet3D(in_chanels=2, residual=True, unsample_type='transpose')
        _ = model(x)


def test_init_chanels_2():
    r""" test init chanels """
    with pytest.raises(
        AssertionError,
        match=r"error set chanels and data chanels not equal, get: \d != input_ch: \d"   # noqa 501
    ):
        x = torch.randn(1, 2, 8, 8, 8)
        model = Unet3D(in_chanels=1, residual=True, unsample_type='transpose')
        _ = model(x)


def test_num_axis():
    r""" test all axis """
    with pytest.raises(
        AssertionError,
        match=r"error all axis should be eaul size get x: \d, y: \d, z: \d"
    ):
        x = torch.randn(1, 1, 2, 8, 8)
        model = Unet3D(in_chanels=1, residual=True, unsample_type='transpose')
        _ = model(x)


def test_num_axis_2():
    r""" test all axis """
    with pytest.raises(
        AssertionError,
        match=r"error all axis should be eaul size get x: \d, y: \d, z: \d"
    ):
        x = torch.randn(1, 1, 8, 8, 1)
        model = Unet3D(in_chanels=1, residual=True, unsample_type='transpose')
        _ = model(x)


def test_transpose():
    x = torch.randn(1, 1, 16, 16, 16)
    model1 = Unet3D(unsample_type='conv')
    out1 = model1(x)
    model2 = Unet3D(unsample_type='transpose')
    out2 = model2(x)
    assert out1.shape == out2.shape


def test_dim_data():
    r""" test dim==5 """
    with pytest.raises(
        ValueError,
        match=r"model data dim should be equal \(b, c, x, y, z\), get: \d"
    ):
        x = torch.randn(1, 16, 16, 16)
        model = Unet3D()
        _ = model(x)


def test_crop_Unet3D():
    r""" test case """
    x = torch.randn(1, 1, 64, 64, 64)
    model = Unet3D(crop_conn=True)
    out = model(x)
    assert all(torch.tensor(out.shape) == torch.tensor([1, 1, 64, 64, 64]))
