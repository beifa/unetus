import torch
import torch.nn as nn
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
