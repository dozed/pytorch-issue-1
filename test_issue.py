import os
import random

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.transforms import functional as FT

from util import set_seed


def test_add_zero_different_result():
    set_seed(42)
    torch.set_printoptions(precision=9)

    # prepare input
    img = torchvision.io.image.read_image('sky1024px.jpg')
    img = FT.convert_image_dtype(img, torch.float32)
    img = img[:, :1, :3]  # take only a subset of the image
    img = img.unsqueeze(dim=0)
    # img = img.contiguous()
    # print(img.is_contiguous())

    # prepare input + zero
    zeros = torch.zeros_like(img)
    img_updated = img + zeros

    # input tensors are identical
    assert torch.allclose(img, img_updated)
    assert torch.equal(img, img_updated)

    # prepare model
    conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1)
    conv1.requires_grad_(False)

    # forward 1
    x = conv1(img)
    y = conv1(img_updated)

    print()
    print(x)
    print(y)
    print(torch.eq(x, y))

    # ISSUE: the results are not equal but should be, since only zeros are added
    print(torch.linalg.norm(x - y))  # 6.0069e-08
    assert torch.allclose(x, y)
    assert torch.equal(x, y)
