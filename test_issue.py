import os
import random

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.transforms import functional as FT


def set_seed(seed: int) -> None:
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # disable cuDNN benchmarking and cuDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def test_add_zero_different_result():
    set_seed(42)

    # prepare input
    img = torchvision.io.image.read_image('sky1024px.jpg')
    img = FT.convert_image_dtype(img, torch.float32)
    img = img.unsqueeze(dim=0)

    # prepare input + zero
    zeros = torch.zeros_like(img)
    img_updated = img + zeros

    # input tensors are identical
    assert torch.allclose(img, img_updated)
    assert torch.equal(img, img_updated)

    # prepare model
    conv1 = nn.Conv2d(in_channels=3, out_channels=129, kernel_size=3, padding=1)
    conv2 = nn.Conv2d(in_channels=129, out_channels=4, kernel_size=1)
    conv1.requires_grad_(False)
    conv2.requires_grad_(False)

    # forward 1
    x = conv1(img)
    result1 = conv2(x)

    # forward 2
    y = conv1(img_updated)
    result2 = conv2(y)

    # tensors after conv1 are equal
    assert torch.allclose(x, y)
    assert torch.equal(x, y)

    # ISSUE: the results are not equal but should be, since only zeros are added
    print(torch.linalg.norm(result1 - result2))  # 1.6691e-06
    assert torch.allclose(result1, result2)
    assert torch.equal(result1, result2)
