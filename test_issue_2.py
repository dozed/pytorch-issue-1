import torch
from torch import nn

from util import set_seed


def test_add_zero_different_result_2():
    set_seed(42)
    torch.set_printoptions(precision=9)

    # prepare non-contiguous input
    img = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ],
        dtype=torch.float32
    )

    img = img.permute([2, 1, 0])
    img = img.unsqueeze(dim=0)

    zeros = torch.zeros_like(img)
    img_updated = img + zeros
    assert not img.is_contiguous()
    assert not img_updated.is_contiguous()

    conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
    conv1.requires_grad_(False)

    x = conv1(img)
    y = conv1(img_updated)
    assert x.is_contiguous()
    assert not y.is_contiguous()

    print(torch.linalg.norm(x - y))  # 5.960464478e-08
    print(torch.eq(x, y))

    assert torch.allclose(x, y)
    assert torch.equal(x, y)
