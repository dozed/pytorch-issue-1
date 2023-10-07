import torch
from torch import nn

from util import set_seed


def print_tensor_info(label: str, x: torch.Tensor) -> None:
    print(f'---[ tensor info : {label} ]---')
    print(x)
    print(f'nbytes : {x.storage().nbytes()}')
    print(f'numel : {x.numel()}')
    print(f'sizes : {x.size()}')
    print(f'is_contiguous : {x.is_contiguous()}')
    print(f'strides : {x.stride()}')
    print(f'dtype : {x.dtype}')
    print(f'layout : {x.layout}')
    print()


def test_add_zero_different_result_2():
    set_seed(42)
    torch.set_printoptions(precision=9)
    print()

    # prepare non-contiguous NCHW input, simulating torchvision.io and multiple batch examples
    img = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ],
        dtype=torch.float32
    )
    img = img.permute([2, 1, 0])  # test passes without permute
    img = img.unsqueeze(dim=0)

    # add zeros
    zeros = torch.zeros_like(img)
    # zeros = torch.zeros(size=img.size(), dtype=img.dtype)  # test passes using torch.zeros
    img_updated = img + zeros
    # img_updated = img_updated.contiguous()  # test passes with contiguous tensor

    print_tensor_info('img', img)
    print_tensor_info('img_updated', img_updated)
    print_tensor_info('zeros', zeros)

    conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
    conv1.requires_grad_(False)

    x = conv1(img)
    y = conv1(img_updated)

    print_tensor_info('x', x)
    print_tensor_info('y', y)

    print(f'torch.linalg.norm(x - y) : {torch.linalg.norm(x - y)}')  # 5.960464478e-08
    print(f'torch.allclose(x, y) : {torch.allclose(x, y)}')
    print(f'torch.equal(x, y) : {torch.equal(x, y)}')
    print(f'torch.eq(x, y) : {torch.eq(x, y)}')

    assert torch.allclose(x, y)
    assert torch.equal(x, y)
