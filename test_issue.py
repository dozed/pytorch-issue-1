import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as FT

from Inception import Inception

device = torch.device('cpu')


def do_forward_pass(
        model: nn.Module,
        add_zeros: bool,
) -> Tensor:
    # prepare input
    img = torchvision.io.image.read_image('sky1024px.jpg')
    img = FT.convert_image_dtype(img, torch.float32)
    img = img.unsqueeze(dim=0)
    img = img.to(device)

    # maybe add zero tensor
    if add_zeros:
        zeros = torch.zeros_like(img)
        img_updated = img + zeros
    else:
        img_updated = img

    # forward pass
    x = model(img_updated)

    return x


def test_add_zero_different_result():
    """
    PyTorch issue: for the same input, different results are computed
    """
    n_channels = 129  # the results are all equal with n_channels <= 128

    # prepare model
    model = nn.Sequential(
        nn.Conv2d(3, n_channels, kernel_size=3, padding=1),
        Inception(n_channels, 4, 6, 8, 4, 8, 8),
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    # the results are equal when adding zeros both times or never
    for add_zeros in [True, False]:
        result1 = do_forward_pass(model, add_zeros=add_zeros)
        result2 = do_forward_pass(model, add_zeros=add_zeros)

        assert torch.allclose(result1, result2)
        assert torch.equal(result1, result2)

    # the results are not equal when one time zeros are added and the other time no zeros are added
    for add_zeros_1, add_zeros_2 in [(False, True), (True, False)]:
        result1 = do_forward_pass(model, add_zeros=add_zeros_1)
        result2 = do_forward_pass(model, add_zeros=add_zeros_2)

        assert not torch.allclose(result1, result2)
        assert not torch.equal(result1, result2)
