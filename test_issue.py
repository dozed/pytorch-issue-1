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

    assert torch.allclose(img, img_updated)
    assert torch.equal(img, img_updated)

    # forward pass
    result = model(img_updated)

    return result


def test_add_zero_different_result():
    # prepare model
    model = nn.Sequential(
        nn.Conv2d(3, 129, kernel_size=3, padding=1),
        Inception(129, 4, 6, 8, 4, 8, 8),
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    result1 = do_forward_pass(model, add_zeros=True)
    result2 = do_forward_pass(model, add_zeros=False)

    # ISSUE: the results are not equal but should be, since only zeros are added
    assert torch.allclose(result1, result2)
    assert torch.equal(result1, result2)
