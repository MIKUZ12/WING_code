import torch
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as F


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x


class Crop(nn.Module):
    """Crop every frame in a ``[T,C,H,W]`` clip to one fixed ROI."""

    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = int(top)
        self.left = int(left)
        self.height = int(height)
        self.width = int(width)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only crop a tensor of 4 dims."
        return F.crop(x, self.top, self.left, self.height, self.width)

    def extra_repr(self):
        return (
            f"top={self.top}, left={self.left}, "
            f"height={self.height}, width={self.width}"
        )

class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)
    
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)
