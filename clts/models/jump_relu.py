import torch.nn as nn
from torch.types import Tensor

# taken from: https://github.com/erichson/JumpReLU/blob/master/activationfun.py#L12


# TODO: add trainable theta parameter
class JumpReLU(nn.Module):
    r"""
    This is a module that implements the Jump Rectified Linear Unit function element-wise
    """

    def __init__(self, jump=0.0):
        super().__init__()
        self.jump = jump

    def _threshold(self, arr: Tensor, threshold=0.0) -> Tensor:
        arr[arr <= threshold] = 0.0
        return arr

    def forward(self, x: Tensor) -> Tensor:
        return self._threshold(x, self.jump)
