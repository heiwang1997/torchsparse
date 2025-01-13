from torch import nn

from torchsparse_20 import SparseTensor
from torchsparse_20.nn.utils import fapply

__all__ = ['ReLU', 'LeakyReLU']


class ReLU(nn.ReLU):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class LeakyReLU(nn.LeakyReLU):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
