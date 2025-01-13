import torch
from torch import nn

from torchsparse_20 import SparseTensor
from torchsparse_20.nn import functional as F

__all__ = ['GlobalAvgPool', 'GlobalMaxPool']


class GlobalAvgPool(nn.Module):

    def forward(self, input: SparseTensor) -> torch.Tensor:
        return F.global_avg_pool(input)


class GlobalMaxPool(nn.Module):

    def forward(self, input: SparseTensor) -> torch.Tensor:
        return F.global_max_pool(input)
