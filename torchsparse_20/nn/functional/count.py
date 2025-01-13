import torch

import torchsparse_20.backend

__all__ = ['spcount']


def spcount(coords: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
    coords = coords.contiguous()
    if coords.device.type == 'cuda':
        return torchsparse_20.backend.count_cuda(coords, num)
    elif coords.device.type == 'cpu':
        return torchsparse_20.backend.count_cpu(coords, num)
    else:
        device = coords.device
        return torchsparse_20.backend.count_cpu(coords.cpu(), num).to(device)
