import glob
import os

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

__version__ = '2.0.0b'

if ((torch.cuda.is_available() and CUDA_HOME is not None)
        or (os.getenv('FORCE_CUDA', '0') == '1')):
    device = 'cuda'
else:
    device = 'cpu'

sources = [os.path.join('torchsparse_20', 'backend', f'pybind_{device}.cpp')]
for fpath in glob.glob(os.path.join('torchsparse_20', 'backend', '**', '*')):
    if ((fpath.endswith('_cpu.cpp') and device in ['cpu', 'cuda'])
            or (fpath.endswith('_cuda.cu') and device == 'cuda')):
        sources.append(fpath)

extension_type = CUDAExtension if device == 'cuda' else CppExtension
extra_compile_args = {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
    'nvcc': ['-O3']
}

setup(
    name='torchsparse_20',
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type('torchsparse_20.backend',
                       sources,
                       extra_compile_args=extra_compile_args)
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
