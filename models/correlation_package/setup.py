#!/usr/bin/env python3
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++17']

nvcc_args = [
    '-gencode', 'arch=compute_75,code=sm_75',    # Turing (e.g., RTX 2080)
    '-gencode', 'arch=compute_80,code=sm_80',    # Ampere (e.g., A100, RTX 30xx)
    '-gencode', 'arch=compute_86,code=sm_86',    # Ampere (e.g., RTX 3090, Jetson Orin)
    '-gencode', 'arch=compute_89,code=sm_89',    # Ada (e.g., RTX 40xx, optional)
    '-gencode', 'arch=compute_90,code=sm_90'     # Hopper (e.g., H100, optional)
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
