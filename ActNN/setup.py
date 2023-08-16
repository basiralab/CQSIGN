from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(name='actnn',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'cpp_extension.calc_precision',
              ['cpp_extension/calc_precision.cc']
          ),
          cpp_extension.CUDAExtension(
              'cpp_extension.minimax',
              ['cpp_extension/minimax.cc', 'cpp_extension/minimax_cuda_kernel.cu']
          ),
          cpp_extension.CUDAExtension(
              'cpp_extension.backward_func',
              ['cpp_extension/backward_func.cc']
          ),
          cpp_extension.CUDAExtension(
              'cpp_extension.quantization',
              ['cpp_extension/quantization.cc', 'cpp_extension/quantization_cuda_kernel.cu']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)
