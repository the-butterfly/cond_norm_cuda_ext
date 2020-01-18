from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='xcn_cuda',
    ext_modules=[
        CUDAExtension('xcn_cuda', [
            'xcn_cuda.cpp',
            'xcn_cuda_kernel.cu',
            'xcn_cuda_backward_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
