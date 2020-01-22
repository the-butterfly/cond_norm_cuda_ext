from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='xcn_cuda',
    version='0.1.0',
    author='dyxuxu',
    license='Apache License 2.0',
    ext_modules=[
        CUDAExtension(
            name='xcn_cuda', 
            sources=[
            'xcn_cuda.cpp',
            'xcn_cuda_kernel.cu',
            'xcn_cuda_backward_kernel.cu']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
