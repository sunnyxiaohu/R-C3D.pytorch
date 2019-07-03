from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_temporal_pool',
    ext_modules=[
        CUDAExtension('roi_temporal_pool_cuda', [
            'src/roi_temporal_pool_cuda.cpp',
            'src/roi_temporal_pool_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})
