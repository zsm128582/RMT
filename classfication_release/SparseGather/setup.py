import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')

setup(
    name='sparse_attention',
    ext_modules=[
            CUDAExtension(
                name='sparse_attention',
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args={'cxx': ['-O2' , '-DTORCH_USE_CUDA_DSA'],
                                    'nvcc': ['-O2' , '-DTORCH_USE_CUDA_DSA']}
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    })
