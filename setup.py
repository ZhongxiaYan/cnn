from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        'convolve',
        ['convolve.pyx'],
        extra_compile_args=['-O3', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)