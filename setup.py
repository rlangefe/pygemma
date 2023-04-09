from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
import numpy as np
import scipy

# Parse requirements.txt
with open('requirements.txt') as f:
    requirements = f.readlines()

extensions = [
    Extension('pygemma.pygemma_model',
              sources=['pygemma_model/pygemma_model.pyx'],
              include_dirs=[np.get_include(), scipy.get_include()],
              extra_compile_args=["-O3"]),

    Extension('pygemma.lmm',
              sources=['lmm/lmm.py'],
              include_dirs=[np.get_include(), scipy.get_include()],
              extra_compile_args=["-O3"])
]

setup(
    name='pygemma',
    version='0.1',
    #packages=['pygemma', 'pygemma.lmm', 'pygemma.pygemma_model'], # Rough fix for now. Should use setuptools.find_packages()
    packages=find_packages(),
    setup_requires=['Cython', 'numpy'],
    install_requires=requirements,
    ext_modules=cythonize(extensions)
)