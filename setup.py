from setuptools import setup, find_packages, Extension
import subprocess

from Cython.Build import cythonize
import numpy as np

# Parse requirements.txt
with open('requirements.txt') as f:
    requirements = f.readlines()

extensions = [
    Extension('pygemma.pygemma_model',
              sources=['pygemma/pygemma_model.pyx'],
              include_dirs=[np.get_include()])
]

setup(
    name='pygemma',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=cythonize(extensions)
)