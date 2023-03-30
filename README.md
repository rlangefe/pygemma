# pyGEMMA

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contact](#contact)

## Installation
The installation of `pyGEMMA` is straightforward and can be informed using Python's `pip` package manager.
1. Create your Python environment and activate it.
2. Ensure the `Numpy` and `Cython` packages are both installed prior to installing `pyGEMMA` (they will not be installed automatically). This can be done by running 
```bash
pip install numpy Cython
```
3. Clone this repository.
4. Ensure that you have a valid `C/C++` compiler loaded. `pyGEMMA` has been tested using the following compilers:
    - `gcc/g++`
5. Install `pyGEMMA`. From the `pygemma` directory, this can be done by running 
```bash
python setup.py install
```

## Usage
The `pyGEMMA` package contains both high-level and low-level functions for fitting the linear mixed model outlined in the original GEMMA paper by [Zhou et al. (Nat Gen 2012)](https://www.nature.com/articles/ng.2310).



## Examples

## Contact