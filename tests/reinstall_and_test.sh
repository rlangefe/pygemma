#!/bin/bash

TOPDIR="/net/mulan/home/rlangefe/gemma_work"
PYGEMMADIR="${TOPDIR}/pygemma"
TESTDIR="${PYGEMMADIR}/tests"

source "/net/mulan/home/rlangefe/gemma_work/pygemma-env/bin/activate"

cd ${PYGEMMADIR}
pip uninstall -y pygemma

THREAD_COUNT=1

export OPENBLAS_NUM_THREADS=${THREAD_COUNT}
export MKL_NUM_THREADS=${THREAD_COUNT}
export OMP_NUM_THREADS=${THREAD_COUNT}

echo "Running pyGEMMA tests"
python setup.py install && cd ${TESTDIR} && python "${TESTDIR}/test_pygemma.py" && 

echo "Plotting comparison"
python "${TESTDIR}/gen_comparison.py"