#!/bin/bash

TOPDIR="/net/mulan/home/rlangefe/gemma_work"
PYGEMMADIR="${TOPDIR}/pygemma"
TESTDIR="${PYGEMMADIR}/tests"

source "${TOPDIR}/test-env/bin/activate"

cd ${PYGEMMADIR}
pip uninstall -y pygemma

echo "Running pyGEMMA tests"
python setup.py install && cd ${TESTDIR} && python "${TESTDIR}/test_pygemma.py" && 

echo "Plotting comparison"
python "${TESTDIR}/gen_comparison.py"