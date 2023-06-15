#!/bin/bash

TOPDIR="/net/mulan/home/rlangefe/gemma_work"
PYGEMMADIR="${TOPDIR}/pygemma"
TESTDIR="${PYGEMMADIR}/tests"

source "${TOPDIR}/test-env/bin/activate"

cd ${PYGEMMADIR}
pip uninstall -y pygemma

echo "Running pyGEMMA tests"
python setup.py install && cd ${TESTDIR}
#python -m cProfile -o pygemma.prof "${TESTDIR}/profile_pygemma.py"
#python "${TESTDIR}/profile_pygemma.py"
python -m cProfile -o pygemma.pstats "${TESTDIR}/profile_pygemma.py"

# pyinstrument -r html -o output.html "${TESTDIR}/profile_pygemma.py"

# gprof2dot --colour-nodes-by-selftime -f pstats  pygemma.pstats | \
#     dot -Tpng -o profiling.png

echo "Plotting comparison"
python "${TESTDIR}/gen_comparison.py"