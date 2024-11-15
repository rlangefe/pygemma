#!/bin/bash
#SBATCH --job-name="Benchmark pyGEMMA"
#SBATCH --partition=mulan
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --output=logs/pygemma-benchmark-%A-%a.o
#SBATCH --error=logs/pygemma-benchmark-%A-%a.e
#SBATCH --array=1-200%200

TOPDIR="/net/mulan/home/rlangefe/gemma_work"
PYGEMMADIR="${TOPDIR}/pygemma"
TESTDIR="${PYGEMMADIR}/tests"

cd ${TESTDIR}

mkdir -p /net/mulan/home/rlangefe/gemma_work/pygemma/tests/benchmark_test_1
mkdir -p scratch

source "${TOPDIR}/test-env/bin/activate"

python -u benchmark_pygemma_1.py
