#!/bin/bash
#SBATCH --job-name="Benchmark pyGEMMA"
#SBATCH --partition=mulan
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --output=logs/pygemma-benchmark-%A-%a.o
#SBATCH --error=logs/pygemma-benchmark-%A-%a.e
#SBATCH --array=1-990%30

TOPDIR="/net/mulan/home/rlangefe/gemma_work"
PYGEMMADIR="${TOPDIR}/pygemma"
TESTDIR="${PYGEMMADIR}/experiments/benchmarks"

cd ${TESTDIR}

# # Testing only
# export SLURM_ARRAY_TASK_ID=1
# export SLURM_JOB_ID=8
# export SLURM_ARRAY_TASK_COUNT=2
# export SLURM_CPUS_PER_TASK=2

# Set threads
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

#mkdir -p /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/benchmark_test
mkdir -p /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/benchmark_grid_test
mkdir -p scratch

source "${TOPDIR}/pygemma-env/bin/activate"

python "${TESTDIR}/benchmarks.py"
