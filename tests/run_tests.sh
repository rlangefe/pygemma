#!/bin/bash
#SBATCH --job-name="Tests pyGEMMA"
#SBATCH --partition=mulan
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=30GB
#SBATCH --output=logs/pygemma-tests-%A.o
#SBATCH --error=logs/pygemma-tests-%A.e

cd /net/mulan/home/rlangefe/gemma_work/pygemma/tests

python test_pygemma.py

python gen_comparison.py