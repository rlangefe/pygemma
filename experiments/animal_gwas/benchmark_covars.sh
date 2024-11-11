#!/bin/bash
#SBATCH --job-name="pyGEMMA Covars"
#SBATCH --partition=mulan
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --output=logs/pygemma-benchmark-%A-%a.o
#SBATCH --error=logs/pygemma-benchmark-%A-%a.e
#SBATCH --array=0-15

python -u benchmark_pygemma.py