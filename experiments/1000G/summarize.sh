#!/bin/bash
#SBATCH --job-name="Summarize pyGEMMA"
#SBATCH --partition=mulan
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --mail-type=ALL
#SBATCH --output=logs/pygemma-summary-%A.o
#SBATCH --error=logs/pygemma-summary-%A.e

source /net/mulan/home/rlangefe/gemma_work/test-env/bin/activate

cd /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/1000G

python summary.py \
        -d /net/mulan/home/rlangefe/gemma_work/1000G_Output_test_parallel \
        -o /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/1000G/summary_output