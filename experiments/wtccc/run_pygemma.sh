#!/bin/bash
#SBATCH --job-name="pyGEMMA WTCCC"
#SBATCH --partition=main
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --output=logs/pygemma-benchmark-%A.o
#SBATCH --error=logs/pygemma-benchmark-%A.e
#SBATCH --exclude=r630[1-9],r6316,r6326,r6321,r6319,r6328,r6329,r6330,r6331


source /net/mulan/home/rlangefe/gemma_work/pygemma-env/bin/activate

export OUTPUT="/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/updated_output_no_pcs"
#export OUTPUT="/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/updated_output"
#export OUTPUT="/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/linear_output"
export PCS=0
export LINEAR=0

# Create output directory if it does not exist
mkdir -p $OUTPUT

/usr/bin/time -v python -u run_pygemma_imputed.py
#python -u run_pygemma_imputed.py