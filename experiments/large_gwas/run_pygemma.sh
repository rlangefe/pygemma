#!/bin/bash

# Move to working directory
cd /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/large_gwas

# Load Python environment
source /net/mulan/home/rlangefe/gemma_work/pygemma-env/bin/activate

# Vars
DATADIR="/net/mulan/home/rlangefe/gemma_work/parallel_ops/eigenvals/test_data"

# Make output directory
mkdir -p output

# Run the script
echo "Running pyGEMMA"
python run_pygemma.py \
            -g ${DATADIR}/output_x.bin \
            -p ${DATADIR}/output_y.bin \
            -c ${DATADIR}/covars.txt \
            -e ${DATADIR}/eigen.txt \
            -s ${DATADIR}/chr22.txt_snps.csv \
            -o output

# Make output directory
mkdir -p output_base

# Run without SLATE eigendecomposition
echo "Running pyGEMMA base"
python run_pygemma_base.py \
            -g ${DATADIR}/chr22.txt.bin \
            -p ${DATADIR}/BMI.txt \
            -s ${DATADIR}/chr22.txt_snps.csv \
            -o output_base

