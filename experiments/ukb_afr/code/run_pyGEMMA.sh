#!/bin/bash
#SBATCH --job-name="UKB pyGEMMA"
#SBATCH --partition=mulan
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --output=logs/pygemma-ukb-%A.o
#SBATCH --error=logs/pygemma-ukb-%A.e

TOPDIR="/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/ukb_afr"
OUTPUT="/net/mulan/home/rlangefe/gemma_work/UKB_AFR_Output"
DATADIR="/net/fantasia/home/borang/Robert/UKB_AFR"
COVARIATESFILE="${DATADIR}/Trait/african_ancestry_smps_7894_info.txt"
PHENOTYPEFILE="${DATADIR}/Trait/Continuous_Trait.csv"
PYGEMMADIR="/net/mulan/home/rlangefe/gemma_work/pygemma"

NPCS=10

# Python environment
source "/net/mulan/home/rlangefe/gemma_work/test-env/bin/activate"

# cd to top dir
cd "${TOPDIR}"

# Make output dir if doesn't exist
mkdir -p "${OUTPUT}"

# Configure SNP-running path
PYGEMMA_RUNSNP="${TOPDIR}/code/run_snp.py"

# Run pyGEMMA
# python "${PYGEMMA_RUNSNP}" \
#         -p "${PHENOTYPEFILE}" \
#         -pcs "${NPCS}" \
#         -c "${COVARIATESFILE}" \
#         --nproc=2 \
#         -o "${OUTPUT}"
python "${PYGEMMA_RUNSNP}" \
        -p "${PHENOTYPEFILE}" \
        -pcs "${NPCS}" \
        -c "${COVARIATESFILE}" \
        --nproc=${SLURM_CPUS_PER_TASK} \
        -o "${OUTPUT}"
