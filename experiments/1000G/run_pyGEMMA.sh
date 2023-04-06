#!/bin/bash
#SBATCH --job-name="1000G pyGEMMA"
#SBATCH --partition=mulan
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=30GB
#SBATCH --output=logs/pygemma-%j-%a.o
#SBATCH --error=logs/pygemma-%j-%a.e
#SBATCH --array=1-300%10

# Set config variables
TOPDIR="/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/1000G"
GENODIR="/net/fantasia/home/borang/Robert/Gene_Expression"
OUTPUTDIR="/net/mulan/home/rlangefe/gemma_work/1000G_Output"
PYGEMMADIR="/net/mulan/home/rlangefe/gemma_work/pygemma"

# cd to top dir
cd "${TOPDIR}"

# Make output dir if doesn't exist
mkdir -p "${OUTPUTDIR}"

# Configure SNP-running path
PYGEMMA_RUNSNP="${PYGEMMADIR}/experiments/1000G/run_snp.py"

# Get list of geno files for this job array idx
ls "${GENODIR}" | grep "Geno" > "${OUTPUTDIR}/geno_files_${SLURM_ARRAY_TASK_ID}.txt"

NGENES=$(cat "${OUTPUTDIR}/geno_files_${SLURM_ARRAY_TASK_ID}.txt" | wc -l)
NGENES=$(((${NGENES} / ${SLURM_ARRAY_TASK_MAX}) + 1))
echo $NGENES

STARTIDX=$(((${SLURM_ARRAY_TASK_ID} - 1) * ${NGENES}))
ENDIDX=$((${SLURM_ARRAY_TASK_ID} * ${NGENES}))

echo $(head -n ${ENDIDX} "${OUTPUTDIR}/geno_files_${SLURM_ARRAY_TASK_ID}.txt" | tail -n ${NGENES}) > "${OUTPUTDIR}/geno_files_${SLURM_ARRAY_TASK_ID}.txt"

# Loop over each gene expression phenotype matching
for GENOFILE in $(cat "${OUTPUTDIR}/geno_files_${SLURM_ARRAY_TASK_ID}.txt")
do
    # Construct output name
    OUTPUT="${OUTPUTDIR}/$(basename ${GENOFILE%_Geno.txt})"

    echo "Running $(basename ${GENOFILE%_Geno.txt})"

    # Make output directory for phenotype
    mkdir -p "${OUTPUT}"

    python "${PYGEMMA_RUNSNP}" \
            -s "${GENODIR}/${GENOFILE}" \
            -p "${GENODIR}/${GENOFILE%_Geno.txt}_Gene.txt" \
            -k /net/fantasia/home/borang/Robert/Genotype/output/chr_all.sXX.txt \
            -pcs 5 \
            --pcfile=/net/fantasia/home/borang/Robert/Genotype/chr_all_pc.eigenvec \
            -o "${OUTPUT}"
done

rm "${OUTPUTDIR}/geno_files_${SLURM_ARRAY_TASK_ID}.txt"
