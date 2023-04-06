#!/bin/bash
#SBATCH --job-name="1000G pyGEMMA"
#SBATCH --partition=mulan
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=30GB
#SBATCH --output=logs/pygemma-%j-%a.o
#SBATCH --error=logs/pygemma-%j-%a.e
#SBATCH --array=1-400%20

# Set config variables
TOPDIR="/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/1000G"
GENODIR="/net/fantasia/home/borang/Robert/Gene_Expression"
OUTPUTDIR="/net/mulan/home/rlangefe/gemma_work/1000G_Output"
PYGEMMADIR="/net/mulan/home/rlangefe/gemma_work/pygemma"
PCFILE="/net/fantasia/home/borang/Robert/Genotype/chr_all_pc.eigenvec"
RELATEDNESSMATRIX="/net/fantasia/home/borang/Robert/Genotype/output/chr_all.sXX.txt"
PYGEMMAFIXPHENO="${PYGEMMADIR}/experiments/1000G/fix_pheno.py"
PLOTGEMMA="${PYGEMMADIR}/experiments/1000G/plot_gemma.py"
NPCS=5

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
            -k "${RELATEDNESSMATRIX}" \
            -pcs ${NPCS} \
            --pcfile=${PCFILE} \
            -o "${OUTPUT}"

    # Make Gemma run directory
    mkdir -p "${OUTPUT}/gemma_run"
    cd "${OUTPUT}/gemma_run"

    python "${PYGEMMADIR}/experiments/1000G/fix_pcs.py" \
                -i "${PCFILE}" \
                -pcs ${NPCS} \
                -o "${OUTPUT}/gemma_run/pcs.txt"

    # Transpose genotypes
    python "${PYGEMMADIR}/experiments/1000G/transpose.py" \
            -i "${GENODIR}/${GENOFILE}" \
            -o "${OUTPUT}/gemma_run/geno.tsv"

    python "${PYGEMMAFIXPHENO}" \
                -i "${GENODIR}/${GENOFILE%_Geno.txt}_Gene.txt" \
                -o "${OUTPUT}/gemma_run/pheno.tsv"

    /net/fantasia/home/jiaqiang/shiquan_backup/Poisson_Mixed_Model/experiments/methods/LMM/gemma \
        -gene "${OUTPUT}/gemma_run/geno.tsv" \
        -p "${OUTPUT}/gemma_run/pheno.tsv" \
        -c "${OUTPUT}/gemma_run/pcs.txt" \
        -n 1 \
        -k "${RELATEDNESSMATRIX}" \
        -lmm

    mv "${OUTPUT}/gemma_run/output/result.assoc.txt" "${OUTPUT}/gemma_results.tsv"

    python ${PLOTGEMMA} \
        -i "${OUTPUT}/gemma_results.tsv" \
        -o "${OUTPUT}"

    cd $OUTPUT

    # Cleanup
    rm -rf "${OUTPUT}/gemma_run"

done

rm "${OUTPUTDIR}/geno_files_${SLURM_ARRAY_TASK_ID}.txt"
