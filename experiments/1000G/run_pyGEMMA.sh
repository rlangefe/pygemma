#!/bin/bash
#SBATCH --job-name="1000G pyGEMMA"
#SBATCH --partition=mulan
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mem=30GB
#SBATCH --output=logs/pygemma-%A-%a.o
#SBATCH --error=logs/pygemma-%A-%a.e
#SBATCH --array=1-400%30

# Set config variables
TOPDIR="/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/1000G"
GENODIR="/net/fantasia/home/borang/Robert/Gene_Expression"
#OUTPUTDIR="/net/mulan/home/rlangefe/gemma_work/1000G_Output"
OUTPUTDIR="/net/mulan/home/rlangefe/gemma_work/1000G_Output_test_parallel"
PYGEMMADIR="/net/mulan/home/rlangefe/gemma_work/pygemma"
GEMMA="/net/fantasia/home/jiaqiang/shiquan_backup/Poisson_Mixed_Model/experiments/methods/LMM/gemma"
PCFILE="/net/fantasia/home/borang/Robert/Genotype/chr_all_pc.eigenvec"
RELATEDNESSMATRIX="/net/fantasia/home/borang/Robert/Genotype/output/chr_all.sXX.txt"
PYGEMMAFIXPHENO="${PYGEMMADIR}/experiments/1000G/fix_pheno.py"
PLOTGEMMA="${PYGEMMADIR}/experiments/1000G/plot_gemma.py"
PLOTGEMMALINEAR="${PYGEMMADIR}/experiments/1000G/plot_gemma_linear.py"
NPCS=5

# Python environment
source "/net/mulan/home/rlangefe/gemma_work/test-env/bin/activate"

# cd to top dir
cd "${TOPDIR}"

# Make output dir if doesn't exist
mkdir -p "${OUTPUTDIR}"

# Configure SNP-running path
PYGEMMA_RUNSNP="${PYGEMMADIR}/experiments/1000G/run_snp.py"
LINREG_RUNSNP="${PYGEMMADIR}/experiments/1000G/run_lin_reg.py"

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

    # # Run Fixed Effect Linear Regression
    python "${LINREG_RUNSNP}" \
            -s "${GENODIR}/${GENOFILE}" \
            -p "${GENODIR}/${GENOFILE%_Geno.txt}_Gene.txt" \
            -k "${RELATEDNESSMATRIX}" \
            -pcs "${NPCS}" \
            --pcfile=${PCFILE} \
            -o "${OUTPUT}"

    # # Run pyGEMMA
    python "${PYGEMMA_RUNSNP}" \
            -s "${GENODIR}/${GENOFILE}" \
            -p "${GENODIR}/${GENOFILE%_Geno.txt}_Gene.txt" \
            -k "${RELATEDNESSMATRIX}" \
            -pcs "${NPCS}" \
            --pcfile=${PCFILE} \
            --nproc=${SLURM_CPUS_PER_TASK} \
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

    # Modify phenotype
    python "${PYGEMMAFIXPHENO}" \
                -i "${GENODIR}/${GENOFILE%_Geno.txt}_Gene.txt" \
                -o "${OUTPUT}/gemma_run/pheno.tsv"

    # Start timer
    STARTTIME=$(date +%s)

    ${GEMMA} \
        -gene "${OUTPUT}/gemma_run/geno.tsv" \
        -p "${OUTPUT}/gemma_run/pheno.tsv" \
        -c "${OUTPUT}/gemma_run/pcs.txt" \
        -n 1 \
        -k "${RELATEDNESSMATRIX}" \
        -lmm 1
    
    # End timer
    ENDTIME=$(date +%s)
    echo "GEMMA LMM took $(($ENDTIME - $STARTTIME)) seconds to run"

    mv "${OUTPUT}/gemma_run/output/result.assoc.txt" "${OUTPUT}/gemma_results.tsv"

    python ${PLOTGEMMA} \
        -i "${OUTPUT}/gemma_results.tsv" \
        -o "${OUTPUT}"

    cd $OUTPUT

    # Cleanup
    rm -rf "${OUTPUT}/gemma_run"

    # Run GEMMA for linear regression

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

    # Modify phenotype
    python "${PYGEMMAFIXPHENO}" \
                -i "${GENODIR}/${GENOFILE%_Geno.txt}_Gene.txt" \
                -o "${OUTPUT}/gemma_run/pheno.tsv"

    # Start timer
    STARTTIME=$(date +%s)
    ${GEMMA} \
        -gene "${OUTPUT}/gemma_run/geno.tsv" \
        -p "${OUTPUT}/gemma_run/pheno.tsv" \
        -c "${OUTPUT}/gemma_run/pcs.txt" \
        -n 1 \
        -lm

    # End timer
    ENDTIME=$(date +%s)
    echo "GEMMA LM took $(($ENDTIME - $STARTTIME)) seconds to run"

    mv "${OUTPUT}/gemma_run/output/result.assoc.txt" "${OUTPUT}/gemma_results_linear.tsv"

    python ${PLOTGEMMALINEAR} \
        -i "${OUTPUT}/gemma_results_linear.tsv" \
        -o "${OUTPUT}"

    cd $OUTPUT

    # Cleanup
    rm -rf "${OUTPUT}/gemma_run"

done

rm "${OUTPUTDIR}/geno_files_${SLURM_ARRAY_TASK_ID}.txt"
