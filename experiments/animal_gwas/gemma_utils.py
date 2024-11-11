# Python program to provide utility functions for the pygemma tests
# Allows us to format and run with GEMMA from within the python script

import time
import os

import numpy as np
import pandas as pd
import qnorm

import subprocess

from scipy import stats

# Import what I need to copy files with shutil
import shutil

#GEMMA="/net/mulan/home/rlangefe/gemma_work/modified_gemma/GEMMA/bin/gemma"
GEMMA="/net/fantasia/home/jiaqiang/shiquan_backup/Poisson_Mixed_Model/experiments/methods/LMM/gemma"
GCTA="/net/fantasia/home/borang/software/gcta-1.94.1-linux-kernel-3-x86_64/gcta-1.94.1"
#GEMMA="/net/mulan/home/rlangefe/gemma_work/clean_gemma/GEMMA/bin/gemma"

def run_gemma(gemma_dir, gene_df, pheno_arr, covar_matrix, relatedness_matrix):
    # Make output directory if it doesn't exist
    if not os.path.exists(gemma_dir):
        os.makedirs(gemma_dir)

    # Write data files
    print('Writing data files')
    write_gemma_data(gene_df, pheno_arr, covar_matrix, relatedness_matrix, gemma_dir)

    if not os.path.exists(gemma_dir):
        os.makedirs(gemma_dir)

    # Make GEMMA output dir
    if not os.path.exists(os.path.join(gemma_dir, 'output')):
        os.makedirs(os.path.join(gemma_dir, 'output'))

    # Set working directory to GEMMA dir
    orig_dir = os.getcwd()
    os.chdir(gemma_dir)

    # Run GEMMA
    #GEMMA_COMMAND = f'{GEMMA} -gene genotypes.tsv -p phenotypes.tsv -n 1 -k relatedness_matrix.tsv -notsnp -o output -lmm 1'
    GEMMA_COMMAND = f'{GEMMA} -g genotypes.tsv -p phenotypes.tsv -c covariates.tsv -n 1 -k relatedness_matrix.tsv -notsnp -o output -lmm 1'
    
    print('Running GEMMA Command')
    start = time.time()
    os.system(GEMMA_COMMAND)
    end = time.time()
    total_time = end - start

    # Read in GEMMA output
    gemma_output = pd.read_csv(os.path.join('output', 'output.assoc.txt'), sep='\t')

    # Change directory back to original
    os.chdir(orig_dir)

    # Remove GEMMA output directory
    os.system(f'rm -rf {os.path.join(gemma_dir)}')

    return gemma_output, total_time

def run_gcta(gcta_dir, gene_df, pheno, covar_matrix, relatedness_matrix):
    # gcta command from R gcta_model_cmd<-paste0(gcta_path," --mlma --mbfile ",geno_chrs_file_name," --grm ",grm_path, " --pheno ",GCTA_pheno_name," --qcovar ",GCTA_pc_name," --keep ",idx_name," --extract ",GCTA_SNP_list_name," --thread-num 10 --out ",res_file)
    # Make output directory if it doesn't exist
    if not os.path.exists(gcta_dir):
        os.makedirs(gcta_dir)

    if not os.path.exists(gcta_dir):
        os.makedirs(gcta_dir)

    # Make GEMMA output dir
    if not os.path.exists(os.path.join(gcta_dir, 'output')):
        os.makedirs(os.path.join(gcta_dir, 'output'))

    # Set working directory to GEMMA dir
    orig_dir = os.getcwd()
    os.chdir(gcta_dir)

    threads = 1 if os.getenv("SLURM_CPUS_PER_TASK") is None else int(os.getenv("SLURM_CPUS_PER_TASK"))

    fam_data = pd.read_csv('/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/animal_gwas/imputed_data/mouse_hs1940_imputed.fam', sep='\t', header=None)

    # Subset fam_data to only be 0,1,2,3,4, and pheno
    fam_data = fam_data.iloc[:, [0,1,2,3,4, pheno]]

    maxiter = 1000

    # Copy all /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/animal_gwas/imputed_data/mouse_hs1940* files to current directory
    os.system(f'cp /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/animal_gwas/imputed_data/mouse_hs1940.bim mouse_hs1940.bim')
    os.system(f'cp /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/animal_gwas/imputed_data/mouse_hs1940.bed mouse_hs1940.bed')

    # Write fam_data to file
    fam_data.to_csv('mouse_hs1940.fam', sep='\t', index=False, header=False)

    fam_data.iloc[:, [0,1,5]].to_csv('pheno.tsv', sep='\t', index=False, header=False)

    total_time = np.nan

    max_attempts = 8

    if covar_matrix.shape[1] == 1:
        GCTA_COMMAND = f'{GCTA} --bfile mouse_hs1940 --pheno pheno.tsv --grm /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/animal_gwas/imputed_data/grm --out output --mlma-no-preadj-covar --thread-num {threads} --mlma --reml-maxit {maxiter}'

        print('Running GCTA Command')
        run_attempts = 0

        # Need to run GCTA multiple times because they sometimes have uninvertible matrix from randomly chosen snps
        while run_attempts < max_attempts:
            start = time.time()
            #os.system(GCTA_COMMAND)
            run_output_text = subprocess.run(GCTA_COMMAND, shell=True, check=False, capture_output=True)
            end = time.time()
            total_time = end - start

            print(run_output_text.stdout.decode('utf-8'))

            # Check for string 'Error: Xt_Vi_X is not invertible.' in run_output_text
            # Break if no error
            # Else set runtime to nan
            if ('Error: Xt_Vi_X is not invertible.' in run_output_text.stdout.decode('utf-8')) or ('An error occurs, please check the options or data' in run_output_text.stdout.decode('utf-8')):
                total_time = np.nan
                
                print(f'Failed run {run_attempts}')
                if run_attempts < (max_attempts - 1):
                    print('Trying again...')

            else:
                break

            # Increment run_attempts
            run_attempts = run_attempts + 1
    else:
        # Save covariates using fam_data for FID and IID
        covar_matrix = pd.DataFrame(covar_matrix)
    
        # Remove first column from covar_matrix
        covar_matrix = covar_matrix.iloc[:, 1:]

        # Add FID and IID columns (FID will be first column, IID will be second column)
        covar_matrix.insert(0, 'FID', fam_data.iloc[:, 0])
        covar_matrix.insert(1, 'IID', fam_data.iloc[:, 1])

        # Write covar_matrix to file
        covar_matrix.to_csv('covariates.tsv', sep='\t', index=False, header=False)
        

        # Run GCTA
        #GCTA_COMMAND = f'{GCTA} -g genotypes.tsv -p phenotypes.tsv -c covariates.tsv -n 1 -k relatedness_matrix.tsv -notsnp -o output -lmm 1'
        GCTA_COMMAND = f'{GCTA} --bfile mouse_hs1940 --pheno pheno.tsv --grm /net/mulan/home/rlangefe/gemma_work/pygemma/experiments/animal_gwas/imputed_data/grm --qcovar covariates.tsv --out output --mlma-no-preadj-covar --thread-num {threads} --mlma --reml-maxit {maxiter}'

        print('Running GCTA Command')
        run_attempts = 0

        # Need to run GCTA multiple times because they sometimes have uninvertible matrix from randomly chosen snps
        while run_attempts < max_attempts:
            start = time.time()
            #os.system(GCTA_COMMAND)
            run_output_text = subprocess.run(GCTA_COMMAND, shell=True, check=False, capture_output=True)
            end = time.time()
            total_time = end - start

            print(run_output_text.stdout.decode('utf-8'))

            # Check for string 'Error: Xt_Vi_X is not invertible.' in run_output_text
            # Break if no error
            # Else set runtime to nan
            if ('Error: Xt_Vi_X is not invertible.' in run_output_text.stdout.decode('utf-8')) or ('An error occurs, please check the options or data' in run_output_text.stdout.decode('utf-8')):
                total_time = np.nan
                
                print(f'Failed run {run_attempts}')
                if run_attempts < (max_attempts - 1):
                    print('Trying again...')

            else:
                break

            # Increment run_attempts
            run_attempts = run_attempts + 1

    # Read in GEMMA output
    gcta_output = None #pd.read_csv(os.path.join('output', 'output.assoc.txt'), sep='\t')

    # Change directory back to original
    os.chdir(orig_dir)

    # Remove GEMMA output directory
    os.system(f'rm -rf {os.path.join(gcta_dir)}')

    return gcta_output, total_time

def run_emma(emma_dir, gene_df, pheno_arr, covar_matrix, relatedness_matrix):
    # Make output directory if it doesn't exist
    if not os.path.exists(emma_dir):
        os.makedirs(emma_dir)

    # Write data files
    print('Writing data files')
    write_gemma_data(gene_df, pheno_arr, covar_matrix, relatedness_matrix, emma_dir)

    # Make EMMA output dir
    if not os.path.exists(os.path.join(emma_dir, 'output')):
        os.makedirs(os.path.join(emma_dir, 'output'))

    # Set working directory to EMMA dir
    orig_dir = os.getcwd()
    os.chdir(emma_dir)

    EMMA_SCRIPT = '''
    library(emma)
    library(parallelly)
    library(foreach)
    library(doParallel)

    print("Loading data")
    geno <- read.table("genotypes.tsv", header=FALSE, row.names=1)
    geno <- as.matrix(geno[,3:ncol(geno)])

    pheno <- as.matrix(read.table("phenotypes.tsv", header=FALSE))

    covar <- as.matrix(read.table("covariates.tsv", header=FALSE))

    kinship <- as.matrix(read.table("relatedness_matrix.tsv", header=FALSE))
    
    #output = data.frame(emma.REML.t(pheno, geno, kinship, X0=covar, esp=1e-20))

    # Check if tasks per node set
    # n.cores = 1 if not set
    # else n.cores = tasks per node
    if (Sys.getenv("SLURM_TASKS_PER_NODE") == "") {
        n.cores <- 1
    } else {
        n.cores <- as.numeric(Sys.getenv("SLURM_TASKS_PER_NODE"))

    my.cluster <- parallel::makeCluster(
    max(n.cores-1, 1), 
    type = "PSOCK"
    )

    doParallel::registerDoParallel(cl = my.cluster)

    run_function <- function(r) {
        result <- data.frame(emma.REML.t(pheno, geno[r,], kinship, X0 = covar, esp = 1e-20))
        result$geneID <- rownames(geno)[r]
        
        # Rename ps to p_wald
        colnames(result)[1] <- "p_wald"
        
        # Set p_wald to NA when stat is NA
        result$p_wald[is.na(result$stat)] <- NA
        
        return(result)
    }

    print("Running EMMA")

    # Parallelize code over columns of geno
    output <- foreach(r=1:(nrow(geno)), .combine=rbind, .packages="emma") %dopar% {
        return(run_function(r))
    }

    output$geneID = rownames(geno)

    # Rename ps to p_wald
    colnames(output)[1] = "p_wald"
    
    # Set p_wald to NA when stat is NA
    output$p_wald[is.na(output$stat)] = NA
    
    
    ''' + f'write.csv(output, file="{os.path.join("output", "output.assoc.txt")}", row.names=FALSE)'

    # Write EMMA script
    with open('emma_script.R', 'w') as f:
        f.write(EMMA_SCRIPT)
    
    EMMA_COMMAND = f'Rscript emma_script.R'

    # Run EMMA
    print('Running EMMA Command')
    start = time.time()
    os.system(EMMA_COMMAND)
    end = time.time()
    total_time = end - start

    # Read in EMMA output
    emma_output = pd.read_csv(os.path.join('output', 'output.assoc.txt'))

    # Change directory back to original
    os.chdir(orig_dir)

    # Remove EMMA output directory
    os.system(f'rm -rf {emma_dir}')

    return emma_output, total_time

def write_gemma_data(gene_df,
                    pheno_arr,
                    covar_matrix,
                    relatedness_matrix,
                    output_dir):

    # Make output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write genotypes
    write_genos(gene_df, output_dir)

    # Write phenotypes
    write_phenos(pheno_arr, output_dir)

    # Write covariates
    write_covars(covar_matrix, output_dir)

    # Write relatedness matrix
    write_relatedness(relatedness_matrix, output_dir)

def write_relatedness(relatedness_matrix, output_dir):
    # Write to output file (as tsv)
    # pd.DataFrame(relatedness_matrix).to_csv(os.path.join(output_dir, 'relatedness_matrix.tsv'), 
    #                         sep='\t', 
    #                         index=False, 
    #                         header=False, float_format='%.15f')
    pd.DataFrame(relatedness_matrix).to_csv(os.path.join(output_dir, 'relatedness_matrix.tsv'), 
                            sep='\t', 
                            index=False, 
                            header=False)

def write_covars(covar_matrix, output_dir):
    # Write to output file (as tsv)
    # pd.DataFrame(covar_matrix).to_csv(os.path.join(output_dir, 'covariates.tsv'), 
    #                         sep='\t', 
    #                         index=False, 
    #                         header=False, float_format='%.15f')
    pd.DataFrame(covar_matrix).to_csv(os.path.join(output_dir, 'covariates.tsv'), 
                            sep='\t', 
                            index=False, 
                            header=False)

def write_phenos(pheno_arr, output_dir):
    # Write to output file (as tsv)
    # pd.DataFrame(pheno_arr).to_csv(os.path.join(output_dir, 'phenotypes.tsv'), 
    #                         sep='\t', 
    #                         index=False, 
    #                         header=False, float_format='%.15f')
    pd.DataFrame(pheno_arr).to_csv(os.path.join(output_dir, 'phenotypes.tsv'), 
                            sep='\t', 
                            index=False, 
                            header=False)

def write_genos(gene_df, output_dir):
    # Define gene names
    gene_names = list(gene_df.columns)

    # Transpose df
    gene_df = gene_df.transpose().reset_index()

    # Set column names
    sample_ids = [f'sample{i}' for i in gene_df.columns[1:]]
    gene_df.columns = ['geneID'] + sample_ids

    # Take geneID 7:78753158:G:A and extract 7:78753158, G, and A
    # gene_df['chr'] = gene_df['geneID'].apply(lambda x: x.split(':')[0])
    # gene_df['pos'] = gene_df['geneID'].apply(lambda x: x.split(':')[1])
    # gene_df['ref'] = gene_df['geneID'].apply(lambda x: x.split(':')[2])
    # gene_df['alt'] = gene_df['geneID'].apply(lambda x: x.split(':')[3])
    gene_df['ref'] = ['X'] * len(gene_df)
    gene_df['alt'] = ['Y'] * len(gene_df)


    # gene_df with columns geneID, ref, alt, sample0, sample1, ..., sampleN
    gene_df = gene_df[['geneID', 'ref', 'alt'] + sample_ids]

    # Write to output file (as tsv)
    #gene_df.to_csv(os.path.join(output_dir, 'genotypes.tsv'), sep='\t', index=False, float_format='%.15f')
    gene_df.to_csv(os.path.join(output_dir, 'genotypes.tsv'), sep='\t', index=False, header=False)

