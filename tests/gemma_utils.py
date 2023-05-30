# Python program to provide utility functions for the pygemma tests
# Allows us to format and run with GEMMA from within the python script

import time
import os

import numpy as np
import pandas as pd
import qnorm

from scipy import stats

GEMMA="/net/mulan/home/rlangefe/gemma_work/modified_gemma/GEMMA/bin/gemma"
#GEMMA="/net/fantasia/home/jiaqiang/shiquan_backup/Poisson_Mixed_Model/experiments/methods/LMM/gemma"

def run_gemma(gemma_dir, gene_df, pheno_arr, covar_matrix, relatedness_matrix):
    # Make output directory if it doesn't exist
    if not os.path.exists(gemma_dir):
        os.makedirs(gemma_dir)

    # Write data files
    write_gemma_data(gene_df, pheno_arr, covar_matrix, relatedness_matrix, gemma_dir)

    # Make GEMMA output dir
    if not os.path.exists(os.path.join(gemma_dir, 'output')):
        os.makedirs(os.path.join(gemma_dir, 'output'))

    # Set working directory to GEMMA dir
    orig_dir = os.getcwd()
    os.chdir(gemma_dir)

    # Run GEMMA
    GEMMA_COMMAND = f'{GEMMA} -gene genotypes.tsv -p phenotypes.tsv -c covariates.tsv -n 1 -k relatedness_matrix.tsv -notsnp -o output -lmm 1'
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
    gene_df.columns = ['geneID'] + [f'sample{i}' for i in gene_df.columns[1:]]

    # Write to output file (as tsv)
    #gene_df.to_csv(os.path.join(output_dir, 'genotypes.tsv'), sep='\t', index=False, float_format='%.15f')
    gene_df.to_csv(os.path.join(output_dir, 'genotypes.tsv'), sep='\t', index=False)

