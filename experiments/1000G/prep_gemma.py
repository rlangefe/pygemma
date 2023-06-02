# Python program to provide utility functions for the pygemma tests
# Allows us to format and run with GEMMA from within the python script

import time
import os

import numpy as np
import pandas as pd
import qnorm

from scipy import stats

import argparse

from rich.console import Console

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import warnings

sns.set_theme()

#GEMMA="/net/mulan/home/rlangefe/gemma_work/modified_gemma/GEMMA/bin/gemma"
#GEMMA="/net/fantasia/home/jiaqiang/shiquan_backup/Poisson_Mixed_Model/experiments/methods/LMM/gemma"
GEMMA="/net/mulan/home/rlangefe/gemma_work/clean_gemma/GEMMA/bin/gemma"

def run_gemma(gemma_dir, gene_df, pheno_arr, covar_matrix, relatedness_matrix):
    # Make output directory if it doesn't exist
    if not os.path.exists(gemma_dir):
        os.makedirs(gemma_dir)

    # Write data files
    print('Writing data files')
    write_gemma_data(gene_df, pheno_arr, covar_matrix, relatedness_matrix, gemma_dir)

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

    n.cores <- parallelly::availableCores()
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
    gene_df['chr'] = gene_df['geneID'].apply(lambda x: x.split(':')[0])
    gene_df['pos'] = gene_df['geneID'].apply(lambda x: x.split(':')[1])
    gene_df['ref'] = gene_df['geneID'].apply(lambda x: x.split(':')[2])
    gene_df['alt'] = gene_df['geneID'].apply(lambda x: x.split(':')[3])


    # gene_df with columns geneID, ref, alt, sample0, sample1, ..., sampleN
    gene_df = gene_df[['geneID', 'ref', 'alt'] + sample_ids]

    # Write to output file (as tsv)
    #gene_df.to_csv(os.path.join(output_dir, 'genotypes.tsv'), sep='\t', index=False, float_format='%.15f')
    gene_df.to_csv(os.path.join(output_dir, 'genotypes.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--snps", dest="snps", help="Path to snps", type=str, default="snps.csv")
    parser.add_argument("-p", "--phenotype", dest="phenotype", help="Path to phenotype", type=str, default="phenotypes.csv")
    parser.add_argument("-c", "--covars", dest="covars", help="Path to covars", type=str, default=None)
    parser.add_argument("-pcs", "--pcs", dest="pcs", help="Number of PCs", type=int, default=2)
    parser.add_argument("-pcf", "--pcfile", dest="pcfile", help="File containing PCs", type=str, default=None)
    parser.add_argument('-k', '--kinship', dest='kinship', help='Path to kinship matrix', type=str, default=None)
    parser.add_argument("-o", "--output", dest="output", help="Path to output file", type=str, default="output_file.csv")
    args = parser.parse_args()

    # Read in SNPs
    print('Reading in SNPs...')
    snp_df = pd.read_csv(args.snps)
    X = snp_df.values
    snps = snp_df.columns
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    p = X.shape[1]
    del snp_df

    # Read phenotypes
    print('Reading in phenotypes...')
    phenotype_df = pd.read_csv(args.phenotype)
    Y = phenotype_df['Exp_Value'].values.reshape(-1,1)
    Y = qnorm.quantile_normalize(Y, axis=1)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    del phenotype_df

    W = np.ones(shape=(X.shape[0], 1))

    # Read covariates
    if args.covars:
        print('Reading in covariates...')
        covars_df = pd.read_csv(args.covars, sep='\t')
        W = np.c_[W, covars_df.values]
        del covars_df

    if args.kinship:
        print('Reading in kinship matrix...')
        kinship_df = pd.read_csv(args.kinship, sep='\t', header=None)
        K = kinship_df.values
        del kinship_df
    else:
        print('Computing kinship matrix...')
        K = X @ X.T / p       

    if int(args.pcs) > 0:
        if args.pcfile:
            print('Reading in PCs...')
            pcs_df = pd.read_csv(args.pcfile, sep='\t')
            pcs = pcs_df.values[:, 2:2+int(args.pcs)]
            W = np.c_[W, pcs]
            del pcs_df
        else:
            print('Running PCA...')
            pca = PCA(n_components=int(args.pcs))

            pcs = pca.fit_transform(X)

            W = np.c_[W, pcs]

    X = pd.DataFrame(X, columns=snps)
    write_gemma_data(X, Y, W, K, args.output)
