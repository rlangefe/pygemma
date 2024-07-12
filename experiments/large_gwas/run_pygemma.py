import re
import time
import os

import numpy as np
import pandas as pd
import qnorm

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

sns.set_theme()

from pygemma import lmm, pygemma_model, plot

import argparse

np.set_printoptions(formatter={'float': lambda x: f"{x:10.4g}"}, suppress=True, linewidth=np.nan)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write geno file for GEMMA')
    parser.add_argument('-g', '--geno', type=str, help='Genotype file (binary)', required=True)
    parser.add_argument('-p', '--pheno', type=str, help='Phenotype file (binary)', required=True)
    parser.add_argument('-c', '--covar', type=str, help='Covariates file (binary)', required=True)
    parser.add_argument('-e', '--eigen', type=str, help='Eigenvalues file (text)', required=True)
    parser.add_argument('-s', '--snps', type=str, help='SNPs file (csv)', default=None)
    parser.add_argument('-o', '--output', type=str, help='Output directory', default='output')

    args = parser.parse_args()

    print('Reading data...')
    X = np.fromfile(args.geno, dtype=np.float32)
    Y = np.fromfile(args.pheno, dtype=np.float32)
    W = np.fromfile(args.covar, dtype=np.float32)


    # Reshape
    Y = Y.reshape(-1, 1)
    X = X.reshape(Y.shape[0], -1)
    W = W.reshape(Y.shape[0], -1)

    # Read eigenvalues
    #eigenvals = np.loadtxt(args.eigen).reshape(-1)
    eigenvals = pd.read_csv(args.eigen, sep=' ', header=None, engine="c").values.astype(np.float32).reshape(-1)

    print('Samples:', Y.shape[0])
    print('SNPs:', X.shape[1])

    if args.snps is not None:
        # Cols are CHROM,POS,ID,REF,ALT,QUAL,FILTER,INFO,FORMAT
        snps = pd.read_csv(args.snps)

    print('Running pyGEMMA...')
    with np.errstate(over="ignore"):
        data_results = lmm.pygemma(Y=Y, 
                                   X=X,
                                   W=W,
                                   K=eigenvals, 
                                   snps=range(X.shape[1]), 
                                   verbose=1, 
                                   nproc=4, 
                                   grid=False,
                                   eigen=False)

    if args.snps is not None:
        # Add CHR, POS, ID, REF, and ALT to results
        # Change ID to rs
        data_results['CHR'] = snps['CHROM']
        data_results['POS'] = snps['POS']
        data_results['rs'] = snps['ID']
        data_results['REF'] = snps['REF']
        data_results['ALT'] = snps['ALT']

        # Put those columns first
        cols = data_results.columns.tolist()
        cols = cols[-5:] + cols[:-5]
        data_results = data_results[cols]

        print('Creating Manhattan plot...')
        plot.manhattan_plot(pval=data_results['p_wald'], 
                            pos=data_results['POS'], 
                            chrom=data_results['CHR'],
                            cutoff='bonferroni',
                            scale='log',
                            save_path=os.path.join(args.output, 'manhattan.png'))

    print('Writing results...')
    pd.DataFrame(data_results).to_csv(os.path.join(args.output, 'output.txt'), 
                                       sep=' ',
                                       index=False)

