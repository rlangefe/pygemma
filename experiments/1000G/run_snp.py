import time
import os

import numpy as np
import pandas as pd
import qnorm

import argparse

from rich.console import Console

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import warnings

sns.set_theme()

from pygemma import lmm

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
    X = snp_df.values[:,:100]
    snps = snp_df.columns[:100]
    X = (X - X.mean(axis=0)) #/ X.std(axis=0)
    p = X.shape[1]
    del snp_df

    # Read phenotypes
    print('Reading in phenotypes...')
    phenotype_df = pd.read_csv(args.phenotype)
    Y = phenotype_df['Exp_Value'].values#.reshape(-1,1)
    #Y = qnorm.quantile_normalize(Y, axis=1)
    Y = Y - Y.mean()
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

    print('Launching pyGEMMA...')
    data_results = lmm.pygemma(Y, X, W, K, snps=snps, verbose=1)

    data_results.to_csv(os.path.join(args.output, 'pygemma_results.csv'), index=False)

    theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
    pvals = np.sort(data_results['p_wald'])

    plt.scatter(y=-np.log10(pvals+1e-20), x=-np.log10(theoretical))
    plt.axline((0,0), slope=1, color='red')
    plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
    plt.ylabel(r'Observed: $-\log_{10}(p)$')
    plt.title(f'QQ Plot - {os.path.splitext(os.path.basename(args.phenotype))[0][:-5]}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "wald_qq.png"))
    plt.clf()

    # Parse SNP column where SNP is in the format: chr:pos:ref:alt
    snps_values = data_results['SNPs'].str.split(':', expand=True)
    snps_values.columns = ['CHR', 'POS', 'REF', 'ALT']


    # Manhattan plot adapted from https://stackoverflow.com/a/66062857
    results_df = pd.DataFrame(
        {
        'pos'  : snps_values['POS'].values,
        'pval' : -np.log10(data_results['p_wald']+1e-20),
        'chr' : snps_values['CHR'].values
        }
    )

    results_df = results_df.sort_values(['chr', 'pos'])
    results_df.reset_index(inplace=True, drop=True)
    results_df['i'] = results_df.index

    alpha = -np.log10(0.05/len(pvals))
    with sns.color_palette():
        sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'])

    # Annotate snp with small font if results_df['pval'] above alpha
    for i, row in results_df.iterrows():
        if row['pval'] > alpha:
            plt.annotate(str(row['chr']) + ':' + str(row['pos']), (row['i'], row['pval']), fontsize=6)

    plt.axline((0,alpha), slope=0, color='red')
    chrom_df=results_df.groupby('chr')['i'].median()
    plt.xlabel('chr') 
    plt.xticks(chrom_df,chrom_df.index)
    plt.ylabel(r'$-\log_{10}(p)$')
    plt.title(f'Manhattan Plot - {os.path.splitext(os.path.basename(args.phenotype))[0][:-5]}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "wald_Manhattan.png"))
    plt.clf()
    