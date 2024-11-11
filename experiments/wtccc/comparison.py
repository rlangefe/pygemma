from turtle import color
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns

from rich.progress import track

import os

if __name__ == '__main__':
    OUTPUT = '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/comparison'
    gemma_dir = '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/gemma_runs/output'
    pygemma_dir = '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/updated_output_no_pcs'
    #pygemma_dir = '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/updated_output'
    #pygemma_dir = '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/linear_output'

    # Extract matplotlib colors in tableau-colorblind10
    sns.set_theme()
    colors = sns.color_palette('tab10')

    pheno_list = [
            {
                'name' : 'Rhumatoid Arthritis', 
                'gemma' : 'ra_output.assoc.txt',
                'pygemma' : 'RA_RA_pygemma_results.csv'
            },
            {
                'name' : 'Type 1 Diabetes',
                'gemma' : 't1d_output.assoc.txt',
                'pygemma' : 'T1D_T1D_pygemma_results.csv'
            },
            {
                'name' : 'Type 2 Diabetes',
                'gemma' : 't2d_output.assoc.txt',
                'pygemma' : 'T2D_T2D_pygemma_results.csv'
            },
            {
                'name' : 'Bipolar Disorder',
                'gemma' : 'bd_output.assoc.txt',
                'pygemma' : 'BD_BD_pygemma_results.csv'
            },
            {
                'name' : 'Hypertension',
                'gemma' : 'ht_output.assoc.txt',
                'pygemma' : 'HT_HT_pygemma_results.csv'
            },
            {
                'name' : 'Coronary Artery Disease',
                'gemma' : 'cad_output.assoc.txt',
                'pygemma' : 'CAD_CAD_pygemma_results.csv'
            }
        ]
    
    snp_info = pd.read_csv('/net/mulan/data/WTCCC1/snpinfo/snpinfo.txt.gz', 
                               sep='\t',
                               header=None)
    
    snp_info.columns = ['CHR', 'SNP', 'rs', 'POS', 'ref', 'alt']

    # List rs numbers that are in snp_info more than one time
    duplicate_rs = snp_info[snp_info['rs'].duplicated(keep=False)]['rs'].unique()

    # Make output directory if it doesn't exist
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    # Make figures with 2 rows and 3 columns
    fig_pval, axes_pval = plt.subplots(2, 3, figsize=(15, 10))
    axes_pval = axes_pval.flatten()

    fig_beta, axes_beta = plt.subplots(2, 3, figsize=(15, 10))
    axes_beta = axes_beta.flatten()

    for i, pheno in track(list(enumerate(pheno_list))):
        # Read in GEMMA data
        gemma_df = pd.read_csv(os.path.join(gemma_dir, pheno['gemma']), sep='\t', engine='c')
        
        # Read in PyGEMMA data
        pygemma_df = pd.read_csv(os.path.join(pygemma_dir, pheno['pygemma']), engine='c')

        # Map pyGEMMA SNPs to rs numbers
        # if pheno['name'] == 'Bipolar Disorder':
        #     pygemma_df = pygemma_df.merge(snp_info, left_on='SNPs', right_on='SNP')
        # else:
        #     pygemma_df = pygemma_df.merge(snp_info, left_on='SNPs', right_on='rs')

        pygemma_df = pygemma_df.merge(snp_info, left_on='SNPs', right_on='SNP')
        
        # # Rename rs column to SNP
        # gemma_df = gemma_df.rename(columns={'rs': 'SNP'})

        # gemma_df = gemma_df.merge(snp_info, left_on='SNP', right_on='SNP')


        df = gemma_df.merge(pygemma_df, 
                            left_on='rs', 
                            right_on='SNPs', 
                            suffixes=('_gemma', '_pygemma'))
        
        
        # Remove duplicate rs numbers
        #df = df[~df['rs'].isin(duplicate_rs)]

        #print(df[['chr', 'rs', 'beta_gemma', 'se', 'p_wald_gemma', 'beta_pygemma', 'se_beta', 'p_wald_pygemma']].head())

        # Print number of remaining SNPs
        print(f'{pheno["name"]}: {df.shape[0]} SNPs')

        # Clip p-values at 1e-300
        df['p_wald_gemma'] = df['p_wald_gemma'].clip(lower=1e-300)
        df['p_wald_pygemma'] = df['p_wald_pygemma'].clip(lower=1e-300)

        # Plot p-values for GEMMA vs pyGEMMA with all points colored as current phenotype in 'colors'
        sns.scatterplot(x=-np.log10(df['p_wald_gemma']), 
                        y=-np.log10(df['p_wald_pygemma']), 
                        ax=axes_pval[i], 
                        color=colors[i])
        
        # Put x and y axes on the same scale
        axes_pval[i].set_aspect('equal')

        # # Log scale for both axes
        # axes_pval[i].set_xscale('log')
        # axes_pval[i].set_yscale('log')
        
        # add a diagonal line
        axes_pval[i].axline([0, 0], [1, 1], color='k', linestyle='--')

        # Set y axis label if it's the first column
        if i % 3 == 0:
            axes_pval[i].set_ylabel(r'pyGEMMA $-log_{10}(p)$', fontsize=16)
        else:
            axes_pval[i].set_ylabel('', fontsize=16)

        # Set x axis label if it's the last row
        if i > 2:
            axes_pval[i].set_xlabel(r'GEMMA $-log_{10}(p)$', fontsize=16)
        else:
            axes_pval[i].set_xlabel('', fontsize=16)

        # Compute correlation between GEMMA and pyGEMMA p-values
        corr = -np.log10(df['p_wald_gemma']).corr(-np.log10(df['p_wald_pygemma']))

        # Set title
        axes_pval[i].set_title(pheno['name'] + f' (corr = {corr:.2f})', fontsize=16)

        # Plot betas for GEMMA vs pyGEMMA with all points colored as current phenotype in 'colors'
        sns.scatterplot(x=df['beta_gemma'], 
                        y=df['beta_pygemma'], 
                        ax=axes_beta[i], 
                        color=colors[i])
        
        # add a diagonal line
        #axes_beta[i].axline([0, 0], [1, 1], color='k', linestyle='--')
        axes_beta[i].axline((0,0), slope=1, color='k', linestyle='--')
        # Set y axis label if it's the first column
        if i % 3 == 0:
            axes_beta[i].set_ylabel(r'pyGEMMA $\beta$', fontsize=16)
        else:
            axes_beta[i].set_ylabel('', fontsize=16)

        # Set x axis label if it's the last row
        if i > 2:
            axes_beta[i].set_xlabel(r'GEMMA $\beta$', fontsize=16)
        else:
            axes_beta[i].set_xlabel('', fontsize=16)

        # Compute correlation between GEMMA and pyGEMMA betas
        corr = df['beta_gemma'].corr(df['beta_pygemma'])

        # Set title
        axes_beta[i].set_title(pheno['name'] + f' (corr = {corr:.2f})', fontsize=16)

    # Save figures
    fig_pval.tight_layout()

    fig_pval.savefig(os.path.join(OUTPUT, 'pval_comparison.png'), dpi=300)
    fig_pval.clf()

    fig_beta.tight_layout()

    fig_beta.savefig(os.path.join(OUTPUT, 'beta_comparison.png'), dpi=300)
    fig_beta.clf()

        
