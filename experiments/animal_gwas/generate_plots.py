######################################
# Script to read all data and create #
# time comparison plots              #
######################################

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

from rich.progress import track

import os
import sys

if __name__ == '__main__':
    # Parse arguments input and output, both paths to directories
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input directory', type=str, default='.')
    parser.add_argument('-o', '--output', help='Path to output directory', type=str, default='benchmark_plots')
    args = parser.parse_args()

    # Create output directory if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read in data files from all directories in input directory of form 'run_results_*'
    # and store in list
    df_list = []

    for directory in track(os.listdir(args.input), description='Reading in data files...'):
        if directory.startswith('run_results_'):
            if os.path.exists(os.path.join(args.input, directory, 'results.csv')):
                df_list.append(pd.read_csv(os.path.join(args.input, directory, 'results.csv')))

    # Concatenate all dataframes into one
    df_full = pd.concat(df_list)

    df_full = df_full[['covars','pheno','pygemma_time','gemma_time', 'gcta_time']]

    # Print table of how many non-na entries per method at each covar number
    print('Number of non-na entries per method at each covar number')
    print(df_full.groupby(['covars']).count())

    # Pivot to have columns covars, pheno, method, and time
    df = df_full.melt(id_vars=['covars','pheno'], var_name='Method', value_name='Time')

    # Rename columns covars to Covariates and pheno to Phenotype
    df = df.rename(columns={'covars':'Covariates', 'pheno':'Phenotype'})

    # Method is a column of strings
    # pygemma_time should be replaced with pyGEMMA and gemma_time with GEMMA
    df['Method'] = df['Method'].str.replace('pygemma_time', 'pyGEMMA')
    df['Method'] = df['Method'].str.replace('gemma_time', 'GEMMA')
    df['Method'] = df['Method'].str.replace('gcta_time', 'GCTA')

    # Define Pallette
    pallette = {
                'GEMMA': matplotlib.colors.to_hex('tab:blue'),      # blue
                'pyGEMMA': matplotlib.colors.to_hex('tab:orange'),  # orange
                'GCTA': matplotlib.colors.to_hex('tab:green'),      # green
                'fastGWA': matplotlib.colors.to_hex('tab:red'),     # red
                'Regenie': matplotlib.colors.to_hex('tab:purple'),  # purple
                'Linear Regression': matplotlib.colors.to_hex('tab:brown') # brown
                }

    # Plot with seaborn
    sns.set_theme()

    # Plot runtime vs. number of covariates
    # Color by method
    ax = sns.lineplot(data=df, x='Covariates', y='Time', hue='Method', palette=pallette)
    plt.ylabel('Runtime (s)')
    plt.xlabel('Number of Covariates')
    plt.title('Runtime vs. Number of Covariates')
    plt.legend().remove()
    
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(args.output, 'runtime_vs_covars.png'), dpi=1200)
    plt.clf()

    # Compute speedup vs pyGEMMA (method time / pyGEMMA time)
    df_speedup = df_full.copy()

    # Make speedup for each column ending with _time that isn't pygemma_time
    for col in df_speedup.columns:
        if col.endswith('_time') and col != 'pygemma_time':
            df_speedup[col] = df_speedup[col] / df_speedup['pygemma_time']

    # Drop pygemma_time column
    df_speedup = df_speedup.drop(columns=['pygemma_time'])

    # Pivot to have columns covars, pheno, method, and speedup
    df_speedup = df_speedup.melt(id_vars=['covars','pheno'], var_name='Method', value_name='Speedup')

    # Rename columns covars to Covariates and pheno to Phenotype
    df_speedup = df_speedup.rename(columns={'covars':'Covariates', 'pheno':'Phenotype'})

    # Method is a column of strings
    # pygemma_time should be replaced with pyGEMMA and gemma_time with GEMMA
    df_speedup['Method'] = df_speedup['Method'].str.replace('gemma_time', 'GEMMA')
    df_speedup['Method'] = df_speedup['Method'].str.replace('gcta_time', 'GCTA')


    # Plot speedup vs. number of covariates
    # Color by method
    ax = sns.lineplot(data=df_speedup, x='Covariates', y='Speedup', hue='Method', palette=pallette)
    plt.ylabel(r'Speedup $\left( \frac{\mathrm{Method\/ Time}}{\mathrm{pyGEMMA\/ Time}} \right)$')
    plt.xlabel('Number of Covariates')
    plt.title('Speedup vs. Number of Covariates')
    
    # y-axis on log scale
    #plt.yscale('log')

    # Plot black dotted horizontal line at 1
    plt.axhline(y=1, color='black', linestyle='--')

    plt.legend().remove()

    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(args.output, 'speedup_vs_covars.png'), dpi=1200)
    plt.clf()




