import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

BENCHMARK_DIR="/net/mulan/home/rlangefe/gemma_work/pygemma/tests/benchmark_test"

if __name__ == '__main__':
    # For each directory in the benchmark directory, read in results.csv
    # Not everything in BENCHMARK_DIR is a directory
    results = []

    for directory in os.listdir(BENCHMARK_DIR):
        if os.path.isdir(os.path.join(BENCHMARK_DIR, directory)):
            # If results file exists
            if os.path.exists(os.path.join(BENCHMARK_DIR, directory, 'results.csv')):            
                results.append(pd.read_csv(os.path.join(BENCHMARK_DIR, directory, 'results.csv')))
    
    results = pd.concat(results, ignore_index=True)

    results.reset_index(drop=True, inplace=True)

    # Save results
    results.to_csv(os.path.join(BENCHMARK_DIR, 'results.csv'), index=False)

    # Plot results with seaborn on same plot
    sns.set_style("whitegrid")

    times = pd.melt(results, id_vars=['sample_size', 'num_snps', 'num_covars'], value_vars=['GEMMA', 'pyGEMMA'], var_name='Method', value_name='Time (s)')

    # Plot results
    sns.set_style("whitegrid")
    sns.lineplot(x='num_snps', y='Time (s)', hue='Method', data=times, errorbar='ci')
    plt.xlabel('Number of SNPs')
    plt.ylabel('Time (s)')
    plt.title('Time to run GWAS')
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'time_by_snps.png'))
    plt.clf()

    # Plot results
    sns.set_style("whitegrid")
    sns.lineplot(x='sample_size', y='Time (s)', hue='Method', data=times, errorbar='ci')
    plt.xlabel('Sample Size')
    plt.ylabel('Time (s)')
    plt.title('Time to run GWAS')
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'time_by_sample_size.png'))
    plt.clf()

    # Plot results
    sns.set_style("whitegrid")
    sns.lineplot(x='num_covars', y='Time (s)', hue='Method', data=times, errorbar='ci')
    plt.xlabel('Number of Covariates')
    plt.ylabel('Time (s)')
    plt.title('Time to run GWAS')
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'time_by_covars.png'))
    plt.clf()

    # Scatter plot of pyGEMMA vs GEMMA
    sns.set_style("whitegrid")
    sns.scatterplot(x='GEMMA', y='pyGEMMA', hue='num_snps', data=results)
    plt.xlabel('GEMMA')
    plt.ylabel('pyGEMMA')

    # Line on 45 degree angle
    plt.axline((0, 0), slope=1, color='black', linestyle='--')

    plt.title('Time to run GWAS')
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'pygemma_vs_gemma.png'))
    plt.clf()

    # Plot pyGEMMA speedup over GEMMA wrt each variable
    variable_list = ['sample_size', 'num_snps', 'num_covars']
    variable_names = ['Sample Size', 'Number of SNPs', 'Number of Covariates']

    speedup = results['GEMMA']/results['pyGEMMA']

    for variable, var_name in zip(variable_list, variable_names):
        sns.set_style("whitegrid")
        sns.lineplot(x=variable, y=speedup, data=results)
        plt.xlabel(var_name)
        plt.ylabel('Speedup')
        plt.title('pyGEMMA Speedup Over GEMMA by {}'.format(var_name))
        plt.tight_layout()
        plt.savefig(os.path.join(BENCHMARK_DIR, 'speedup_{}.png'.format(variable)))
        plt.clf()
