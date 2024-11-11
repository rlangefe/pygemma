import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

import numpy as np

import warnings

#BENCHMARK_DIR="/net/mulan/home/rlangefe/gemma_work/pygemma/tests/benchmark_test_32"
#BENCHMARK_DIR="/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/benchmark_test"
BENCHMARK_DIR="/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/benchmark_grid_test"
LEGEND = True
LOG = True

#error_bars = 'ci'
#estimator = 'mean'
#error_bars = ('sd', 1)
#estimator = 'mean'
error_bars = ("pi", 50)
estimator = 'median'

if __name__ == '__main__':
    # For each directory in the benchmark directory, read in results.csv
    # Not everything in BENCHMARK_DIR is a directory
    results = []
    result_job = []

    for directory in os.listdir(BENCHMARK_DIR):
        if os.path.isdir(os.path.join(BENCHMARK_DIR, directory)):
            # If results file exists
            if os.path.exists(os.path.join(BENCHMARK_DIR, directory, 'results.csv')):            
                results.append(pd.read_csv(os.path.join(BENCHMARK_DIR, directory, 'results.csv')))
                result_job = result_job + [directory]*len(results[-1])

    # Error out if no results were found
    if len(results) == 0:
        #raise ValueError('No results found in {}'.format(BENCHMARK_DIR))
        warnings.warn('No results found in {}'.format(BENCHMARK_DIR))
        exit()
    
    results = pd.concat(results, ignore_index=True)
    results['job'] = result_job

    # Rename results 'linear' to 'Linear Regression'
    # Rename sim_time to 'Simulation Time'
    results.rename(columns={'linear': 'Linear Regression', 'sim_time': 'Simulation Time'}, inplace=True) 

    results.reset_index(drop=True, inplace=True)

    print('Number of results: {}'.format(len(results)))

    # Group results by sample size, number of SNPs, number of covariates, and number of processors
    # Then count the number of results in each group
    # Then print the mean number for the groups
    print('Mean number of results per config of sample size: {}'.format(results.groupby('sample_size').count()['nproc'].mean()))
    print('Mean number of results per config of number of SNPs: {}'.format(results.groupby('num_snps').count()['nproc'].mean()))
    print('Mean number of results per config of number of covariates: {}'.format(results.groupby('num_covars').count()['nproc'].mean()))
    print('Mean number of results per config of number of processors: {}'.format(results.groupby('nproc').count()['num_snps'].mean()))

    # Print number of GCTA runs that took less than 1 second
    print('Number of GCTA runs that took less than 1 second: {}'.format(len(results[results['GCTA'] < 1])))

    # Print number of fastGWA runs that took less than 1 second
    print('Number of fastGWA runs that took less than 1 second: {}'.format(len(results[results['fastGWA'] < 1])))

    # Print total core-hours (Simulation Time)
    print('Total core-hours: {:.2f} core-hours'.format((results['Simulation Time'] * results['nproc']).sum()/3600))

    # Print mean time per run (Simulation Time)
    print('Mean time per run: {:.2f} seconds'.format(results['Simulation Time'].mean()))

    print('\n')

    # Define Pallette
    pallette = {
                'GEMMA': matplotlib.colors.to_hex('tab:blue'),      # blue
                'pyGEMMA': matplotlib.colors.to_hex('tab:orange'),  # orange
                'pyGEMMA - Grid Search': matplotlib.colors.to_hex('tab:purple'),  # purple
                'GCTA': matplotlib.colors.to_hex('tab:green'),      # green
                'fastGWA': matplotlib.colors.to_hex('tab:red'),     # red
                #'Regenie': matplotlib.colors.to_hex('tab:purple'),  # purple
                'Linear Regression': matplotlib.colors.to_hex('tab:brown') # brown
                }
    
    # If GCTA ran faster than 1 second, it failed, so set it to 
    # This should be done with checking for errors in the log file, but this is a quick fix for now
    results['GCTA'] = results['GCTA'].apply(lambda x: np.nan if x < 1 else x)

    # Same for fastGWA
    results['fastGWA'] = results['fastGWA'].apply(lambda x: np.nan if x < 1 else x)

    # Create table to count number of NaNs for each method's time
    # NaNs occur when the method fails to run
    methods = ['GEMMA', 'pyGEMMA', 'pyGEMMA - Grid Search', 'GCTA', 'fastGWA', 'Regenie', 'Linear Regression']
    missing_df = results[methods].isna().sum(axis=0).to_frame().T
    
    missing_df['Total'] = missing_df.sum(axis=1)

    missing_df.index = ['Number Failed']

    # Add row that computes the percentage of missing values
    missing_df.loc['Prop of Runs Failed'] = missing_df.loc['Number Failed']/len(results)

    # Modify total percentage
    missing_df.loc['Prop of Runs Failed', 'Total'] = missing_df.loc['Number Failed', 'Total']/(len(methods) * len(results))

    # Add row that computes the total number of runs attempted
    missing_df.loc['Total Runs Attempted'] = [len(results) for _ in range(len(methods))] + [len(results) * len(methods)]

    # Successful runs
    missing_df.loc['Number Successful Runs'] = missing_df.loc['Total Runs Attempted'] - missing_df.loc['Number Failed']

    # Print table
    print(missing_df)

    # Save results
    results.to_csv(os.path.join(BENCHMARK_DIR, 'results.csv'), index=False)

    # Plot results with seaborn on same plot
    sns.set_theme()

    times = pd.melt(results, 
                    id_vars=['sample_size', 'num_snps', 'num_covars', 'nproc'], 
                    #value_vars=['GEMMA', 'pyGEMMA', 'GCTA', 'fastGWA', 'Regenie', 'Linear Regression', 'pyGEMMA - Grid Search'], 
                    value_vars=['GEMMA', 'pyGEMMA', 'pyGEMMA - Grid Search', 'GCTA', 'fastGWA', 'Linear Regression'], 
                    var_name='Method', 
                    value_name='Time (s)')

    # Plot results
    print('Plotting number of SNPs vs time')
    sns.set_theme()
    sns.lineplot(x='num_snps', 
                 y='Time (s)', 
                 hue='Method', 
                 data=times, 
                 errorbar=error_bars, 
                 n_boot=1000, 
                 estimator=estimator,
                 palette=pallette)
    plt.xlabel('Number of SNPs')
    plt.ylabel('Time (s)')
    plt.title('Time to run GWAS')

    if not LEGEND:
        plt.legend().remove()

    if LOG:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'time_by_snps.png'), dpi=1200)
    plt.clf()

    # Plot results
    print('Plotting sample size vs time')
    sns.set_theme()
    sns.lineplot(x='sample_size', 
                 y='Time (s)', 
                 hue='Method', 
                 palette=pallette,
                 data=times, 
                 errorbar=error_bars,
                    n_boot=1000,
                    estimator=estimator)
    plt.xlabel('Sample Size')
    plt.ylabel('Time (s)')
    plt.title('Time to run GWAS')

    if not LEGEND:
        plt.legend().remove()

    if LOG:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'time_by_sample_size.png'), dpi=1200)
    plt.clf()

    # Plot results
    print('Plotting number of covariates vs time')
    sns.set_theme()
    sns.lineplot(x='num_covars', 
                 y='Time (s)', 
                 hue='Method', 
                 palette=pallette,
                 data=times, 
                 errorbar=error_bars,
                    n_boot=1000,
                    estimator=estimator)
    plt.xlabel('Number of Covariates')
    plt.ylabel('Time (s)')
    plt.title('Time to run GWAS')
    
    if not LEGEND:
        plt.legend().remove()
    
    if LOG:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'time_by_covars.png'), dpi=1200)
    plt.clf()

    # Scatter plot of pyGEMMA vs GEMMA
    print('Plotting pyGEMMA vs GEMMA')

    cmap = plt.cm.ScalarMappable(cmap='YlOrBr', norm=plt.Normalize(vmin=0, vmax=results['num_snps'].max()))

    sns.set_theme()
    ax = sns.scatterplot(x='GEMMA', 
                    y='pyGEMMA', 
                    hue='num_snps',
                    palette='YlOrBr', 
                    hue_norm=(0, results['num_snps'].max()),
                    data=results)
    plt.xlabel('GEMMA (s)')
    plt.ylabel('pyGEMMA (s)')

    # Line on 45 degree angle
    plt.axline((0, 0), slope=1, color='black', linestyle='--')

    # Make num_snps colorbar
    cb = plt.colorbar(cmap, ax=ax, label='Number of SNPs')

    plt.title('Time to run GWAS')
    
    if not LEGEND:
        plt.legend().remove()

    if LOG:
        plt.yscale('log')
        
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'pygemma_vs_gemma.png'), dpi=1200)
    plt.clf()

    # Plot pyGEMMA speedup over GEMMA wrt each variable
    variable_list = ['sample_size', 'num_snps', 'num_covars']
    variable_names = ['Sample Size', 'Number of SNPs', 'Number of Covariates']

    #speedup = results['GEMMA']/results['pyGEMMA']

    for method in ['GEMMA', 'GCTA', 'fastGWA', 'Regenie', 'Linear Regression']:
        results[method + ' Speedup'] = results[method]/results['pyGEMMA']

    # Melt results
    speedup = pd.melt(results,
                        id_vars=['sample_size', 'num_snps', 'num_covars', 'nproc'],
                        #value_vars=[method + ' Speedup' for method in ['GEMMA', 'GCTA', 'fastGWA', 'Regenie', 'Linear Regression']],
                        value_vars=[method + ' Speedup' for method in ['GEMMA', 'GCTA', 'fastGWA', 'Linear Regression']],
                        var_name='Method',
                        value_name='Speedup')
    
    # Rename methods
    speedup['Method'] = speedup['Method'].str.replace(' Speedup', '')

    # Plot speedup vs each variable
    for variable, var_name in zip(variable_list, variable_names):
        print('Plotting speedup vs {}'.format(variable))
        sns.set_theme()
        sns.lineplot(x=variable, 
                     y='Speedup', 
                     hue='Method',
                     palette=pallette,
                     errorbar=error_bars,
                    n_boot=1000,
                    estimator=estimator,
                     data=speedup)
        
        # Plot black dotted horizontal line at 1
        plt.axhline(y=1, color='black', linestyle='--')

        # Make y axis log scale
        if LOG:
            plt.yscale('log')

        plt.xlabel(var_name)
        plt.ylabel(r'Speedup $\left( \frac{\mathrm{Method\/ Time}}{\mathrm{pyGEMMA\/ Time}} \right)$')
        plt.title('pyGEMMA Speedup Over Methods by {}'.format(var_name))

        # Remove legend
        
        if not LEGEND:
            plt.legend().remove()

        #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(BENCHMARK_DIR, 'speedup_{}.png'.format(variable)), dpi=1200)
        plt.clf()

    # Grid Search Speedup
    for method in ['GEMMA', 'GCTA', 'fastGWA', 'Regenie', 'Linear Regression']:
        results[method + ' Speedup'] = results[method]/results['pyGEMMA - Grid Search']

    # Melt results
    speedup = pd.melt(results,
                        id_vars=['sample_size', 'num_snps', 'num_covars', 'nproc'],
                        value_vars=[method + ' Speedup' for method in ['GEMMA', 'GCTA', 'fastGWA', 'Linear Regression']],
                        #value_vars=[method + ' Speedup' for method in ['GEMMA', 'GCTA', 'fastGWA', 'Regenie', 'Linear Regression']],
                        var_name='Method',
                        value_name='Speedup')
    
    # Rename methods
    speedup['Method'] = speedup['Method'].str.replace(' Speedup', '')

    # Plot speedup vs each variable
    for variable, var_name in zip(variable_list, variable_names):
        print('Plotting speedup vs {}'.format(variable))
        sns.set_theme()
        sns.lineplot(x=variable, 
                     y='Speedup', 
                     hue='Method',
                     palette=pallette,
                        errorbar=error_bars,
                        n_boot=1000,
                        estimator=estimator,
                     data=speedup)
        
        # Plot black dotted horizontal line at 1
        plt.axhline(y=1, color='black', linestyle='--')

        # Make y axis log scale
        if LOG:
            plt.yscale('log')

        plt.xlabel(var_name)
        plt.ylabel(r'Speedup $\left( \frac{\mathrm{Method\/ Time}}{\mathrm{pyGEMMA\/ Time}} \right)$')
        plt.title('pyGEMMA Grid Search Speedup Over Methods by {}'.format(var_name))

        # Remove legend
        
        if not LEGEND:
            plt.legend().remove()

        #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(BENCHMARK_DIR, 'grid_speedup_{}.png'.format(variable)), dpi=1200)
        plt.clf()

    # Make grid of pairwise scatter plots
    # Diagonal is histogram of times for each method
    # Off diagonal is scatter plot of times for each method vs each other method
    # Off diagonal should have dotted black line at 45 degree angle
    print('Plotting pairplot')

    # Remove Regenie
    methods = ['GEMMA', 'pyGEMMA', 'GCTA', 'fastGWA', 'Linear Regression', 'pyGEMMA - Grid Search']

    sns.set_theme()

    # Make figure
    plt.figure(figsize=(20, 20))

    ax = sns.pairplot(results[methods], diag_kind='hist', plot_kws={'alpha': 0.5, 'linewidth': 0})
    for i in range(len(methods)):
        for j in range(len(methods)):
            if i != j:
                ax.axes[i, j].axline((0, 0), slope=1, color='black', linestyle='--')

                # # Axis range should start at slightly less than (0,0)
                # ax.axes[i, j].set_xlim(left=-0.1)
                # ax.axes[i, j].set_ylim(bottom=-0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, 'pairplot.png'), dpi=600)
    
