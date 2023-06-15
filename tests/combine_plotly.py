import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BENCHMARK_DIR = "/net/mulan/home/rlangefe/gemma_work/pygemma/tests/benchmark_test"

if __name__ == '__main__':
    # For each directory in the benchmark directory, read in results.csv
    # Not everything in BENCHMARK_DIR is a directory
    results = []

    for directory in os.listdir(BENCHMARK_DIR):
        if os.path.isdir(os.path.join(BENCHMARK_DIR, directory)):
            # If results file exists
            if os.path.exists(os.path.join(BENCHMARK_DIR, directory, 'results.csv')):            
                results.append(pd.read_csv(os.path.join(BENCHMARK_DIR, directory, 'results.csv')))
    
    results = pd.concat(results)

    # Save results
    #results.to_csv(os.path.join(BENCHMARK_DIR, 'results.csv'), index=False)

    # Plot results with Plotly

    times = pd.melt(results, id_vars=['sample_size', 'num_snps', 'num_covars'], value_vars=['GEMMA', 'pyGEMMA'], var_name='Method', value_name='Time (s)')

    # Plot time by number of SNPs
    fig = px.scatter(times, x='num_snps', y='Time (s)', color='Method', trendline='lowess', hover_data=['num_snps', 'sample_size', 'num_covars', 'Time (s)'])
    fig.update_layout(
        xaxis_title='Number of SNPs',
        yaxis_title='Time (s)',
        title='Time to run GWAS: Number of SNPs'
    )
    fig.write_html(os.path.join(BENCHMARK_DIR, 'time_by_snps.html'))

    # Plot time by sample size
    fig = px.scatter(times, x='sample_size', y='Time (s)', color='Method', trendline='lowess', hover_data=['num_snps', 'sample_size', 'num_covars', 'Time (s)'])
    fig.update_layout(
        xaxis_title='Sample Size',
        yaxis_title='Time (s)',
        title='Time to run GWAS: Sample Size'
    )
    fig.write_html(os.path.join(BENCHMARK_DIR, 'time_by_sample_size.html'))

    # Plot time by number of covariates
    fig = px.scatter(times, x='num_covars', y='Time (s)', color='Method', trendline='lowess', hover_data=['num_snps', 'sample_size', 'num_covars', 'Time (s)'])
    fig.update_layout(
        xaxis_title='Number of Covariates',
        yaxis_title='Time (s)',
        title='Time to run GWAS: Number of Covariates'
    )
    fig.write_html(os.path.join(BENCHMARK_DIR, 'time_by_covars.html'))

    # Scatter plot of pyGEMMA vs GEMMA
    fig = px.scatter(results, x='GEMMA', y='pyGEMMA', color='num_snps')
    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color='black', dash='dash')
    )
    fig.update_layout(
        xaxis_title='GEMMA',
        yaxis_title='pyGEMMA',
        title='Time to run GWAS: pyGEMMA vs GEMMA'
    )
    fig.write_html(os.path.join(BENCHMARK_DIR, 'pygemma_vs_gemma.html'))

    # Plot pyGEMMA speedup over GEMMA wrt each variable
    variable_list = ['sample_size', 'num_snps', 'num_covars']
    variable_names = ['Sample Size', 'Number of SNPs', 'Number of Covariates']

    speedup = results['GEMMA'] / results['pyGEMMA']

    for variable, var_name in zip(variable_list, variable_names):
        fig = px.scatter(results, x=variable, y=speedup, hover_data=['num_snps', 'sample_size', 'num_covars', 'GEMMA', 'pyGEMMA'])
        fig.update_layout(
            xaxis_title=var_name,
            yaxis_title='Speedup',
            title='pyGEMMA Speedup Over GEMMA by {}'.format(var_name)
        )
        fig.write_html(os.path.join(BENCHMARK_DIR, 'speedup_{}.html'.format(variable)))
