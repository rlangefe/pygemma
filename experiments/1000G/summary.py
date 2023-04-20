import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from rich.progress import track
from rich.console import Console

import os
import argparse

def make_ref_dicts(name='Model', manhattan='manhattan.png', qq='qq.png'):
    return {
        'Name' : name,
        'Manhattan' : manhattan,
        'QQ' : qq,
    }

pygemma_files = make_ref_dicts('pyGEMMA', 'wald_Manhattan.png', 'wald_qq.png')

gemma_files = make_ref_dicts('GEMMA', 'gemma_wald_Manhattan.png', 'gemma_wald_qq.png')

ols_files = make_ref_dicts('OLS', 'linreg_wald_Manhattan.png', 'linreg_wald_qq.png')

ols_gc_files = make_ref_dicts('OLS_Genomic_Control', 'linreg_wald_Manhattan_gc.png', 'linreg_wald_qq_gc.png')
    
model_list = [pygemma_files, gemma_files, ols_files, ols_gc_files]

def create_comparison_plots(directory_path):
    # Grid of Manhattan plots
    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    
    for i, model_dict in enumerate(model_list):
            
        manhattan_plot = plt.imread(os.path.join(directory_path, model_dict['Manhattan']))

        ax[i//2, i%2].imshow(manhattan_plot)

        ax[i//2, i%2].set_title(f"{model_dict['Name'].replace('_', ' ')}")
    
        ax[i//2, i%2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(directory_path, 'manhatten_summary.png'))
    plt.close(fig)
    plt.clf()

    # Grid of QQ plots
    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    
    for i, model_dict in enumerate(model_list):
            
        qq_plot = plt.imread(os.path.join(directory_path, model_dict['QQ']))

        ax[i//2, i%2].imshow(qq_plot)

        ax[i//2, i%2].set_title(f"{model_dict['Name'].replace('_', ' ')}")
    
        ax[i//2, i%2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(directory_path, 'qq_summary.png'))
    plt.close(fig)
    plt.clf()

    # Grid of Manhattan plots
    fig, ax = plt.subplots(1,3, figsize=(10, 10))
    
    for i, model_dict in enumerate([pygemma_files, gemma_files, ols_gc_files]):
            
        manhattan_plot = plt.imread(os.path.join(directory_path, model_dict['Manhattan']))

        ax[i].imshow(manhattan_plot)

        ax[i].set_title(f"{model_dict['Name'].replace('_', ' ')}")
    
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(directory_path, 'manhatten_summary_no_ols.png'))
    plt.close(fig)
    plt.clf()

    # Grid of QQ plots
    fig, ax = plt.subplots(1,3, figsize=(10, 10))
    
    for i, model_dict in enumerate([pygemma_files, gemma_files, ols_gc_files]):
            
        qq_plot = plt.imread(os.path.join(directory_path, model_dict['QQ']))

        ax[i].imshow(qq_plot)

        ax[i].set_title(f"{model_dict['Name'].replace('_', ' ')}")
    
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(directory_path, 'qq_summary_no_ols.png'))
    plt.close(fig)
    plt.clf()

    # Joint plot for Manhattan and QQ
    for i, model_dict in enumerate(model_list):
        fig, ax = plt.subplots(1,2, figsize=(10, 5))

        manhattan_plot = plt.imread(os.path.join(directory_path, model_dict['Manhattan']))
        qq_plot = plt.imread(os.path.join(directory_path, model_dict['QQ']))

        ax[0].imshow(manhattan_plot)
        ax[1].imshow(qq_plot)

        ax[0].axis('off')
        ax[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(directory_path, f"{model_dict['Name']}_summary.png"))
        plt.close(fig)
        plt.clf()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str, default='data')
    parser.add_argument('-o', '--out-dir', dest='out_dir', type=str, default='out')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with Console() as console:
        results_df = pd.DataFrame(columns=['pygemma_p_wald', 'gemma_p_wald', 'pygemma_sig', 'gemma_sig', 'pygemma_beta', 'gemma_beta', 'chr', 'pos', 'SNPs'])

        summary_df = pd.DataFrame(columns=['r2_beta', 'r2_p_wald', 'r2_beta_log', 'r2_p_wald_log', 'r2_significant'])

        results_list = []
        summary_list = []

        snp_count = []
            
        for directory in track(os.listdir(args.data_dir), description='Reading data...'):
            create_comparison_plots(os.path.join(args.data_dir, directory))

            pygemma_file = os.path.join(args.data_dir, directory, 'pygemma_results.csv')
            gemma_file = os.path.join(args.data_dir, directory, 'gemma_results.tsv')

            if os.path.exists(pygemma_file) and os.path.exists(gemma_file):
                pygemma_df = pd.read_csv(pygemma_file, engine='c')
                gemma_df = pd.read_csv(gemma_file, sep='\t', engine='c')

                snps_values = pygemma_df['SNPs'].str.split(':', expand=True)
                snps_values.columns = ['CHR', 'POS', 'REF', 'ALT']

                # Bonferroni corrected alpha
                alpha = 0.05/len(pygemma_df)

                
                temp_df = pd.DataFrame({
                    'pygemma_p_wald': pygemma_df['p_wald'],
                    'gemma_p_wald': gemma_df['p_wald'],
                    'pygemma_sig': pygemma_df['p_wald'] < alpha,
                    'gemma_sig': gemma_df['p_wald'] < alpha,
                    'pygemma_beta': pygemma_df['beta'],
                    'gemma_beta': gemma_df['beta'],
                    'chr': snps_values['CHR'],
                    'pos': snps_values['POS'],
                    'SNPs': pygemma_df['SNPs']
                })

                snp_count.append(len(pygemma_df))

                #results_df = pd.concat([results_df,temp_df])
                results_list.append(temp_df)

                sig_pygemma = np.array(pygemma_df['p_wald'] < alpha)*1.0
                sig_gemma = np.array(gemma_df['p_wald'] < alpha)*1.0

                if len(sig_pygemma) != 0 and len(sig_gemma) != 0:
                    
                    if len(sig_pygemma) == 1:
                        temp_df = pd.DataFrame({
                            'r2_beta': list(sig_pygemma == sig_gemma),
                            'r2_p_wald': list(sig_pygemma == sig_gemma),
                            'r2_beta_log': list(sig_pygemma == sig_gemma),
                            'r2_p_wald_log': list(sig_pygemma == sig_gemma),
                            'r2_significant':list(sig_pygemma == sig_gemma),
                        })
                    else:
                        if len(sig_gemma) == np.sum(sig_pygemma == sig_gemma):
                            r2_sig = 1.0
                        elif len(sig_gemma) == np.sum(sig_pygemma != sig_gemma):
                            r2_sig = 0.0
                        else:
                            r2_sig = np.corrcoef(sig_pygemma, sig_gemma)[0, 1]**2

                        temp_df = pd.DataFrame({
                            'r2_beta': [np.corrcoef(pygemma_df['beta'], gemma_df['beta'])[0, 1]**2],
                            'r2_p_wald': [np.corrcoef(pygemma_df['p_wald'], gemma_df['p_wald'])[0, 1]**2],
                            'r2_beta_log': [np.corrcoef(-np.log10(np.abs(pygemma_df['beta'])+1e-20), -np.log10(np.abs(gemma_df['beta'])+1e-20))[0, 1]**2],
                            'r2_p_wald_log': [np.corrcoef(-np.log10(np.abs(pygemma_df['p_wald'])+1e-20), -np.log10(np.abs(gemma_df['p_wald'])+1e-20))[0, 1]**2],
                            'r2_significant':[r2_sig],
                        })

                    #summary_df = pd.concat([summary_df, temp_df])
                    summary_list.append(temp_df)

        console.print(f"Average number of SNPs: {np.mean(snp_count)}")
        console.print(f"Median number of SNPs: {np.median(snp_count)}")
        console.print(f"SD of number of SNPs: {np.std(snp_count)}")
        console.print(f"5th percentile of number of SNPs: {np.percentile(snp_count, 5)}")
        console.print(f"95th percentile of number of SNPs: {np.percentile(snp_count, 95)}")

        console.status('Concatenating data...')
        results_df = pd.concat(results_list, axis=0)
        summary_df = pd.concat(summary_list, axis=0)

        console.print(f"Number of unique SNPs: {len(results_df['SNPs'].unique())}")

        for pygemma_col, gemma_col in zip(['pygemma_p_wald', 'pygemma_beta'], ['gemma_p_wald', 'gemma_beta']):
            console.status(f'Running {pygemma_col} and {gemma_col}...')

            results_df = results_df.sort_values(['chr', 'pos'])
            results_df.reset_index(inplace=True, drop=True)
            results_df['i'] = results_df.index

            mse = (results_df[pygemma_col] - results_df[gemma_col])**2

            sns.set_theme()
            plt.figure(figsize=(10, 10))
            plt.title(f"{str(pygemma_col).replace('_', ' ').title()} and {str(gemma_col).replace('_', ' ').title()} MSE Distribution")
            plt.ylabel('MSE')
            plt.xlabel('chr') 

            sns.lineplot(x=results_df.index, y=mse, errorbar='ci')

            chrom_df=results_df.groupby('chr')['i'].median()
            plt.xticks(chrom_df,chrom_df.index)
            plt.tight_layout()

            plt.savefig(os.path.join(args.out_dir, f'{pygemma_col}_{gemma_col}_mse.png'))
            plt.clf()

            # Log scale
            mse = (-np.log10(np.abs(results_df[pygemma_col])+1e-20) + np.log10(np.abs(results_df[gemma_col])+1e-20))**2

            sns.set_theme()
            plt.figure(figsize=(10, 10))
            plt.title(f"{str(pygemma_col).replace('_', ' ').title()} -log10 and {str(gemma_col).replace('_', ' ').title()} -log10 MSE Distribution")
            plt.ylabel('MSE')
            plt.xlabel('chr') 

            sns.lineplot(x=np.array(results_df.index), y=mse, errorbar='ci')

            chrom_df=results_df.groupby('chr')['i'].median()
            plt.xticks(chrom_df,chrom_df.index)
            plt.tight_layout()

            plt.savefig(os.path.join(args.out_dir, f'{pygemma_col}_{gemma_col}_mse_log.png'))
            plt.clf()

        # Histplot of location
        console.status('Plotting location distribution...')
        sns.set_theme()
        plt.figure(figsize=(10, 10))
        plt.title('Location Distribution')
        plt.ylabel('Count')
        plt.xlabel('chr')

        sns.kdeplot(results_df['i'], fill=True, alpha=0.5)

        chrom_df=results_df.groupby('chr')['i'].median()
        plt.xticks(chrom_df,chrom_df.index)
        plt.tight_layout()

        plt.savefig(os.path.join(args.out_dir, 'location_dist.png'))
        plt.clf()

        for name, col in zip(['R^2 Beta',' R^2 P Wald', '-log10(|R^2 Beta|)', '-log10(|R^2 P Wald|)', 'R^2 Significant'],
                                ['r2_beta', 'r2_p_wald', 'r2_beta_log', 'r2_p_wald_log', 'r2_significant']):
            
            console.status(f'Plotting {name}...')

            sns.set_theme()
            plt.figure(figsize=(10, 10))
            plt.title(f'{name} Distribution')

            #sns.displot(summary_df[col], kind='kde')
            sns.histplot(data=pd.Series(summary_df[col].values), element='poly', stat='probability')
            plt.ylabel('Probability')
            plt.xlabel(name)
            plt.tight_layout()

            plt.savefig(os.path.join(args.out_dir, f'{name}_dist.png'))
            plt.clf()

            console.print(f'{name} Stats\nMean: {summary_df[col].mean()}\nStd: {summary_df[col].std()}\nMedian: {summary_df[col].median()}\nMin: {summary_df[col].min()}\nMax: {summary_df[col].max()}\n')




