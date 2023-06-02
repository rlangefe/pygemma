import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os

# Function to make Manhattan plot
# df in format geneID	beta	se	logl_H1	l_remle	l_mle	p_wald	p_lrt	p_score
# geneID is chr:pos:ref:alt
def manhattan_plot(df, output_path):
    sns.set_theme()

    # Parse SNP column where SNP is in the format: chr:pos:ref:alt
    snps_values = df['geneID'].str.split(':', expand=True)
    snps_values.columns = ['CHR', 'POS', 'REF', 'ALT']


    # Manhattan plot adapted from https://stackoverflow.com/a/66062857
    results_df = pd.DataFrame(
        {
        'pos'  : snps_values['POS'].values,
        'pval' : -np.log10(df['p_wald']+1e-20),
        'chr' : snps_values['CHR'].values
        }
    )

    results_df = results_df.sort_values(['chr', 'pos'])
    results_df.reset_index(inplace=True, drop=True)
    results_df['i'] = results_df.index

    alpha = -np.log10(0.05/len(results_df))
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
    plt.title(f'Manhattan Plot - {os.path.basename(output_path)}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "gemma_wald_Manhattan.png"))
    plt.clf()

def qq_plot(df, output_path):
    sns.set_theme()

    theoretical = np.linspace(1/len(df),1.0,len(df))
    pvals = np.sort(df['p_wald'])

    plt.scatter(y=-np.log10(pvals+1e-20), x=-np.log10(theoretical))
    plt.axline((0,0), slope=1, color='red')
    plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
    plt.ylabel(r'Observed: $-\log_{10}(p)$')
    plt.title(f'QQ Plot - {os.path.basename(output_path)}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "gemma_wald_qq.png"))
    plt.clf()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Path to input file", type=str, default="input.csv")
    parser.add_argument("-o", "--output", dest="output", help="Path to output file", type=str, default=".")

    args = parser.parse_args()

    # Read in input file
    df = pd.read_csv(args.input, sep='\t')
    df_pygemma = pd.read_csv(os.path.join(args.output, 'pygemma_results.csv'))

    # Set rs to geneID
    df['geneID'] = df['rs']

    if len(df) > 0:
        # Make manhattan plot
        manhattan_plot(df, args.output)

        # Make qq plot
        qq_plot(df, args.output)

        # Make figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

        # Subset df_pygemma and df to only include SNPs that are in both
        df_pygemma = df_pygemma[df_pygemma['SNPs'].isin(df['geneID'])]
        df = df[df['geneID'].isin(df_pygemma['SNPs'])]

        # Plot pygemma -log10(p_wald) vs. gemma -log10(p_wald)
        sns.scatterplot(ax=ax1, x=-np.log10(df_pygemma['p_wald']), y=-np.log10(df['p_wald']))
        ax1.set_xlabel(r'PyGEMMA: $-\log_{10}(p)$')
        ax1.set_ylabel(r'GEMMA: $-\log_{10}(p)$')

        # Add line on y=x
        ax1.axline((0,0), slope=1, color='red')

        # Plot pygemma -log10(beta) vs. gemma -log10(beta)
        sns.scatterplot(ax=ax2, x=-np.log10(df_pygemma['beta']), y=-np.log10(df['beta']))
        ax2.set_xlabel(r'PyGEMMA: $-\log_{10}(\beta)$')
        ax2.set_ylabel(r'GEMMA: $-\log_{10}(\beta)$')

        plt.suptitle(f'PyGEMMA vs. GEMMA - {os.path.basename(args.output)}')
        plt.tight_layout()

        plt.savefig(os.path.join(args.output, "pygemma_vs_gemma.png"))
        plt.clf()

        # Separate scatter for just p-values
        sns.scatterplot(x=-np.log10(df_pygemma['p_wald']), y=-np.log10(df['p_wald']))
        plt.xlabel(r'PyGEMMA: $-\log_{10}(p)$')
        plt.ylabel(r'GEMMA: $-\log_{10}(p)$')
        plt.axline((0,0), slope=1, color='red')
        plt.title(f'PyGEMMA vs. GEMMA - {os.path.basename(args.output)}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "pygemma_vs_gemma_p.png"))
        plt.clf()

        # Separate scatter for just betas
        sns.scatterplot(x=-np.log10(df_pygemma['beta']), y=-np.log10(df['beta']))
        plt.xlabel(r'PyGEMMA: $-\log_{10}(\beta)$')
        plt.ylabel(r'GEMMA: $-\log_{10}(\beta)$')
        plt.axline((0,0), slope=1, color='red')
        plt.title(f'PyGEMMA vs. GEMMA - {os.path.basename(args.output)}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "pygemma_vs_gemma_beta.png"))
        plt.clf()

        # Scatterplot of lambda values
        sns.scatterplot(x=-np.log10(df_pygemma['lambda']), y=-np.log10(df['l_remle']))
        plt.xlabel(r'PyGEMMA: $-\log_{10}(\lambda$)')
        plt.ylabel(r'GEMMA: $-\log_{10}(\lambda)$')
        plt.title(f'PyGEMMA vs. GEMMA - {os.path.basename(args.output)}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "pygemma_vs_gemma_lambda.png"))
        plt.clf()
