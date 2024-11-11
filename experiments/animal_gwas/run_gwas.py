import time
import os

import numpy as np
import pandas as pd
import qnorm

from pygemma import lmm, pygemma_model, plot

from rich.console import Console
from rich.progress import track
from rich.traceback import Traceback

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from scipy import stats

import warnings

import argparse

def calculate_allele_frequency_matrix(genetic_data_matrix):
    # Convert genetic data matrix to numpy array
    data_matrix = np.array(genetic_data_matrix)

    # Calculate the total number of individuals
    total_individuals = data_matrix.shape[0]

    # Calculate the total number of alleles
    total_alleles = 2 * total_individuals

    # Calculate the sum of allele dosages along the columns
    sum_dosages = np.sum(data_matrix, axis=0)

    # Calculate the frequency of the minor allele (q) for each column
    frequency_minor_allele = sum_dosages / total_alleles

    # Calculate the frequency of the major allele (p) for each column
    frequency_major_allele = 1 - frequency_minor_allele

    return frequency_major_allele, frequency_minor_allele

def calculate_genetic_relatedness_matrix(X):
    #_, mafs = calculate_allele_frequency_matrix(X)
    sd = np.std(X, axis=0)
    sd[sd == 0] = 1
    K = (X - np.mean(X, axis=0)) / sd
    # Calculate the genetic relatedness matrix from GCTA
    # K_jk = 1/n * sum_i (x_ji - 2*p_i) * (x_ki - 2*p_i) / (2*p_i*(1-p_i))
    #K = (X - 2*mafs) / np.sqrt(2*mafs*(1-mafs))
    K = K @ K.T / X.shape[1]    

    return K

if __name__ =='__main__':
    OUTPUT = os.path.join(os.getcwd(), 'output')
    PCS = 5
    alpha = 0.05

    # Make output directory if it doesn't exist
    if not os.path.exists(OUTPUT):
        os.mkdir(OUTPUT)

    print('Reading genotypes...')
    genotypes = pd.read_csv("/net/mulan/home/rlangefe/gemma_work/clean_gemma/GEMMA/example/mouse_hs1940.geno.txt.gz",
                            engine='c',
                            header=None)

    genotypes.columns = ['rs', 'major', 'minor'] + list(range(0, genotypes.shape[1] - 3))

    print('Reading phenotypes...')
    phenotypes = pd.read_csv("/net/mulan/home/rlangefe/gemma_work/clean_gemma/GEMMA/example/mouse_hs1940.pheno.txt",
                            sep='\t',
                            engine='c',
                            header=None)

    phenotypes.columns = [f'Pheno_{i}' for i in range(phenotypes.shape[1])]

    # Impute missing phenotypes with mean for each column other than rs, major, minor
    print('Imputing missing phenotypes...')
    phenotypes = phenotypes.fillna(phenotypes.mean())

    # Impute missing genotypes with mean for each column
    print('Imputing missing genotypes...')
    #genotypes[genotypes.columns[3:]] = genotypes[genotypes.columns[3:]].fillna(genotypes[genotypes.columns[3:]].mean())

    # Convert the genotype columns to floats
    X = genotypes[genotypes.columns[3:]].values.astype(np.float32).T

    # Impute with mean of each column of X with sklearn
    from sklearn.impute import SimpleImputer
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X)
    X = imp_mean.transform(X)
    

    # Convert the SNP columns to strings
    snps = genotypes[genotypes.columns[0]].astype(str)
    snps.columns = ['rs']

    # Calculate the genetic relatedness matrix
    print('Calculating genetic relatedness matrix...')
    K = calculate_genetic_relatedness_matrix(X)

    nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
    print(f"Using {nproc} processors")

    # Initialize covariate matrix
    W = np.ones((X.shape[0], 1))

    if PCS > 0:
        # Calculate the principal components
        print('Calculating principal components...')
        pca = PCA(n_components=PCS)
        pcs = pca.fit_transform(X)

        # Add the principal components to the covariate matrix
        W = np.hstack((W, pcs))
    

    # Read annotations file tsv
    annotations = pd.read_csv("/net/mulan/home/rlangefe/gemma_work/clean_gemma/GEMMA/example/mouse_hs1940.anno.txt",
                            sep='\t',
                            engine='c',
                            header=None)
    
    annotations.columns = ['rs', 'POS', 'CHR', 'IDK']

    # For any rs of form "CEL-1_44668113", extract 44668113 and assign to pos
    
    

    # snps is rs number. Want to match rs number to annotations
    # after this, snps should have chr:pos:ref:alt
    snps = snps.astype(str)
    
    # Merge snps and annotations based on rs number
    # if annotations doesn't find match in snps, it will be dropped
    # if snps doesn't find match in annotations, value will be NaN
    snps = pd.merge(snps, annotations, on='rs')

    # snps['SNP'] = snps['CHR'].astype(str) + ':' + snps['POS'].astype(str) + ':' + genotypes['major'].astype(str) + ':' + genotypes['minor'].astype(str)
    snps['SNPs'] = snps['CHR'].astype(str) + ':' + snps['POS'].astype(str) + ':' + genotypes['major'].astype(str) + ':' + genotypes['minor'].astype(str)

    print(snps)

    # Get index of non nan pos or chr
    not_nan_index = snps[~snps['POS'].isnull()].index

    # Subset snps and X to remove nan pos or chr
    snps = snps.iloc[not_nan_index,:]
    X = X[:,not_nan_index]

    # convert POS and CHR to int
    snps['POS'] = snps['POS'].astype(int)
    snps['CHR'] = snps['CHR'].astype(int)

    # Randomly sample SNPs
    #selection = np.random.choice(X.shape[1], 2000, replace=False)
    selection = range(0, X.shape[1])
    X = X[:, selection]
    snps = snps.iloc[selection,:]

    # Run GWAS for each phenotype
    for pheno in phenotypes.columns:
        print(f"Running GWAS for {pheno}...")

        # Convert the phenotype to a numpy array
        Y = phenotypes[pheno].values.astype(np.float32).reshape(-1, 1)

        # Run GWAS
        start = time.time()
        data_results = lmm.pygemma(X=X, Y=Y, W=W, K=K, snps=snps['rs'], verbose=1)
        print('PyGemma Run Time:', time.time() - start, 's')
        print(data_results.head(10))

        # Make a QQ plot
        fig, ax = plot.qq_plot(data_results['p_wald'], scale='log')

        theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
        pvals = np.sort(data_results['p_wald'])

        lambda_gc = np.median(stats.chi2.ppf(theoretical, 1) / stats.chi2.ppf(pvals, 1))

        plt.title(f'Q-Q Plot for {pheno} (Lambda GC: {lambda_gc})')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{pheno}_wald_qq.png"))
        
        # Close figure
        plt.close(fig)


        # Make a Manhattan plot with matplotlib
        fig, ax = plot.manhattan_plot(pval=data_results['p_wald'], 
                                        pos=snps['POS'].values, 
                                        chrom=snps['CHR'].values,
                                        beta=None,
                                        cutoff='bonferroni',
                                        scale='log',
                                        plotly=False)
        
        plt.title(f'Manhattan Plot for {pheno}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{pheno}_wald_manhattan.png"))

        # Close figure
        plt.close(fig)

        # Make a Manhattan plot with plotly
        plot.manhattan_plot(pval=data_results['p_wald'], 
                            pos=snps['POS'].values, 
                            chrom=snps['CHR'].values,
                            beta=data_results['beta'],
                            cutoff='bonferroni',
                            scale='log',
                            plotly=True,
                            save_path=os.path.join(OUTPUT, f"{pheno}_wald_manhattan.html"))


        # results_df = pd.DataFrame(
        #     {
        #     'pos'  : snps['POS'].values,
        #     'pval' : -np.log10(data_results['p_wald']+1/len(data_results)),
        #     'chr' : snps['CHR'].values
        #     }
        # )

        # results_df = results_df.sort_values(['chr', 'pos'])
        # results_df.reset_index(inplace=True, drop=True)
        # results_df['i'] = results_df.index


        # sns.set_theme()
        # sns.scatterplot(x=results_df['i'], 
        #                 y=results_df['pval'], 
        #                 hue=results_df['chr'], 
        #                 linewidth=0)
        
        # # Annotate snp with small font if results_df['pval'] above alpha
        # alpha_star = -np.log10(alpha/len(data_results))
        # for i, row in results_df.iterrows():
        #     if row['pval'] > alpha_star:
        #         plt.annotate(str(row['chr']) + ':' + str(row['pos']), (row['i'], row['pval']), fontsize=6)

        # plt.axline((0,-np.log10(alpha/len(data_results))), slope=0, color='red')
        # chrom_df=results_df.groupby('chr')['i'].median()
        # plt.xlabel('Chromosome') 
        # plt.xticks(chrom_df,chrom_df.index)
        # plt.ylabel(r'$-\log_{10}(p)$')
        # plt.legend().remove()
        # plt.title(f"{pheno} Manhattan Plot")
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{pheno}_wald_manhattan.png"))
        # plt.clf()


