import time
import os

import numpy as np
import pandas as pd
#import qnorm
from sklearn.preprocessing import StandardScaler

from pygemma import lmm, pygemma_model, plot

from pysnptools.snpreader import Bed

from functools import reduce

# Get output from env
OUTPUT = str(os.environ.get('OUTPUT'))

# Extract PCS from env if it exists
PCS = int(os.environ.get('PCS')) if os.environ.get('PCS') is not None else 0

LINEAR = True if int(os.environ.get('LINEAR')) == 1 else False

WRITEDATA=True

# Error out if PCS is negative
if PCS < 0:
    raise ValueError('PCS must be greater than or equal to 0')

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

from rich.console import Console
from rich.progress import track
from rich.traceback import Traceback

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from scipy import stats

import gemma_utils

import warnings

sns.set_theme()

console = Console()

def write_genos(gene_df, output_file):
    # Define gene names
    gene_names = list(gene_df.columns)

    # Transpose df
    gene_df = gene_df.transpose().reset_index()

    # Set column names
    sample_ids = [f'sample{i}' for i in gene_df.columns[1:]]
    gene_df.columns = ['geneID'] + sample_ids

    # Take geneID 7:78753158:G:A and extract 7:78753158, G, and A
    # gene_df['chr'] = gene_df['geneID'].apply(lambda x: x.split(':')[0])
    # gene_df['pos'] = gene_df['geneID'].apply(lambda x: x.split(':')[1])
    # gene_df['ref'] = gene_df['geneID'].apply(lambda x: x.split(':')[2])
    # gene_df['alt'] = gene_df['geneID'].apply(lambda x: x.split(':')[3])
    gene_df['ref'] = ['X'] * len(gene_df)
    gene_df['alt'] = ['Y'] * len(gene_df)


    # gene_df with columns geneID, ref, alt, sample0, sample1, ..., sampleN
    gene_df = gene_df[['geneID', 'ref', 'alt'] + sample_ids]

    # Write to output file (as tsv)
    #gene_df.to_csv(os.path.join(output_dir, 'genotypes.tsv'), sep='\t', index=False, float_format='%.15f')
    gene_df.to_csv(output_file, sep='\t', index=False, header=False)

# def run_gwas(Y,W,X, snps=None, verbose=0, nproc=1):
#     if verbose > 0:
#         progress = track(range(X.shape[1]), description='Running GWAS...')
#     else:
#         progress = range(X.shape[1])

#     results_dict = {
#         'SNPs': [],
#         'beta': [],
#         'se_beta': [],
#         'p_wald': [],
#     }
    
#     Y = Y.reshape(-1,1).astype(np.float64)
#     X = X.astype(np.float64)
#     W = W.astype(np.float64)

#     # for col in range(W.shape[1]):
#     #     if W[:,col].std() > 0:
#     #         W[:,col] = (W[:,col] - W[:,col].mean()) / W[:,col].std()

#     covar_list = [f'Covar{i}' for i in range(W.shape[1])]
    
#     n = Y.shape[0]
#     c = W.shape[1]

#     H = W @ np.linalg.inv(W.T @ W) @ W.T

#     # Regress out W matrix from Y
#     y = Y.reshape(-1,1)
#     Y = Y -  H @ Y

#     for g in progress:
#         if snps is not None:
#             results_dict['SNPs'].append(snps[g])
#         else:
#             results_dict['SNPs'].append(g)

#         Xj = X[:,g].reshape(-1,1)

#         # Regress out W matrix from each X
#         Xj = Xj - H @ Xj
        
#         design_design_inv = 1/np.sum(np.power(Xj,2.0)) #np.linalg.inv(X.T @ X)

#         #beta_vec, resid, _, _ = np.linalg.lstsq(design_matrix, Y, rcond=None)
#         beta_vec = design_design_inv * (Xj.T @ Y)
#         resid = y - X[:,g].reshape(-1,1) * beta_vec - H @ y
        
#         sigma_sq = np.sum(np.power(resid,2.0)) / (n-c-1)
#         var_covar = float((1/(Xj.T @ Xj - Xj.T @ H @ Xj)) * (resid.T @ resid) / (n-c-1))

#         beta = beta_vec[0]
#         se_beta = np.sqrt(var_covar)

#         results_dict['beta'].append(beta)
#         results_dict['se_beta'].append(se_beta)

#         F_wald = np.power(beta/se_beta, 2.0)

#         results_dict['p_wald'].append(1-stats.f.cdf(x=F_wald, dfn=1, dfd=n-c-1))

#     return pd.DataFrame(results_dict)

import multiprocessing as mp

def run_gwas(Y, W, X, snps=None, verbose=0, nproc=1):
    if verbose > 0:
        progress = track(range(X.shape[1]), description='Running GWAS...')
    else:
        progress = range(X.shape[1])

    results_dict = {
        'SNPs': [],
        'beta': [],
        'se_beta': [],
        'p_wald': [],
    }
    
    Y = Y.reshape(-1,1).astype(np.float64)
    X = X.astype(np.float64)
    W = W.astype(np.float64)

    covar_list = [f'Covar{i}' for i in range(W.shape[1])]
    
    n = Y.shape[0]
    c = W.shape[1]

    H = W @ np.linalg.inv(W.T @ W) @ W.T

    # Regress out W matrix from Y
    y = Y.reshape(-1,1)
    Y = Y -  H @ Y

    progress = SampleIter(X, y, Y, H, snps, c)

    with mp.Pool(processes=nproc) as pool:
        results = pool.map(run_gwas_single, progress)

    results = pd.DataFrame.from_dict(results)

    return results

def run_gwas_single(t):
    Xj, y, H, Y, snp, c = t

    results_dict = {
        'SNPs': np.nan,
        'beta': np.nan,
        'se_beta': np.nan,
        'p_wald': np.nan,
    }

    results_dict['SNPs'] = snp

    Xj = Xj.reshape(-1,1)

    n = Y.shape[0]

    # Regress out W matrix from each X
    Xj = Xj - H @ Xj
    
    design_design_inv = 1/np.sum(np.power(Xj,2.0)) #np.linalg.inv(X.T @ X)

    #beta_vec, resid, _, _ = np.linalg.lstsq(design_matrix, Y, rcond=None)
    beta_vec = design_design_inv * (Xj.T @ Y)
    resid = y - Xj.reshape(-1,1) * beta_vec - H @ y
    
    sigma_sq = np.sum(np.power(resid,2.0)) / (n-c-1)
    var_covar = float((1/(Xj.T @ Xj - Xj.T @ H @ Xj)) * (resid.T @ resid) / (n-c-1))

    beta = beta_vec[0]
    se_beta = np.sqrt(var_covar)

    results_dict['beta'] = beta[0]
    results_dict['se_beta'] = se_beta

    F_wald = np.power(beta/se_beta, 2.0)

    #results_dict['p_wald'] = (1-stats.f.cdf(x=F_wald, dfn=1, dfd=n-c-1))[0]
    results_dict['p_wald'] = stats.f.sf(F_wald, dfn=1, dfd=n-c-1)[0]

    return results_dict

class SampleIter:
    def __init__(self, X, y, Y, H, snps, c):
        self.X = X
        self.Y = Y
        self.y = y
        self.H = H
        self.c = c
        self.iteration = 0
        self.snps = snps if snps is not None else range(X.shape[1])
        self.p = X.shape[1]

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration < self.p:
            self.iteration += 1
            return self.X[:,self.iteration-1].reshape(-1,1), self.y, self.H, self.Y, self.snps[self.iteration-1], self.c

        else:
            raise StopIteration

def run_function_test(function, parameters):
    failed = False

    start = time.time()
    try:
        result = function(*parameters)
    except Exception as e:
        #print(e)
        console.print_exception(show_locals=False)
        failed = True
   
    diff = str(round(time.time() - start, 8))
    if failed:
        console.log(f"[red]Failed {function.__name__} - {diff} s")
        return 1
    else:
        console.log(f"[green]Passed {function.__name__} - {diff} s")
        return 0

def run_test_list(functions_and_args):
    failures = 0

    for function, arguments in functions_and_args:
        failures = failures + run_function_test(function, arguments)

    console.log(f"Failed {failures} out of {len(functions_and_args)} tests")

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
    _, mafs = calculate_allele_frequency_matrix(X)

    # Calculate the genetic relatedness matrix from GCTA
    # K_jk = 1/n * sum_i (x_ji - 2*p_i) * (x_ki - 2*p_i) / (2*p_i*(1-p_i))
    K = (X - 2*mafs) / np.sqrt(2*mafs*(1-mafs))
    K = K @ K.T / X.shape[1]    

    return K


DATADIR = "/net/mulan/data/WTCCC1/All"
dataset_list = [
        {
            'name'    : 'BD',
            'snps'    : os.path.join(DATADIR, "BD_imputed_1kgp_merged.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "BD_imputed_1kgp_merged.fam"),
            'kinship' : '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/gemma_runs/output/temp_k_bd.sXX.txt'
        },
        {
            'name'    : 'CAD',
            'snps'    : os.path.join(DATADIR, "CAD_imputed_1kgp_merged.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "CAD_imputed_1kgp_merged.fam"),
            'kinship' : '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/gemma_runs/output/temp_k_cad.sXX.txt'
        },
        {
            'name'    : 'CTRL',
            'snps'    : os.path.join(DATADIR, "Control_imputed_1kgp_merged.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "Control_imputed_1kgp_merged.fam"),
            'kinship' : '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/gemma_runs/output/temp_k_ctrl.sXX.txt'
        },
        {
            'name'    : 'HT',
            'snps'    : os.path.join(DATADIR, "HT_imputed_1kgp_merged.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "HT_imputed_1kgp_merged.fam"),
            'kinship' : '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/gemma_runs/output/temp_k_ht.sXX.txt'
        },
        {
            'name'    : 'RA',
            'snps'    : os.path.join(DATADIR, "RA_imputed_1kgp_merged.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "RA_imputed_1kgp_merged.fam"),
            'kinship' : '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/gemma_runs/output/temp_k_ra.sXX.txt'
        },
        {
            'name'    : 'T1D',
            'snps'    : os.path.join(DATADIR, "T1D_imputed_1kgp_merged.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "T1D_imputed_1kgp_merged.fam"),
            'kinship' : '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/gemma_runs/output/temp_k_t1d.sXX.txt'
        },
        {
            'name'    : 'T2D',
            'snps'    : os.path.join(DATADIR, "T2D_imputed_1kgp_merged.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "T2D_imputed_1kgp_merged.fam"),
            'kinship' : '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/gemma_runs/output/temp_k_t2d.sXX.txt'
        }
    ]

def run_dataset(dataset):
    dataset_name = dataset['name']

    print(f"Running GWAS for {dataset_name}...")

    print('Loading data...')
    if dataset_name == 'Homework3':
        snps = pd.read_csv(dataset['snps'])
        pheno = pd.read_csv(dataset['pheno'], sep='\t', index_col='IID')

        X = snps.values[:,7:].T.astype(np.float32)
        #X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    else:
        nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
        plink_data = Bed(dataset['snps'],
                         num_threads=nproc,
                         count_A1=False)

        # Genotype
        genotypes = plink_data.read(dtype=np.float32)

        snp_info = pd.read_csv('/net/mulan/data/WTCCC1/snpinfo/snpinfo.txt.gz', 
                               sep='\t',
                               header=None)
        
        snp_info.columns = ['CHR', 'SNP', 'rs', 'POS', 'ref', 'alt']

        rs_nums = np.array(pd.read_csv('/net/mulan/data/WTCCC1/All/chrall.rs.txt',
                              header=None).values.reshape(-1))

        # Subset snp_info to only include SNPs with rs num in the first column of rs_nums
        # Then reorder snp_info to match the order of rs_nums
        print('Subsetting genotypes to only include SNPs with rs num in the first column of rs_nums...')
        _, rs_ind, snp_ind  = np.intersect1d(ar1=rs_nums, ar2=snp_info['rs'], assume_unique=False, return_indices=True)
        #X = genotypes.val[:, np.isin(rs_nums, snp_info['rs'], assume_unique=True)]
        X = genotypes.val[:, rs_ind]
        
        print('Subsetting snp_info to only include SNPs with rs num in the first column of rs_nums...')
        #snp_info = snp_info[snp_info['rs'].isin(rs_nums, assume_unique=True)]
        snp_info = snp_info.loc[snp_ind]

        del genotypes
        del plink_data


        # Reorder snp_info to match the order of sid
        print('Reordering snp_info to match order of sid...')
        #snp_info = snp_info.reindex(snp_info['rs'].map(dict(zip(rs_nums, range(len(rs_nums)))))).reset_index(drop=True)


        # Subset to just those in rs_nums
        snp_info = snp_info.set_index('rs').loc[rs_nums[rs_ind]].reset_index()

        print('Number of Initial SNPs:', X.shape[1])
        print('Number of Individuals:', X.shape[0])
        #print('Average MAF:', np.mean(X.mean(axis=0)) / 2)
        #print('Average STD:', np.mean(X.std(axis=0)))
        print('SNP Info SNPs:', snp_info.shape)

        # Remove SNPs with std of 0 from snp_info and X
        X_std = X.std(axis=0)
        snp_info = snp_info.iloc[X_std > 0,:]
        X = X[:, X_std > 0]
        del X_std

        print('Number of SNPs after removing SNPs with std of 0:', X.shape[1])

        # # Compute minor allele frequency >= 0.02
        # X_mean = X.mean(axis=0)
        # maf = (X_mean >= 0.01 * 2) & ((1-X_mean) >= 0.01 * 2)

        # # Remove SNPs with MAF < 0.01
        # X = X[:, maf]
        # snp_info = snp_info[maf]

        # Print number of SNPs
        print(f"Number of SNPs that pass QC: {X.shape[1]}")

        if X.shape[1] == 0:
            warnings.warn(f'No SNPs passed QC for {dataset_name}!')
            return

        #X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        #print('Standardizing X...')
        #X = StandardScaler().fit_transform(X)

        #snps = snp_info[['CHR', 'BP', 'SNP']]

        # Rename BP to pos, CHR to chr, and SNP to rs
        #snps = snps.rename(columns={'BP': 'POS', 'CHR': 'CHR', 'SNP': 'SNP'})
        snps = snp_info[['CHR', 'POS', 'SNP']]

        pheno = pd.DataFrame({dataset['name'] : pd.read_csv(dataset['pheno'], sep=' ', header=None).values[:,5].astype(np.float32)})


        # If WRITEDATA is set to True, write the snps and pheno in BIMBAM format
        if WRITEDATA:
            '''
            This file contains genotype information. The first column is SNP id, the second and third columns
            are allele types with minor allele first, and the remaining columns are the posterior/imputed mean
            genotypes of different individuals numbered between 0 and 2. An example mean genotype file with
            two SNPs and three individuals is as follows:
            rs1, A, T, 0.02, 0.80, 1.50
            rs2, G, C, 0.98, 0.04, 1.00
            '''
            print('Writing genotypes to BIMBAM format...')
            geno_data = pd.DataFrame.from_dict({'SNP' : snps['SNP'].values,
                                                  'A' : ['A'] * X.shape[1],
                                                  'B' : ['B'] * X.shape[1]})
            pd.concat([geno_data, pd.DataFrame(X.T, columns=pheno.index)], axis=1).to_csv(os.path.join(OUTPUT, f"{dataset_name.lower()}_genotypes.tsv"),
                                                                                        sep='\t',
                                                                                        index=False,
                                                                                        header=False)
            
            del geno_data

            print('Writing phenotypes to BIMBAM format...')
            pheno.to_csv(os.path.join(OUTPUT, f"{dataset_name.lower()}_phenotypes.tsv"),
                         sep='\t',
                         index=False,
                         header=False)
            
            return


    n,p = X.shape

    if not LINEAR:
        print('Getting kinship matrix...')
        if dataset['kinship'] is None:
            #K = calculate_genetic_relatedness_matrix(X)
            K = X @ X.T / p
        else:
            if isinstance(dataset['kinship'], str):
                K = pd.read_csv(dataset['kinship'], header=None, sep='\t').values.astype(np.float32)
            else:
                K = dataset['kinship']

    #K = ((K - np.mean(K, axis=0)) / np.std(K, axis=0)).astype(np.float32)

    if PCS > 0:
        print('Running PCA...')
        pca = PCA(n_components=PCS)

        pcs = pca.fit_transform(X)

        # Write pcs to tsv with
        pd.DataFrame(pcs).to_csv(os.path.join(OUTPUT, f"{dataset_name.lower()}_pcs.tsv"), 
                                 sep='\t', 
                                 index=False,
                                 header=False)

        #pcs = pd.read_csv(os.path.join(OUTPUT, f"{dataset_name.lower()}_pcs.tsv"),
        # pcs = pd.read_csv(os.path.join("/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/updated_output", f"{dataset_name.lower()}_pcs.tsv"),
        #                     sep='\t',
        #                     header=None).values

        W = np.c_[pcs, np.ones(n)]
    else:
        W = np.ones((n,1)).astype(np.float32)

    for pheno_name in pheno.columns:
        p_vals_dict = {'SNP' : snps['SNP'].values}

        Y = pheno[pheno_name].values.reshape(-1,1).astype(np.float32)
        #Y = qnorm.quantile_normalize(Y, axis=1)
        #Y = (Y-np.mean(Y))/np.std(Y)
        Y = Y.reshape(-1,1)

        nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
        print(f"Using {nproc} processors")
        np.seterr(over="ignore")
        start = time.time()

        if LINEAR:
            data_results = run_gwas(Y, W, X, snps=snps['SNP'].values, verbose=1, nproc=nproc)
        else:
            data_results = lmm.pygemma(Y, X, W, K, snps=snps['SNP'].values, verbose=1, nproc=nproc)

        print('PyGemma Run Time:', time.time() - start, 's')
        print(data_results.head(10))

        # Save results
        data_results.to_csv(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_pygemma_results.csv"), index=False)

        p_vals_dict['pygemma'] = data_results['p_wald'].values

        # theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
        # pvals = np.sort(data_results['p_wald'])
        
        # plt.scatter(y=-np.log10(pvals+1/len(data_results)), x=-np.log10(theoretical))
        # plt.axline((0,0), slope=1, color='red')
        # plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        # plt.ylabel(r'Observed: $-\log_{10}(p)$')
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_qq.png"))
        # plt.clf()

        fig, ax = plot.qq_plot(data_results['p_wald'], scale='log')

        plt.title(f'QQ Plot for {pheno_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_qq.png"))

        # Close figure
        plt.close(fig)

        # # Manhatten plot adapted from https://stackoverflow.com/a/66062857
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

        # alpha = -np.log10(0.05/len(pvals))
        # with sns.color_palette():
        #     sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'], linewidth=0)
        # plt.axline((0,alpha), slope=0, color='red')
        # chrom_df=results_df.groupby('chr')['i'].median()
        # plt.xlabel('chr') 
        # plt.xticks(chrom_df,chrom_df.index)
        # plt.ylabel(r'$-\log_{10}(p)$')
        
        # # Remove legend
        # plt.legend([],[], frameon=False)

        # # Rotate xticks 90 degrees and make them smaller
        # plt.xticks(rotation=90, fontsize=8)


        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten.png"))
        # plt.clf()

        # Make a Manhattan plot with matplotlib
        fig, ax = plot.manhattan_plot(pval=data_results['p_wald'], 
                                        pos=snps['POS'].values, 
                                        chrom=snps['CHR'].values,
                                        beta=None,
                                        cutoff='bonferroni',
                                        scale='log',
                                        plotly=False)
        
        plt.title(f'Manhattan Plot for {pheno_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhattan.png"))

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
                            save_path=os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhattan.html"))


print('Running GWAS...')
for dataset in dataset_list:
    run_dataset(dataset)
        
