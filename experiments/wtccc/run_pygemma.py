import time
import os

import numpy as np
import pandas as pd
import qnorm
from sklearn.preprocessing import StandardScaler

from pygemma import lmm, pygemma_model

from pysnptools.snpreader import Bed

# Get output from env
OUTPUT = str(os.environ.get('OUTPUT'))

# Extract PCS from env if it exists
PCS = int(os.environ.get('PCS')) if os.environ.get('PCS') is not None else 0

LINEAR = True if int(os.environ.get('LINEAR')) == 1 else False

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

import matplotlib.animation as animation

# Function to make a gwas GIF using matplotlib that fills in incrementally
def make_animated_gwas(results_df, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Manhattan Plot for GWAS")

    results_df = results_df.sort_values(['chr', 'pos'])
    results_df.reset_index(inplace=True, drop=True)
    results_df['i'] = results_df.index

    alpha = -np.log10(0.05/len(pvals))
    with sns.color_palette():
        sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'])
    plt.axline((0,alpha), slope=0, color='red')
    chrom_df=results_df.groupby('chr')['i'].median()
    plt.xlabel('chr') 
    plt.xticks(chrom_df,chrom_df.index)
    plt.ylabel(r'$-\log_{10}(p)$')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten.png"))
    plt.clf()

    # Break snps into 50 approximately equal sized groups and assign it an index
    groups = np.array_split(results_df[['i','']], 50)

    def update(frame):
        # Simulate data points (replace this with your actual data)
        num_points = 100
        x = np.random.randint(1, 23, size=num_points)
        y = -np.log10(np.random.rand(num_points) * 0.001)
        
        ax.scatter(x, y, c='blue', s=10, alpha=0.5)
        ax.set_xlim(0, 23)
        ax.set_ylim(0, max(y) + 1)




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
        'F_wald': [],
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
        'F_wald': np.nan,
        'p_wald': np.nan,
    }

    results_dict['SNPs'] = snp

    Xj = Xj.reshape(-1,1)

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

    F_wald = np.power(np.float64(beta)/np.float64(se_beta), 2.0)

    results_dict['F_wald'] = F_wald[0]

    results_dict['p_wald'] = (1-stats.f.cdf(x=F_wald, dfn=1, dfd=n-c-1))[0]

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


DATADIR = "/net/mulan/data/WTCCC/processed"
dataset_list = [
        {
            'name'    : 'BD',
            'snps'    : os.path.join(DATADIR, "bd_plink.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "bd_plink.fam"),
            'kinship' : None
        },
        {
            'name'    : 'CAD',
            'snps'    : os.path.join(DATADIR, "cad_plink.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "cad_plink.fam"),
            'kinship' : None
        },
        {
            'name'    : 'CTRL',
            'snps'    : os.path.join(DATADIR, "ctrl_plink.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "ctrl_plink.fam"),
            'kinship' : None
        },
        {
            'name'    : 'HT',
            'snps'    : os.path.join(DATADIR, "ht_plink.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "ht_plink.fam"),
            'kinship' : None
        },
        {
            'name'    : 'RA',
            'snps'    : os.path.join(DATADIR, "ra_plink.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "ra_plink.fam"),
            'kinship' : None
        },
        {
            'name'    : 'T1D',
            'snps'    : os.path.join(DATADIR, "t1d_plink.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "t1d_plink.fam"),
            'kinship' : None
        },
        {
            'name'    : 'T2D',
            'snps'    : os.path.join(DATADIR, "t2d_plink.bed"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "t2d_plink.fam"),
            'kinship' : None
        }
    ]

print('Running GWAS...')
for dataset in dataset_list:
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

        # ID info which can be matched up eid
        geno_id = pd.Series(plink_data.iid[:, 1])

        # Sample individuals
        individuals = range(len(plink_data.iid))
        #individuals = np.random.choice(range(len(plink_data.iid)), 100, replace=False)
        geno_id = geno_id[individuals]

        #sample = np.random.choice(range(len(plink_data.sid)), 1000, replace=False)

        # Genotype
        genotypes = plink_data.read(dtype=np.float32)

        snp_info = pd.read_csv('/net/mulan/data/WTCCC/processed/WTCCC_plink_chrannot.cat.txt.gz', sep='\t')

        X = genotypes.val

        print('Number of Initial SNPs:', X.shape[1])
        print('Number of Individuals:', X.shape[0])
        print('Average MAF:', np.mean(X.mean(axis=0)) / 2)
        print('Average STD:', np.mean(X.std(axis=0)))

        # Remove SNPs with std of 0 from snp_info and X
        X_std = X.std(axis=0)
        snp_info = snp_info[X_std > 0]
        X = X[:, X_std > 0]

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
            continue

        #X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        print('Standardizing X...')
        X = StandardScaler().fit_transform(X)


        snps = snp_info[['CHR', 'BP', 'SNP']]

        # Rename BP to pos, CHR to chr, and SNP to rs
        snps = snps.rename(columns={'BP': 'POS', 'CHR': 'CHR', 'SNP': 'SNP'})

        pheno = pd.DataFrame({dataset['name'] : pd.read_csv(dataset['pheno'], sep='\t', header=None).values[:,5].astype(np.float32)})

    n,p = X.shape

    if not LINEAR:
        print('Getting kinship matrix...')
        if dataset['kinship'] is None:
            #K = calculate_genetic_relatedness_matrix(X)
            K = X @ X.T / p
        else:
            if isinstance(dataset['kinship'], str):
                K = pd.read_csv(dataset['kinship'], header=None).values
            else:
                K = dataset['kinship']

    #K = ((K - np.mean(K, axis=0)) / np.std(K, axis=0)).astype(np.float32)

    if PCS > 0:
        print('Running PCA...')
        pca = PCA(n_components=PCS)

        pcs = pca.fit_transform(X)

        W = np.c_[pcs, np.ones(n)]
    else:
        W = np.ones((n,1))

    for pheno_name in pheno.columns:
        p_vals_dict = {'SNP' : snps['SNP'].values}

        Y = pheno[pheno_name].values.reshape(-1,1).astype(np.float32)
        #Y = qnorm.quantile_normalize(Y, axis=1)
        #Y = (Y-np.mean(Y))/np.std(Y)
        Y = Y.reshape(-1,1)
        
        # Run GEMMA
        # print('Running GEMMA...')
        # data_results, total_time = gemma_utils.run_gemma('gemma_run',
        #                                                     pd.DataFrame(X, columns=snps['SNP'].values),
        #                                                     #pd.DataFrame(X[:,0:1], columns=snps['SNP'].values[0:1]),
        #                                                     Y,
        #                                                     W,
        #                                                     K)

        # p_vals_dict['gemma'] = data_results['p_wald'].values
        
        # print('GEMMA Run Time:', total_time, 's')
        # print(data_results.head(10))
        # theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
        # pvals = np.sort(data_results['p_wald'])
        
        # plt.scatter(y=-np.log10(pvals+1/len(data_results)), x=-np.log10(theoretical))
        # plt.axline((0,0), slope=1, color='red')
        # plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        # plt.ylabel(r'Observed: $-\log_{10}(p)$')
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_gemma_wald_qq.png"))
        # plt.clf()

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
        #     sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'])
        # plt.axline((0,alpha), slope=0, color='red')
        # chrom_df=results_df.groupby('chr')['i'].median()
        # plt.xlabel('chr') 
        # plt.xticks(chrom_df,chrom_df.index)
        # plt.ylabel(r'$-\log_{10}(p)$')
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_gemma_wald_manhatten.png"))
        # plt.clf()

        nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
        print(f"Using {nproc} processors")
        np.seterr(over="ignore")
        start = time.time()

        if LINEAR:
            #data_results = run_gwas(Y, W, X, snps=snps['SNP'].values, verbose=1, nproc=nproc)
            data_results = run_gwas(Y, W, X, snps=snps['SNP'].values, verbose=1, nproc=nproc)
        else:
            #data_results = lmm.pygemma(Y, X, W, K, snps=snps['SNP'].values, verbose=1, nproc=nproc)
            data_results = lmm.pygemma(Y, X, W, K, snps=snps['SNP'].values, verbose=1, nproc=nproc)

        print('PyGemma Run Time:', time.time() - start, 's')
        print(data_results.head(10))

        # Save results
        data_results.to_csv(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_pygemma_results.csv"), index=False)

        p_vals_dict['pygemma'] = data_results['p_wald'].values

        theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
        pvals = np.sort(data_results['p_wald'])
        
        plt.scatter(y=-np.log10(pvals+1/len(data_results)), x=-np.log10(theoretical))
        plt.axline((0,0), slope=1, color='red')
        plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        plt.ylabel(r'Observed: $-\log_{10}(p)$')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_qq.png"))
        plt.clf()

        # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        results_df = pd.DataFrame(
            {
            'pos'  : snps['POS'].values,
            'pval' : -np.log10(data_results['p_wald']+1/len(data_results)),
            'chr' : snps['CHR'].values
            }
        )

        results_df = results_df.sort_values(['chr', 'pos'])
        results_df.reset_index(inplace=True, drop=True)
        results_df['i'] = results_df.index

        alpha = -np.log10(0.05/len(pvals))
        with sns.color_palette():
            sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'], linewidth=0)
        plt.axline((0,alpha), slope=0, color='red')
        chrom_df=results_df.groupby('chr')['i'].median()
        plt.xlabel('chr') 
        plt.xticks(chrom_df,chrom_df.index)
        plt.ylabel(r'$-\log_{10}(p)$')
        
        # Remove legend
        plt.legend([],[], frameon=False)

        # Rotate xticks 90 degrees and make them smaller
        plt.xticks(rotation=90, fontsize=8)


        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten.png"))
        plt.clf()        

        # ##### Fixed Effects only #####
        # data_results = run_gwas(Y, W, X, snps=snps['SNP'].values, verbose=1)
        # median_p = np.median(data_results['p_wald'].values)

        # median_chisq = stats.chi2.ppf(1-median_p, 1)

        # lambda_gc = median_chisq/stats.chi2.ppf(0.5, 1)

        # print(f'Lambda GC: {lambda_gc}')
        # data_results['p_wald_gc'] = 1-stats.chi2.cdf(stats.chi2.ppf(1-data_results['p_wald'] , 1)/lambda_gc, df=1)
        # print(data_results.head(10))

        # p_vals_dict['linear_model'] = data_results['p_wald'].values
        # p_vals_dict['linear_model_gc'] = data_results['p_wald_gc'].values

        # theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
        # pvals = np.sort(data_results['p_wald'])
        
        # plt.scatter(y=-np.log10(pvals+1/len(data_results)), x=-np.log10(theoretical))
        # plt.axline((0,0), slope=1, color='red')
        # plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        # plt.ylabel(r'Observed: $-\log_{10}(p)$')
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_qq_fixed.png"))
        # plt.clf()

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
        #     sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'])
        # plt.axline((0,alpha), slope=0, color='red')
        # chrom_df=results_df.groupby('chr')['i'].median()
        # plt.xlabel('chr') 
        # plt.xticks(chrom_df,chrom_df.index)
        # plt.ylabel(r'$-\log_{10}(p)$')
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten_fixed.png"))
        # plt.clf()

        # # GC
        # theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
        # pvals = np.sort(data_results['p_wald_gc'])
        
        # plt.scatter(y=-np.log10(pvals+1/len(data_results)), x=-np.log10(theoretical))
        # plt.axline((0,0), slope=1, color='red')
        # plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        # plt.ylabel(r'Observed: $-\log_{10}(p)$')
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_qq_fixed_gc.png"))
        # plt.clf()

        # # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        # results_df = pd.DataFrame(
        #     {
        #     'pos'  : snps['POS'].values,
        #     'pval' : -np.log10(data_results['p_wald_gc']+1/len(data_results)),
        #     'chr' : snps['CHR'].values
        #     }
        # )

        # results_df = results_df.sort_values(['chr', 'pos'])
        # results_df.reset_index(inplace=True, drop=True)
        # results_df['i'] = results_df.index

        # alpha = -np.log10(0.05/len(pvals))
        # with sns.color_palette():
        #     sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'])
        # plt.axline((0,alpha), slope=0, color='red')
        # chrom_df=results_df.groupby('chr')['i'].median()
        # plt.xlabel('chr') 
        # plt.xticks(chrom_df,chrom_df.index)
        # plt.ylabel(r'$-\log_{10}(p)$')
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten_fixed_gc.png"))
        # plt.clf()

        # # Convert p_vals_dict to dataframe and save
        # p_vals_df = pd.DataFrame(p_vals_dict)

        # p_vals_df.to_csv(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_pvals.csv"), index=False)

        # # Grid of plots for methods
        # # For gemma, emma, and pygemma plot pairwise the p-values off diagonal
        # # Plot the QQ plot on the diagonal
        
        # #method_list = ['gemma', 'emma', 'pygemma']
        # method_list = ['gemma', 'pygemma']

        # figure, axes = plt.subplots(nrows=len(method_list), ncols=len(method_list), figsize=(20,20))

        # # for idx_1, method1 in enumerate(['gemma', 'emma', 'pygemma']):
        # #     for idx_2, method2 in enumerate(['gemma', 'emma', 'pygemma']):
        # for idx_1, method1 in enumerate(method_list):
        #     for idx_2, method2 in enumerate(method_list):
        #         if idx_1 == idx_2:
        #             # Plot QQ plot for method
        #             pvals = np.sort(p_vals_df[method1])
        #             theoretical = np.linspace(1/len(pvals),1.0,len(pvals))

        #             # Seaborn qq plot
        #             sns.scatterplot(x=-np.log10(pvals+1/len(p_vals_df)), y=-np.log10(theoretical), ax=axes[idx_1, idx_2])
        #             axes[idx_1, idx_2].axline((0,0), slope=1, color='red')
        #             axes[idx_1, idx_2].set_xlabel(r'Theoretical: $-\log_{10}(p)$')
        #             axes[idx_1, idx_2].set_ylabel(r'Observed: $-\log_{10}(p)$')
                    
        #             # Set title
        #             axes[idx_1, idx_2].set_title(f"{method1} QQ Plot")
                    
        #         else:
        #             # Plot p-value scatter plot with seaborn
        #             #sns.scatterplot(x=-np.log10(np.maximum(p_vals_df[method1], 1e-25)), y=-np.log10(np.maximum(p_vals_df[method2], 1e-25)), ax=axes[idx_1, idx_2])
        #             sns.scatterplot(x=-np.log10(p_vals_df[method1]), y=-np.log10(p_vals_df[method2]), ax=axes[idx_1, idx_2])
        #             axes[idx_1, idx_2].set_xlabel(f"{method1}")
        #             axes[idx_1, idx_2].set_ylabel(f"{method2}")

        #             # Plot diagonal line
        #             axes[idx_1, idx_2].axline((0,0), slope=1, color='red')

        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_pval_comparison.png"))
        # figure.clf()
        # plt.close(figure)
        # plt.clf()

        # # Reset plotting
        # plt.rcParams.update(plt.rcParamsDefault)
        # sns.set_theme()


        
