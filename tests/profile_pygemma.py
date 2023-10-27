import time
import os

import numpy as np
import pandas as pd
import qnorm

import pstats, cProfile

from pygemma import lmm, pygemma_model

OUTPUT = "/net/mulan/home/rlangefe/gemma_work/pygemma/tests/output"

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

def run_gwas(Y,W,X, snps=None, verbose=0):
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

    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    # for col in range(W.shape[1]):
    #     if W[:,col].std() > 0:
    #         W[:,col] = (W[:,col] - W[:,col].mean()) / W[:,col].std()

    covar_list = [f'Covar{i}' for i in range(W.shape[1])]
    
    n = Y.shape[0]
    c = W.shape[1]

    H = W @ np.linalg.inv(W.T @ W) @ W.T

    # Regress out W matrix from Y
    y = Y.reshape(-1,1)
    Y = Y -  H @ Y

    # Regress out W matrix from each X
    X = X - H @ X

    for g in progress:
        if snps is not None:
            results_dict['SNPs'].append(snps[g])
        else:
            results_dict['SNPs'].append(g)
        
        design_design_inv = 1/np.sum(np.power(X[:,g],2.0)) #np.linalg.inv(X.T @ X)

        #beta_vec, resid, _, _ = np.linalg.lstsq(design_matrix, Y, rcond=None)
        beta_vec = design_design_inv * (X[:,g].T @ Y)
        resid = y - X[:,g].reshape(-1,1) * beta_vec - H @ y
        
        sigma_sq = np.sum(np.power(resid,2.0)) / (n-c-1)
        var_covar = float((1/(X[:,g].T @ X[:,g] - X[:,g].T @ H @ X[:,g])) * (resid.T @ resid) / (n-c-1))

        beta = beta_vec[0]
        se_beta = np.sqrt(var_covar)

        results_dict['beta'].append(beta)
        results_dict['se_beta'].append(se_beta)

        F_wald = np.power(beta/se_beta, 2.0)

        results_dict['p_wald'].append(1-stats.f.cdf(x=F_wald, dfn=1, dfd=n-c-1))

    return pd.DataFrame(results_dict)

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

DATADIR = os.path.join("..","data")
dataset_list = [
        {
            'name'    : 'Homework3',
            'snps'    : os.path.join(DATADIR, "test_data.csv"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "GD449.example.pheno.tsv"),
            'kinship' : None
        },
    ]

print('Running GWAS...')
for dataset in dataset_list:
    dataset_name = dataset['name']

    print('Loading data...')
    if dataset_name == 'Homework3':
        snps = pd.read_csv(dataset['snps'])
        pheno = pd.read_csv(dataset['pheno'], sep='\t', index_col='IID')

        X = snps.values[:,7:].T.astype(np.float32)
        #X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    else:
        snps = dataset['snps']
        pheno = dataset['pheno']

        X = snps.values
        #X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

        snp_names = snps.columns
        snps = snps.transpose()
        snps['SNP'] = snp_names

        # Extract POS and CHR from SNP column
        snps_values = snps['SNP'].str.split(':', expand=True)
        snps_values.columns = ['CHR', 'POS', 'REF', 'ALT']

        # Add to snps dataframe
        snps = pd.concat([snps_values, snps], axis=1)

    

    print('Running PCA...')
    pca = PCA(n_components=2)

    pcs = pca.fit_transform(X)

    #sample = range(0,X.shape[1]) 
    sample = np.random.choice(range(0,X.shape[1]), size=1000, replace=False)
    X = X[:,sample]
    
    n,p = X.shape

    W = np.c_[np.ones(shape=(n, 1)), pcs].astype(np.float32)


    print('Getting kinship matrix...')
    if dataset['kinship'] is None:
        K = calculate_genetic_relatedness_matrix(X)
        #K = X @ X.T / p
    else:
        if isinstance(dataset['kinship'], str):
            K = pd.read_csv(dataset['kinship'], header=None).values
        else:
            K = dataset['kinship']

    for pheno_name in pheno.columns:

        Y = pheno[pheno_name].values.reshape(-1,1).astype(np.float32)
        Y = qnorm.quantile_normalize(Y, axis=1)
        Y = (Y-np.mean(Y))/np.std(Y)
        Y = Y.reshape(-1,1)
        
        #data_results = lmm.pygemma(Y - H @ Y, X - H @ X, np.ones(shape=(n, 1)), K, snps=snps['SNP'].values[sample], verbose=1)
        nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
        print(f"Using {nproc} processors")
        #data_results = jax_pygemma.pygemma_jax(Y, X, W, K, snps=snps['SNP'].values[sample], verbose=1, nproc=nproc)
        
        x = X[:,0:1]
        lam = 0.1

        # Calculate eigenvalues and eigenvectors
        eigenVals, U = np.linalg.eigh(K)
        eigenVals = np.maximum(eigenVals, 0)

        pr = cProfile.Profile()
        pr.enable()

        lmm.precompute_mat(lam, eigenVals, U.T @ np.c_[W, x], U.T @ Y, True)

        pr.disable()
        pr.dump_stats('pygemma.prof')
        lmm.pygemma(Y, X, W, K, snps=snps['SNP'].values[sample], verbose=1, nproc=nproc)
        #data_results = lmm.pygemma(Y, X, W, K, snps=snps['SNP'].values[sample], verbose=1, nproc=nproc)
        exit(0)
        


        
