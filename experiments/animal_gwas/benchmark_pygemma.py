import time
import os

import numpy as np
import pandas as pd
import qnorm

from pygemma import lmm, pygemma_model

from rich.console import Console
from rich.progress import track
from rich.traceback import Traceback

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from scipy import stats
from sklearn.impute import SimpleImputer

# Import standard scalar
from sklearn.preprocessing import StandardScaler

import gemma_utils

import warnings

import argparse

import multiprocessing as mp

def run_gwas_lm(Y, W, X, snps=None, verbose=0, nproc=1):
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

    n = y.shape[0]

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

    F_wald = np.power(beta/se_beta, 2.0)

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

if __name__ =='__main__':
    
    # Set TRIAL by job array
    if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
        TRIAL = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    else:
        TRIAL = 1

    OUTPUT = os.path.join(os.getcwd(), f'run_results_{TRIAL}')

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
    genotypes[genotypes.columns[3:]] = genotypes[genotypes.columns[3:]].fillna(genotypes[genotypes.columns[3:]].mean())

    # Convert the genotype columns to floats
    X = genotypes[genotypes.columns[3:]].values.astype(np.float32).T

    # Impute with mean of each column of X with sklearn
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X)
    X = imp_mean.transform(X)
    
    # Scale
    X = StandardScaler().fit_transform(X)

    # Convert the SNP columns to strings
    snps = genotypes[genotypes.columns[0]].astype(str)
    snps.columns = ['rs']

    # Read annotations file tsv
    print('Reading annotations file...')
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

    # Get index of non nan pos or chr
    not_nan_index = snps[~snps['POS'].isnull()].index

    # Subset snps and X to remove nan pos or chr
    snps = snps.iloc[not_nan_index,:]
    X = X[:,not_nan_index]

    # Remove SNPs with zero standard deviation
    sd = np.std(X, axis=0)

    # convert POS and CHR to int
    snps['POS'] = snps['POS'].astype(int)
    snps['CHR'] = snps['CHR'].astype(int)

    # Randomly sample SNPs
    #selection = np.random.choice(X.shape[1], 2000, replace=False)
    selection = range(0, X.shape[1])
    X = X[:, selection]
    snps = snps.iloc[selection,:]

    # Calculate the genetic relatedness matrix
    print('Calculating genetic relatedness matrix...')
    K = X @ X.T / X.shape[1]

    nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
    print(f"Using {nproc} processors")

    for trial in [TRIAL]:
        for PCS in range(0,30,5):
            # Initialize covariate matrix
            W = np.ones((X.shape[0], 1))

            if PCS > 0:
                # Calculate the principal components
                print('Calculating principal components...')
                pca = PCA(n_components=PCS)
                pcs = pca.fit_transform(X)

                # Add the principal components to the covariate matrix
                W = np.hstack((W, pcs))

            # Run GWAS for each phenotype
            for p,pheno in enumerate(phenotypes.columns):
                print(f"Running GWAS for {pheno}...")
                results_dict = {
                    'covars': PCS+1,
                    'pheno': pheno,
                    'pygemma_time': np.nan,
                    'gemma_time': np.nan,
                    'gcta_time': np.nan,
                    'linear' : np.nan,
                    'trial': trial,
                }


                # Convert the phenotype to a numpy array
                Y = phenotypes[pheno].values.astype(np.float32).reshape(-1, 1)

                # Run GWAS
                print('Running GWAS with pygemma...')
                start = time.time()
                data_results = lmm.pygemma(X=X, Y=Y, W=W, K=K, snps=snps['rs'], verbose=1)
                pygemma_time = time.time() - start

                # Run with GEMMA
                print('Running GEMMA...')
                gemma_results, gemma_time = gemma_utils.run_gemma(f'gemma_run_{trial}',
                                                                    pd.DataFrame(X, columns=snps['SNPs'].values),
                                                                    Y,
                                                                    W,
                                                                    K)

                # Run with GCTA
                print('Running GCTA...')
                gcta_results, gcta_time = gemma_utils.run_gcta(f'gcta_run_{trial}',
                                                                    pd.DataFrame(X, columns=snps['SNPs'].values),
                                                                    p+5,
                                                                    W,
                                                                    K)
                
                # Run with linear regression
                # print('Running linear regression...')
                # start = time.time()
                # linear_results = run_gwas_lm(Y, W, X, snps=snps['SNPs'].values, verbose=1, nproc=nproc)
                # linear_time = time.time() - start

                # Add results to results dictionary
                results_dict['pygemma_time'] = pygemma_time
                results_dict['gemma_time'] = gemma_time
                results_dict['gcta_time'] = gcta_time
                #results_dict['linear'] = linear_time



                # If output file exists, append results to it, otherwise create it
                results_df = pd.DataFrame.from_dict([results_dict])

                if os.path.exists(os.path.join(OUTPUT, 'results.csv')):
                    results_df.to_csv(os.path.join(OUTPUT, 'results.csv'), mode='a', header=False, index=False)
                else:
                    results_df.to_csv(os.path.join(OUTPUT, 'results.csv'), index=False)

        


