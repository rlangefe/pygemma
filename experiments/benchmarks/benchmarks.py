import time
import os

import numpy as np
import pandas as pd
import qnorm

from pygemma import lmm, pygemma_model
from pysnptools.snpreader import Bed

import random

from rich.console import Console
from rich.progress import track
from rich.traceback import Traceback

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Import sklearn simple imputer
from sklearn.impute import SimpleImputer

from scipy import stats

import gemma_utils

import warnings

import argparse

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
        
        try:
            design_design_inv = 1/np.sum(np.power(X[:,g],2.0)) #np.linalg.inv(X.T @ X)

            #beta_vec, resid, _, _ = np.linalg.lstsq(design_matrix, Y, rcond=None)
            beta_vec = design_design_inv * (X[:,g].T @ Y)
            resid = y - X[:,g].reshape(-1,1) * beta_vec - H @ y
            
            sigma_sq = np.sum(np.power(resid,2.0)) / (n-c-1)
            var_covar = float((1/(X[:,g].T @ X[:,g] - X[:,g].T @ H @ X[:,g])) * (resid.T @ resid) / (n-c-1))

            beta = beta_vec[0]
            se_beta = np.sqrt(var_covar)
            F_wald = np.power(beta/se_beta, 2.0)

            p_wald = 1-stats.f.cdf(x=F_wald, dfn=1, dfd=n-c-1)
        except Exception as e:
            beta = np.nan
            se_beta = np.nan
            p_wald = np.nan

        results_dict['beta'].append(beta)
        results_dict['se_beta'].append(se_beta)
        results_dict['p_wald'].append(p_wald)

    return pd.DataFrame(results_dict)

if __name__ == "__main__":
    # Make benchmark directory
    #BENCHMARK_DIR = '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/benchmark_test'
    BENCHMARK_DIR = '/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/benchmark_grid_test'

    VERBOSE = False

    num_runs = 2
    max_snps = 100000
    max_individuals = 10000
    max_covars = 20

    # Test run
    # num_runs = 3
    # max_snps = 10000
    # max_individuals = 1000
    # max_covars = 20

    if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
        RUNDIR = os.path.join(BENCHMARK_DIR, f'run_{os.environ.get("SLURM_ARRAY_TASK_ID")}_{os.environ.get("SLURM_JOB_ID")}')

        if not os.path.exists(RUNDIR):
            os.makedirs(RUNDIR)

        # Switch to run directory
        os.chdir(RUNDIR)

    else:
        # Remove results file if exists
        if os.path.exists(os.path.join(BENCHMARK_DIR, 'results.csv')):
            os.remove(os.path.join(BENCHMARK_DIR, 'results.csv'))

    param_array = []

    # Full Run with Covars
    # for _ in range(num_runs):
    #     for sample_size in list(range(50, max_individuals, 1000)) + [max_individuals]:
    #         for num_snps in list(range(20, max_snps, 1000)) + [max_snps]:
    #             for num_covars in list(range(0, max_covars, 5)) + [max_covars]:
    #                 param_array.append((sample_size, num_snps, num_covars))

    # Full Run without Covars
    for _ in range(num_runs):
        for sample_size in list(range(50, max_individuals, 2000)) + [max_individuals]:
            for num_snps in list(range(20, max_snps, 1000)) + [max_snps]:
                for num_covars in [0]:
                    param_array.append((sample_size, num_snps, num_covars))

    
    console.print(f'Running {len(param_array)} simulations')

    random.Random(42).shuffle(param_array)
                    
    # Get param indices for this part of the array
    if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID')) - 1
        task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT'))

        start_index = task_id * len(param_array) // task_count
        end_index = min(len(param_array), (task_id + 1) * len(param_array) // task_count)

        param_array = param_array[start_index:end_index]

    console.print(f'Running {len(param_array)} simulations on task {os.environ.get("SLURM_ARRAY_TASK_ID")}')

    def sim_function(sample_size, num_snps, num_covars):
        # Make "{RUNDIR}/subsample_data" if it doesn't exist
        if not os.path.exists(os.path.join(RUNDIR, 'subsample_data')):
            os.makedirs(os.path.join(RUNDIR, 'subsample_data'))


        time_dict ={
            'GCTA'                  : np.nan,
            'fastGWA'               : np.nan,
            'Regenie'               : np.nan,
            'GEMMA'                 : np.nan,
            'pyGEMMA'               : np.nan,
            'pyGEMMA - Grid Search' : np.nan,
            'linear'                : np.nan,
        }

        if VERBOSE:
            console.print(f'Running simulation with sample size {sample_size}, num_snps {num_snps}, num_covars {num_covars}')

        with open('/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/benchmarks/subsample.R', 'r') as f:
            subsample_script = f.read()

        subsample_script = f'''
N_subsample = {sample_size} 
N_SNP = {num_snps}
N_covar = {num_covars}

subsample_data_dir = "{RUNDIR}/subsample_data"

''' + subsample_script


        if VERBOSE:
            console.print(f'Writing R script to {os.path.join(RUNDIR, "subsample_mod.R")}')

        with open(os.path.join(RUNDIR, 'subsample_mod.R'), 'w') as f:
            f.write(subsample_script)

        # Run R script
        if VERBOSE:
            console.print(f'Running R script to generate subsample data')

        os.system(f'Rscript {os.path.join(RUNDIR, "subsample_mod.R")}')

        # Read timing.csv
        if VERBOSE:
            console.print(f'Reading timing.csv')

        timing = pd.read_csv(os.path.join(RUNDIR, 'subsample_data', 'timing.csv'), sep=',')

        # Get GCTA time
        time_dict['GCTA'] = timing['gcta'].values[0]

        # Get fastGWA time
        time_dict['fastGWA'] = timing['fastgwa'].values[0]

        # Get Regenie time
        time_dict['Regenie'] = timing['regenie'].values[0]

        # Get GEMMA time
        time_dict['GEMMA'] = timing['gemma'].values[0]

        if VERBOSE:
            console.print(f'Reading plink data')

        plink_data = Bed(os.path.join(RUNDIR, f'subsample_dataGEMMA_N_{sample_size}_rep_1.bed'), count_A1=False)

        # ID info which can be matched up eid
        geno_id = pd.Series(plink_data.iid[:, 1])

        # Genotype
        X = plink_data.read(dtype=np.float32).val.astype(np.float32)

        # Imputation
        if VERBOSE:
            console.print(f'Imputing missing values')

        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        if VERBOSE:
            console.print(f'Reading phenotype data')

        Y = pd.read_csv(os.path.join(RUNDIR, f'subsample_dataGEMMA_N_{sample_size}_rep_1_pheno.txt'), sep=' ', header=None).values.reshape(-1,1).astype(np.float32)

        if VERBOSE:
            console.print(f'Reading kinship data')

        K = pd.read_csv(os.path.join(RUNDIR, f'subsample_dataGEMMA_N_{sample_size}_rep_1.sXX.txt'), sep=' ', header=None).values.astype(np.float32)

        n,p = X.shape

        if VERBOSE:
            console.print(f'Reading covariate data')

        if num_covars > 0:
            pcs = pd.read_csv(os.path.join(RUNDIR, f'subsample_dataGEMMA_N_{sample_size}_rep_1_pc.txt'), sep=' ', header=None).values.astype(np.float32)
            W = np.c_[np.ones(shape=(n, 1)), pcs].astype(np.float32)
        else:
            W = np.ones(shape=(n, 1)).astype(np.float32)

        snps = pd.read_csv(os.path.join(RUNDIR, f'subsample_dataGCTA_N_{sample_size}_rep_1_SNP.txt'), sep=' ', header=None).values.reshape(-1)

        # Run linear regression
        if VERBOSE:
            console.print(f'Running linear regression')

        try:
            start = time.time()
            _ = run_gwas(Y, W, X, snps=snps, verbose=1)
            end = time.time()

            time_dict['linear'] = end - start
        except Exception as e:
            print(e)
            time_dict['linear'] = np.nan

        # Run PyGemma with grid search
        if VERBOSE:
            console.print(f'Running pyGEMMA with grid search')

        try:
            nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
            start = time.time()
            with np.errstate(over='ignore'):
                _ = lmm.pygemma(Y, X, W, K, snps=snps, verbose=1, nproc=nproc, grid=True)
            end = time.time()

            pygemma_grid_time = end - start
            
        except Exception as e:
            print(e)

            pygemma_grid_time = np.nan

        time_dict['pyGEMMA - Grid Search'] = pygemma_grid_time
        
        # Run PyGemma
        if VERBOSE:
            console.print(f'Running pyGEMMA')

        try:
            nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
            start = time.time()
            with np.errstate(over='ignore'):
                _ = lmm.pygemma(Y, X, W, K, snps=snps, verbose=1, nproc=nproc)
            end = time.time()

            # Cleanup
            os.system('rm *.txt')
            os.system('rm *.bim')
            os.system('rm *.bed')
            os.system('rm *.fam')
            os.system('rm *.log')
            os.system('rm *.regenie')
            os.system('rm *.list')
            os.system('rm *.loco')
            os.system('rm -rf subsample_data/*')

            pygemma_time = end - start
            
        except Exception as e:
            print(e)

            # Cleanup
            os.system('rm *.txt')
            os.system('rm *.bim')
            os.system('rm *.bed')
            os.system('rm *.fam')
            os.system('rm *.log')
            os.system('rm *.regenie')
            os.system('rm *.list')
            os.system('rm *.loco')
            os.system('rm -rf subsample_data/*')

            pygemma_time = np.nan

        time_dict['pyGEMMA'] = pygemma_time

        # For each entry in time_dict, if it is negative, set to 0.0
        for key in time_dict.keys():
            if time_dict[key] < 0.0:
                time_dict[key] = 0.0

        return time_dict

    for sample_size, num_snps, num_covars in param_array:
        start_sim = time.time()
        time_dict = sim_function(sample_size, num_snps, num_covars)
        sim_time = time.time() - start_sim

        nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))

        # Append results to file using pd dataframe
        if VERBOSE:
            console.print(f'Writing results to {os.path.join(RUNDIR, "results.csv")}')
        results = pd.DataFrame([[sample_size, num_snps, num_covars, nproc, time_dict['GEMMA'], time_dict['pyGEMMA'], time_dict['GCTA'], time_dict['fastGWA'], time_dict['Regenie'], time_dict['linear'], time_dict['pyGEMMA - Grid Search'], sim_time]],
                               columns=['sample_size', 'num_snps', 'num_covars', 'nproc', 'GEMMA', 'pyGEMMA', 'GCTA', 'fastGWA', 'Regenie', 'linear', 'pyGEMMA - Grid Search', 'sim_time'])
        results.to_csv(os.path.join(RUNDIR, 'results.csv'), index=False, mode='a', header=not os.path.exists(os.path.join(RUNDIR, 'results.csv')))
    
        
