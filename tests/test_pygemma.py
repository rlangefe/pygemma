import time
import os

import numpy as np
import pandas as pd
import qnorm


from pygemma import lmm, pygemma_model

OUTPUT = "/net/mulan/home/rlangefe/gemma_work/pygemma/tests/output"

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

from rich.console import Console

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import warnings

sns.set_theme()

console = Console()

def run_function_test(function, parameters):
    failed = False

    start = time.time()
    
    try:
        result = function(*parameters)
    except Exception as e:
        print(e)
        failed = True
        
    diff = str(round(time.time() - start, 4))
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


def generate_test_matrices(n=1000, covars=10):
    K = np.random.uniform(size=(n, n))
    K = np.abs(np.tril(K) + np.tril(K, -1).T)
    K = np.dot(K, K.T)
    eigenVals, U = np.linalg.eig(K)
    W = np.random.rand(n, covars)
    x = np.random.choice([0,1,2], 
                        size=(n, 1),
                        replace=True)
    Y = np.random.rand(n, 1).reshape(-1,1)
    lam = 5
    tau = 10
    beta = np.random.rand(covars+1, 1)

    return x.astype(np.float32), Y.astype(np.float32), W.astype(np.float32), eigenVals.astype(np.float32), U.astype(np.float32), np.float32(lam), beta.astype(np.float32), np.float32(tau)

with console.status("[bold green]Running pyGEMMA Function Run Tests...") as status:

    # Seed tests
    np.random.seed(42)

    n = 100
    covars = 10

    # Initializing parameters for tests
    x, Y, W, eigenVals, U, lam, beta, tau = generate_test_matrices(n=n, covars=covars)
    W = np.c_[W,x]
    console.log(f'Test Parameters: n={n}, lam={lam}, tau={tau}')
    
    functions_and_args = [
                            (lmm.compute_Pc, [eigenVals, U, W, lam]),
                            (pygemma_model.compute_Pc, [eigenVals, U, W, lam]),
                            (lmm.likelihood_lambda, [lam, eigenVals, U, Y, W]),
                            (lmm.likelihood_derivative1_lambda, [lam, eigenVals, U, Y, W]),
                            (lmm.likelihood_derivative2_lambda, [lam, eigenVals, U, Y, W]),
                            (lmm.calc_lambda, [eigenVals, U, Y, W]),
                            (lmm.calc_lambda_restricted, [eigenVals, U, Y, W]),
                            (lmm.likelihood, [lam, tau, beta, eigenVals, U, Y, W]),
                            (lmm.likelihood_restricted, [lam, tau, eigenVals, U, Y, W]),
                            (lmm.likelihood_restricted_lambda, [lam, eigenVals, U, Y, W]),
                            (lmm.likelihood_derivative1_restricted_lambda, [lam, eigenVals, U, Y, W]),
                            (lmm.likelihood_derivative2_restricted_lambda, [lam, eigenVals, U, Y, W]),
                          ]
    
    run_test_list(functions_and_args)

DATADIR = os.path.join("..","data")
dataset_list = [
        {
            'name'    : 'Homework3',
            'snps'    : os.path.join(DATADIR, "test_data.csv"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "GD449.example.pheno.tsv"),
            'kinship' : None
        }
    ]

for dataset in dataset_list:
    dataset_name = dataset['name']

    snps = pd.read_csv(dataset['snps'])
    pheno = pd.read_csv(dataset['pheno'], sep='\t', index_col='IID')

    X = snps.values[:,7:].T.astype(np.float32)
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    n,p = X.shape

    if not dataset['kinship']:
        K = X @ X.T / p

    pca = PCA(n_components=2)

    pcs = pca.fit_transform(X)

    sample = np.random.choice(range(0,X.shape[1]), size=100, replace=False)
    X = X[:,sample]
    pheno_name = pheno.columns[0]
    Y = pheno[pheno_name].values.reshape(-1,1).astype(np.float32)
    Y = qnorm.quantile_normalize(Y, axis=1)
    #Y = (Y-np.mean(Y))/np.std(Y)

    # Likelihood tests
    x = X[:,0].reshape(-1,1)
    n = Y.shape[0]

    W = np.c_[np.ones(shape=(n, 1)), pcs].astype(np.float32)
    lam_vals = np.array([np.power(10.0, i) for i in np.arange(-4.0,4.0,1.0)], dtype=np.float32)
    eigenVals, U = np.linalg.eig(K)
    eigenVals = np.maximum(0, eigenVals)

    eigenVals = eigenVals.astype(np.float32)
    U = U.astype(np.float32)

    lik = [lmm.likelihood_restricted_lambda(l, eigenVals, U, Y, np.c_[W,x]) for l in lam_vals]
    lik_der1 = [lmm.likelihood_derivative1_restricted_lambda(l, eigenVals, U, Y, np.c_[W,x]) for l in lam_vals]
    plt.scatter(x=lam_vals, y=lik)
    plt.savefig(os.path.join(OUTPUT, "likelihood.png"))
    plt.clf()
    plt.scatter(x=lam_vals, y=lik_der1, c='red')
    plt.savefig(os.path.join(OUTPUT, "likelihood_derivative.png"))
    plt.clf()

    for pheno_name in pheno.columns:
        Y = pheno[pheno_name].values.reshape(-1,1).astype(np.float32)
        #Y = (Y-np.mean(Y))/np.std(Y)

        #with console.status(f"[bold green]Running pyGEMMA Tests - {dataset_name}: {pheno_name}...") as status:
        warnings.filterwarnings("error")
        data_results = lmm.pygemma(Y, X, W, K, snps=snps['SNP'].values[sample], verbose=1)
        print(data_results.head(10))

        theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
        pvals = np.sort(data_results['p_wald'])

        plt.scatter(y=-np.log10(pvals+1e-20), x=-np.log10(theoretical))
        plt.axline((0,0), slope=1, color='red')
        plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        plt.ylabel(r'Observed: $-\log_{10}(p)$')
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_qq.png"))
        plt.clf()

        # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        results_df = pd.DataFrame(
            {
            'pos'  : snps['POS'].values[sample],
            'pval' : -np.log10(data_results['p_wald']+1e-20),
            'chr' : snps['CHR'].values[sample]
            }
        )

        results_df = results_df.sort_values(['chr', 'pos'])
        results_df.reset_index(inplace=True, drop=True)
        results_df['i'] = results_df.index

        alpha = -np.log10(0.05/len(pvals))
        sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'])
        plt.axline((0,alpha), slope=0, color='red')
        chrom_df=results_df.groupby('chr')['i'].median()
        plt.xlabel('chr') 
        plt.xticks(chrom_df,chrom_df.index)
        plt.ylabel(r'$-\log_{10}(p)$')
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten.png"))
        plt.clf()

        pvals = np.sort(data_results['p_lrt'])

        plt.scatter(y=-np.log10(pvals+1e-20), x=-np.log10(theoretical))
        plt.axline((0,0), slope=1, color='red')
        plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        plt.ylabel(r'Observed: $-\log_{10}(p)$')
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_lrt_qq.png"))
        plt.clf()

        # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        results_df = pd.DataFrame(
            {
            'pos'  : snps['POS'].values[sample],
            'pval' : -np.log10(data_results['p_lrt']+1e-20),
            'chr' : snps['CHR'].values[sample]
            }
        )

        results_df = results_df.sort_values(['chr', 'pos'])
        results_df.reset_index(inplace=True, drop=True)
        results_df['i'] = results_df.index

        alpha = -np.log10(0.05/len(pvals))
        sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'])
        plt.axline((0,alpha), slope=0, color='red')
        chrom_df=results_df.groupby('chr')['i'].median()
        plt.xlabel('chr') 
        plt.xticks(chrom_df,chrom_df.index)
        plt.ylabel(r'$-\log_{10}(p)$')
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_lrt_manhatten.png"))
        plt.clf()

        plt.scatter(y=-np.log10(data_results['p_wald']+1e-20), x=-np.log10(data_results['p_lrt']+1e-20))
        plt.axline((0,0), slope=1, color='red')
        plt.xlabel(r'LRT: $-\log_{10}(p)$')
        plt.ylabel(r'Wald: $-\log_{10}(p)$')
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_lrt.png"))
        plt.clf()
        
