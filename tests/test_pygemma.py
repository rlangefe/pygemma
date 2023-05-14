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
from rich.progress import track

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from scipy import stats

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

    for col in range(W.shape[1]):
        if W[:,col].std() > 0:
            W[:,col] = (W[:,col] - W[:,col].mean()) / W[:,col].std()

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


def generate_test_matrices(n=1000, covars=10, seed=42):
    np.random.seed(seed)
    K = np.random.uniform(size=(n, n))
    K = np.abs(np.tril(K) + np.tril(K, -1).T)
    K = np.dot(K, K.T)
    eigenVals, U = np.linalg.eig(K)
    W = np.random.rand(n, covars)
    W = np.c_[W, np.ones(n)]
    x = np.random.choice([0,1,2], 
                        size=(n, 1),
                        replace=True)
    Y = np.random.rand(n, 1).reshape(-1,1)
    lam = 5
    tau = 10
    beta = np.random.rand(covars+2, 1)

    return x.astype(np.float32), Y.astype(np.float32), W.astype(np.float32), eigenVals.astype(np.float32), U.astype(np.float32), np.float32(lam), beta.astype(np.float32), np.float32(tau)

with console.status("[bold green]Running pyGEMMA Function Run Tests...") as status:

    # Seed tests
    np.random.seed(42)

    n = 100
    covars = 10

    # Initializing parameters for tests
    x, Y, W, eigenVals, U, lam, beta, tau = generate_test_matrices(n=n, covars=covars)
    W = np.c_[W,x]
    Px = pygemma_model.compute_Pc(eigenVals, U, W, lam)
    print(Y.T @ Px @ Y)
    print(lmm.compute_at_Pi_b(lam, W.shape[1],
                      lam*eigenVals + 1.0,
                      U @ W,
                      U @ Y,
                      U @ Y))

    x = (U @ x).reshape(-1)
    Y = U @ Y
    W = U @ W

    for lam in [1e-3, 5.0, 400, 1e3, 1e5]:
        console.log(f'Test Parameters: n={n}, lam={lam}, tau={tau}')
        
        functions_and_args = [
                                # (lmm.compute_Pc, [eigenVals, W, lam]),
                                # (pygemma_model.compute_Pc, [eigenVals, W, lam]),
                                (lmm.likelihood_lambda, [lam, eigenVals, Y, W]),
                                (lmm.likelihood_derivative1_lambda, [lam, eigenVals, Y, W]),
                                (lmm.likelihood_derivative2_lambda, [lam, eigenVals, Y, W]),
                                (lmm.calc_lambda, [eigenVals, Y, W]),
                                (lmm.calc_lambda_restricted, [eigenVals, Y, W]),
                                (lmm.likelihood, [lam, tau, beta, eigenVals, Y, W]),
                                (lmm.likelihood_restricted, [lam, tau, eigenVals, Y, W]),
                                (lmm.likelihood_restricted_lambda, [lam, eigenVals, Y, W]),
                                (lmm.likelihood_derivative1_restricted_lambda, [lam, eigenVals, Y, W]),
                                (lmm.likelihood_derivative2_restricted_lambda, [lam, eigenVals, Y, W]),
                                (lmm.trace_Pi, [lam, W.shape[1], lam*eigenVals+1.0, W]),
                                (lmm.trace_Pi_Pi, [lam, W.shape[1], lam*eigenVals+1.0, W]),
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
    X = X

    n,p = X.shape

    if not dataset['kinship']:
        K = X @ X.T / p
    else:
        K = pd.read_csv(dataset['kinship'], header=None).values

    #K = ((K - np.mean(K, axis=0)) / np.std(K, axis=0)).astype(np.float32)

    pca = PCA(n_components=2)

    pcs = pca.fit_transform(X)

    sample = range(0,X.shape[1]) 
    #sample = np.random.choice(range(0,X.shape[1]), size=1000, replace=False)
    X = X[:,sample]
    pheno_name = pheno.columns[0]
    Y = pheno[pheno_name].values.reshape(-1,1).astype(np.float32)
    #Y = qnorm.quantile_normalize(Y, axis=1)
    #Y = (Y-np.mean(Y))/np.std(Y)
    #print(Y.mean(), Y.std())
    # Likelihood tests
    x = X[:,2].reshape(-1,1)
    #x = (x - np.mean(x))/np.std(x)
    n = Y.shape[0]
    print(pcs.mean(axis=0), pcs.std(axis=0))

    W = np.c_[np.ones(shape=(n, 1)), pcs].astype(np.float32)
    
    #W = np.ones(shape=(n, 1)).astype(np.float32)
    lam_vals = np.array([np.power(10.0, i) for i in np.arange(-5.0,5.5,0.01)], dtype=np.float32)
    eigenVals, U = np.linalg.eig(K)
    eigenVals = np.maximum(0, eigenVals)

    eigenVals = eigenVals.astype(np.float32)
    U = U.astype(np.float32)

    # l_star = 117489.7578125
    # Px = pygemma_model.compute_Pc(eigenVals, U, np.c_[W, x], l_star)
    # H_inv = U.T @ (1.0/(l_star*eigenVals + 1.0)[:, np.newaxis] * U)#np.linalg.inv(l_star * K + np.eye(n))
    # print(Y.T @ (H_inv - H_inv @ np.c_[W, x] @ np.linalg.inv(np.c_[W, x].T @ H_inv @ np.c_[W, x]) @ np.c_[W, x].T @ H_inv) @ Y)
    # print(Y.T @ Px @ Y)
    # print(lmm.compute_at_Pi_b(l_star, np.c_[W, x].shape[1],
    #                   l_star*eigenVals + 1.0,
    #                   U @ np.c_[W, x],
    #                   U @ Y,
    #                   U @ Y))

    Y_star = U @ Y
    W_star = U @ W
    x_star = U @ x
    W_x_star = np.c_[W_star, x_star]
    x_star = x_star.reshape(-1,1)

    lik = [lmm.likelihood_restricted_lambda(l, eigenVals, Y_star, W_x_star) for l in lam_vals]
    # warnings.filterwarnings('error')
    # lik = []
    # for l in lam_vals:
    #     try:
    #         lik.append(lmm.likelihood_restricted_lambda(l, eigenVals, Y_star, W_x_star))
    #     except Warning as e:
    #         mod_eig = l*eigenVals + 1.0
    #         print(e)
    #         print('Error: ', l)
    #         print(np.any(l*eigenVals + 1.0 < 0))
    #         print(np.sum(np.log(l*eigenVals + 1.0)))
    #         print(lmm.compute_at_Pi_b(lam, W.shape[1], l*eigenVals + 1.0, W, Y, Y))
    #         print(np.log(lmm.compute_at_Pi_b(lam, W.shape[1], l*eigenVals + 1.0, W, Y, Y)))
    #         print(np.linalg.slogdet(W.T @ W))
    #         print(np.linalg.slogdet(W.T @ (W / mod_eig[:, np.newaxis])))
    #         exit(0)

    lik_der1 = [lmm.likelihood_derivative1_restricted_lambda(l, eigenVals, Y_star, W_x_star) for l in lam_vals]
    lik_der2 = [lmm.likelihood_derivative2_restricted_lambda(l, eigenVals, Y_star, W_x_star) for l in lam_vals]
    lam_temp = lmm.calc_lambda_restricted(eigenVals, Y_star, W_x_star)
    print('Best Lambda: ', lam_temp)
    print('Results: ', lmm.calc_beta_vg_ve_restricted(eigenVals, W_star, x_star, lam_temp, Y_star))
    print('Likelihood Min and Max: ', np.min(lik), np.max(lik))
    print('Likelihood Derivative 1 Min and Max: ', np.min(lik_der1), np.max(lik_der1))
    print('Likelihood Derivative 2 Min and Max: ', np.min(lik_der2), np.max(lik_der2))
    plt.scatter(x=lam_vals, y=lik)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "likelihood.png"))
    plt.clf()
    plt.scatter(x=lam_vals, y=lik_der1, c='red')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "likelihood_derivative.png"))
    plt.clf()
    plt.scatter(x=lam_vals, y=lik_der2, c='red')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "likelihood_derivative2.png"))
    plt.clf()

    for pheno_name in pheno.columns:
        Y = pheno[pheno_name].values.reshape(-1,1).astype(np.float32)
        Y = qnorm.quantile_normalize(Y, axis=1)
        #(Y-np.mean(Y))/np.std(Y)

        #Y = (Y - Y.mean())/Y.std()
        Y = Y.reshape(-1,1)
        #with console.status(f"[bold green]Running pyGEMMA Tests - {dataset_name}: {pheno_name}...") as status:
        #warnings.filterwarnings("error")
        
        #data_results = lmm.pygemma(Y - H @ Y, X - H @ X, np.ones(shape=(n, 1)), K, snps=snps['SNP'].values[sample], verbose=1)
        nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
        print(f"Using {nproc} processors")
        data_results = lmm.pygemma(Y, X, W, K, snps=snps['SNP'].values[sample], verbose=1, nproc=nproc)
        print(data_results.head(20))

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
            'pos'  : snps['POS'].values[sample],
            'pval' : -np.log10(data_results['p_wald']+1/len(data_results)),
            'chr' : snps['CHR'].values[sample]
            }
        )

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

        #############
        # LRT Plots #
        #############

        # pvals = np.sort(data_results['p_lrt'])

        # plt.scatter(y=-np.log10(pvals+1/len(data_results)), x=-np.log10(theoretical))
        # plt.axline((0,0), slope=1, color='red')
        # plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        # plt.ylabel(r'Observed: $-\log_{10}(p)$')
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_lrt_qq.png"))
        # plt.clf()

        # # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        # results_df = pd.DataFrame(
        #     {
        #     'pos'  : snps['POS'].values[sample],
        #     'pval' : -np.log10(data_results['p_lrt']+1/len(data_results)),
        #     'chr' : snps['CHR'].values[sample]
        #     }
        # )

        # results_df = results_df.sort_values(['chr', 'pos'])
        # results_df.reset_index(inplace=True, drop=True)
        # results_df['i'] = results_df.index

        # alpha = -np.log10(0.05/len(pvals))
        # with sns.color_palette():
        #    sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'])
        # plt.axline((0,alpha), slope=0, color='red')
        # chrom_df=results_df.groupby('chr')['i'].median()
        # plt.xlabel('chr') 
        # plt.xticks(chrom_df,chrom_df.index)
        # plt.ylabel(r'$-\log_{10}(p)$')
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_lrt_manhatten.png"))
        # plt.clf()

        # plt.scatter(y=-np.log10(data_results['p_wald']+1/len(data_results)), x=-np.log10(data_results['p_lrt']+1/len(data_results)))
        # plt.axline((0,0), slope=1, color='red')
        # plt.xlabel(r'LRT: $-\log_{10}(p)$')
        # plt.ylabel(r'Wald: $-\log_{10}(p)$')
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_lrt.png"))
        # plt.clf()
        

        ##### Fixed Effects only #####
        data_results = run_gwas(Y, W, X, snps=snps['SNP'].values[sample], verbose=1)
        median_p = np.median(data_results['p_wald'].values)

        median_chisq = stats.chi2.ppf(1-median_p, 1)

        lambda_gc = median_chisq/stats.chi2.ppf(0.5, 1)

        print(f'Lambda GC: {lambda_gc}')
        data_results['p_wald_gc'] = 1-stats.chi2.cdf(stats.chi2.ppf(1-data_results['p_wald'] , 1)/lambda_gc, df=1)
        print(data_results.head(20))

        theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
        pvals = np.sort(data_results['p_wald'])
        
        plt.scatter(y=-np.log10(pvals+1/len(data_results)), x=-np.log10(theoretical))
        plt.axline((0,0), slope=1, color='red')
        plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        plt.ylabel(r'Observed: $-\log_{10}(p)$')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_qq_fixed.png"))
        plt.clf()

        # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        results_df = pd.DataFrame(
            {
            'pos'  : snps['POS'].values[sample],
            'pval' : -np.log10(data_results['p_wald']+1/len(data_results)),
            'chr' : snps['CHR'].values[sample]
            }
        )

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
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten_fixed.png"))
        plt.clf()

        # GC
        theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
        pvals = np.sort(data_results['p_wald_gc'])
        
        plt.scatter(y=-np.log10(pvals+1/len(data_results)), x=-np.log10(theoretical))
        plt.axline((0,0), slope=1, color='red')
        plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        plt.ylabel(r'Observed: $-\log_{10}(p)$')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_qq_fixed_gc.png"))
        plt.clf()

        # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        results_df = pd.DataFrame(
            {
            'pos'  : snps['POS'].values[sample],
            'pval' : -np.log10(data_results['p_wald_gc']+1/len(data_results)),
            'chr' : snps['CHR'].values[sample]
            }
        )

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
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten_fixed_gc.png"))
        plt.clf()

        
