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
        sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'], linewidth=0)
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


def generate_test_matrices(n=1000, covars=10, seed=42):
    np.random.seed(seed)
    K = np.random.uniform(size=(n, n))
    K = np.abs(np.tril(K) + np.tril(K, -1).T)
    K = np.dot(K, K.T)
    eigenVals, U = np.linalg.eig(K)
    eigenVals = np.maximum(0, eigenVals)
    W = np.random.rand(n, covars)
    W = np.c_[W, np.ones(n)]
    x = np.random.choice([0,1,2], 
                        size=(n, 1),
                        replace=True)
    Y = np.random.rand(n, 1).reshape(-1,1)
    lam = 500
    tau = 10
    beta = np.random.rand(covars+2, 1)

    return x.astype(np.float32), Y.astype(np.float32), W.astype(np.float32), eigenVals.astype(np.float32), U.astype(np.float32), np.float32(lam), beta.astype(np.float32), np.float32(tau)

with console.status("[bold green]Running pyGEMMA Function Run Tests...") as status:
    print(np.show_config())

    # Seed tests
    np.random.seed(42)

    n = 1000
    covars = 10

    # Initializing parameters for tests
    x, Y, W, eigenVals, U, lam, beta, tau = generate_test_matrices(n=n, covars=covars)
    W = np.c_[W,x]

    precompute_mat = lmm.precompute_mat(lam, eigenVals, U.T @ W, U.T @ Y, full=True)
    print(W.shape)
    Px = pygemma_model.compute_Pc(eigenVals, U, W, lam)
    print('Y.T @ Px @ Y: ', float(Y.T @ Px @ Y), precompute_mat['yt_Pi_y'][W.shape[1]])
    print('Y.T @ Px @ Px @ Y: ', float(Y.T @ Px @ Px @ Y), precompute_mat['yt_Pi_Pi_y'][W.shape[1]])
    print('Y.T @ Px @ Px @ Px @ Y: ', float(Y.T @ Px @ Px @ Px @ Y), precompute_mat['yt_Pi_Pi_Pi_y'][W.shape[1]])
    print('Tr(Px): ', np.trace(Px), precompute_mat['tr_Pi'][W.shape[1]])
    print('Tr(Px @ Px): ', np.trace(Px @ Px), precompute_mat['tr_Pi_Pi'][W.shape[1]])
    print('logdet_Wt_H_inv_W: ', np.linalg.slogdet((U.T @ W).T @ (1.0/(lam*eigenVals + 1.0)[:, np.newaxis] * (U.T @ W)))[1], precompute_mat['logdet_Wt_H_inv_W'])

    # print(lmm.compute_at_Pi_Pi_Pi_b(lam, W.shape[1],
    #                   lam*eigenVals + 1.0,
    #                   U.T @ W,
    #                   U.T @ Y,
    #                   U.T @ Y))

    x = (U.T @ x).reshape(-1)
    Y = U.T @ Y
    W = U.T @ W

    n, c = W.shape

    #warnings.filterwarnings('error')

    #ww = np.multiply(np.c_[W, x, Y].T[:, :, np.newaxis], np.c_[W, x, Y][np.newaxis, :, :])

    for lam in [1e-3, 5.0, 400, 1e3, 1e5]:
        console.log(f'\nTest Parameters: n={n}, lam={lam}, tau={tau}')
        
        precompute_mat = lmm.precompute_mat(lam, eigenVals, np.c_[W, x], Y)

        functions_and_args = [
                                # (lmm.compute_Pc, [eigenVals, W, lam]),
                                # (pygemma_model.compute_Pc, [eigenVals, W, lam]),
                                (lmm.precompute_mat, [lam, eigenVals, np.c_[W, x], Y, False]),
                                (lmm.precompute_mat, [lam, eigenVals, np.c_[W, x], Y]),
                                (lmm.precompute_mat_test, [lam, eigenVals, np.c_[W, x], Y, False]),
                                (lmm.precompute_mat_test, [lam, eigenVals, np.c_[W, x], Y]),
                                (lmm.precompute_mat_optimized, [lam, eigenVals, np.c_[W, x], Y, False]),
                                (lmm.precompute_mat_optimized, [lam, eigenVals, np.c_[W, x], Y]),
                                (lmm.precompute_mat_flipped, [lam, eigenVals, np.c_[W, x], Y, False]),
                                (lmm.precompute_mat_flipped, [lam, eigenVals, np.c_[W, x], Y]),
                                (lmm.precompute_mat_modified, [lam, eigenVals, np.c_[W, x], Y, False]),
                                (lmm.precompute_mat_modified, [lam, eigenVals, np.c_[W, x], Y]),
                                (lmm.precompute_mat_concat, [lam, eigenVals, np.asfortranarray(np.c_[W, x, Y]), False]),
                                (lmm.precompute_mat_concat, [lam, eigenVals, np.asfortranarray(np.c_[W, x, Y])]),
                                (lmm.calc_beta_vg_ve_restricted, [eigenVals, W, x.reshape(-1,1), lam, Y]),
                                (lmm.calc_beta_vg_ve_restricted_overload, [eigenVals, W, x.reshape(-1,1), lam, Y]),
                                (lmm.likelihood_lambda, [lam, eigenVals, Y, W]),
                                (lmm.likelihood_derivative1_lambda, [lam, eigenVals, Y, W]),
                                (lmm.likelihood_derivative2_lambda, [lam, eigenVals, Y, W]),
                                (lmm.newton, [lam, eigenVals, Y,  np.c_[W, x], True]),
                                (lmm.calc_lambda, [eigenVals, Y, W]),
                                (lmm.calc_lambda_restricted, [eigenVals, Y, W]),
                                (lmm.likelihood, [lam, tau, beta, eigenVals, Y, W]),
                                (lmm.likelihood_restricted, [lam, tau, eigenVals, Y, W]),
                                (lmm.likelihood_restricted_lambda, [lam, eigenVals, Y, W]),
                                (lmm.likelihood_derivative1_restricted_lambda, [lam, eigenVals, Y, W]),
                                (lmm.wrapper_likelihood_derivative1_restricted_lambda, [lam, eigenVals, Y, W]),
                                (lmm.likelihood_derivative2_restricted_lambda, [lam, eigenVals, Y, W]),
                                (lmm.likelihood_restricted_lambda_overload, [lam, n, c, precompute_mat['yt_Pi_y'][c], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_W'],precompute_mat['logdet_Wt_H_inv_W']]),
                                (lmm.likelihood_derivative1_restricted_lambda_overload, [lam, n, c, precompute_mat['yt_Pi_y'][c], precompute_mat['yt_Pi_Pi_y'][c], precompute_mat['tr_Pi'][c]]),
                                (lmm.likelihood_derivative2_restricted_lambda_overload, [lam, n, c, precompute_mat['yt_Pi_y'][c], precompute_mat['yt_Pi_Pi_y'][c], precompute_mat['yt_Pi_Pi_Pi_y'][c], precompute_mat['tr_Pi'][c], precompute_mat['tr_Pi_Pi'][c]]),
                                (lmm.trace_Pi, [lam, W.shape[1], lam*eigenVals+1.0, W]),
                                (lmm.trace_Pi_Pi, [lam, W.shape[1], lam*eigenVals+1.0, W]),
                                (lmm.compute_at_Pi_b, [lam, W.shape[1], lam*eigenVals+1.0, W, Y, Y]),
                                (lmm.compute_at_Pi_Pi_b, [lam, W.shape[1], lam*eigenVals+1.0, W, Y, Y]),
                                (lmm.compute_at_Pi_Pi_Pi_b, [lam, W.shape[1], lam*eigenVals+1.0, W, Y, Y]),
                            ]
        
        run_test_list(functions_and_args)


# Function to simulate GWAS dataset
def simulate_gwas_dataset(n=1000, p=10000, c=100, seed=42):
    lam = 20
    tau_inv = 5
    np.random.seed(seed)
    
    # Generate fake SNP names and posisions from chr 1
    # Format EX: 1:182686:A:G
    snp_names = [f'1:{i}:A:G' for i in range(1, p+1)]
    
    # Generate fake SNP data
    snp_data = np.random.randint(0, 3, size=(n, p))
    snp_data = snp_data.astype(np.float32)
    snp_data_norm = (snp_data - np.mean(snp_data, axis=0))/np.std(snp_data, axis=0)

    # Relatedness matrix
    K = snp_data_norm @ snp_data_norm.T / p

    # Choose significant SNP beta values
    # Make peak at c/2
    beta = np.random.uniform(low=0.5, high=1.5, size=(c, 1)) + np.random.normal(loc=0.0, scale=0.1, size=(c, 1)) + (np.abs((c/2) - np.arange(0, c).reshape(-1,1)) / c).reshape(-1,1)

    # Create phenotype
    Y = snp_data[:,0:c] @ beta + np.random.normal(loc=0.0, scale=tau_inv, size=(n, 1)) 

    # Add multivariate normal to phenotype
    Y = Y + np.random.multivariate_normal(mean=np.zeros(n), cov=lam*tau_inv*K, size=1).T

    Y = pd.DataFrame(Y, columns=['Phenotype'])

    X = pd.DataFrame(snp_data, columns=snp_names)

    return X, Y, K, beta


# Simulate GWAS data
print('Simulating GWAS data...')
#snp_data, Y, K, beta = simulate_gwas_dataset(n=1000, p=1000, c=100, seed=42)

DATADIR = os.path.join("..","data")
dataset_list = [
        {
            'name'    : 'Homework3',
            'snps'    : os.path.join(DATADIR, "test_data.csv"),
            'covars'  : None,
            'pheno'   : os.path.join(DATADIR, "GD449.example.pheno.tsv"),
            'kinship' : None
        },
        # {
        #     'name'    : 'SimData',
        #     'snps'    : snp_data,
        #     'covars'  : None,
        #     'pheno'   : Y,
        #     'kinship' : K
        # }
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



    n,p = X.shape

    print('Getting kinship matrix...')
    if dataset['kinship'] is None:
        K = calculate_genetic_relatedness_matrix(X)
        #K = X @ X.T / p
    else:
        if isinstance(dataset['kinship'], str):
            K = pd.read_csv(dataset['kinship'], header=None).values
        else:
            K = dataset['kinship']

    #K = ((K - np.mean(K, axis=0)) / np.std(K, axis=0)).astype(np.float32)

    print('Running PCA...')
    pca = PCA(n_components=5)

    pcs = pca.fit_transform(X)

    sample = range(0,X.shape[1]) 
    #sample = np.random.choice(range(0,X.shape[1]), size=1000, replace=False)
    X = X[:,sample]
    pheno_name = pheno.columns[0]
    Y = pheno[pheno_name].values.reshape(-1,1).astype(np.float32)
    Y = qnorm.quantile_normalize(Y, axis=1)
    Y = (Y-np.mean(Y))/np.std(Y)
    Y = Y.reshape(-1,1)
    #print(Y.mean(), Y.std())
    # Likelihood tests
    x = X[:,0].reshape(-1,1)
    #x = (x - np.mean(x))/np.std(x)
    n = Y.shape[0]
    #print(pcs.mean(axis=0), pcs.std(axis=0))

    W = np.c_[np.ones(shape=(n, 1)), pcs].astype(np.float32)
    #W = np.ones(shape=(n, 1)).astype(np.float32)
    
    #W = np.ones(shape=(n, 1)).astype(np.float32)
    step_size = 0.05
    lam_vals = np.array([np.power(10.0, i) for i in np.arange(-5.0,5.0+step_size,step_size)], dtype=np.float32)
    eigenVals, U = np.linalg.eig(K)
    eigenVals = np.maximum(1e-10, eigenVals)

    eigenVals = eigenVals.astype(np.float32)
    U = U.astype(np.float32)

    # # Sort eigenvalues in descending order
    # sorted_indices = np.argsort(eigenVals)
    # eigenVals = eigenVals[sorted_indices]

    # # Reorder eigenvectors according to sorted eigenvalues
    # U = U[:, sorted_indices]

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

    yt_Px_y = [float(np.trace(pygemma_model.compute_Pc(eigenVals, U, np.c_[W,x.reshape(-1,1)], l) @ pygemma_model.compute_Pc(eigenVals, U, np.c_[W,x.reshape(-1,1)], l))) for l in lam_vals]
    Y_star = U.T @ Y
    W_star = U.T @ W
    x_star = U.T @ x
    W_x_star = np.c_[W_star, x_star]
    x_star = x_star.reshape(-1,1)

    yt_Px_y_test = [float(lmm.trace_Pi_Pi(l, W_x_star.shape[1], l*eigenVals + 1.0, W_x_star)) for l in lam_vals]

    print(pd.DataFrame({'lam': lam_vals, 'yt_Px_y': yt_Px_y, 'yt_Px_y_test': yt_Px_y_test}))

    sns.scatterplot(x=lam_vals, 
                    y=np.array(yt_Px_y_test) - np.array(yt_Px_y))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "yt_Px_y.png"))
    plt.clf()

    c = W_star.shape[1]

    # lik = [lmm.likelihood_restricted_lambda(l, eigenVals, Y_star, W_x_star) for l in lam_vals]
    # lik_der1 = [lmm.likelihood_derivative1_restricted_lambda(l, eigenVals, Y_star, W_x_star) for l in lam_vals]
    # lik_der2 = [lmm.likelihood_derivative2_restricted_lambda(l, eigenVals, Y_star, W_x_star) for l in lam_vals]
    print('Calculating Lambda...')
    lam_temp = lmm.calc_lambda_restricted(eigenVals, Y_star, W_x_star)

    lik = []
    lik_der1 = []
    lik_der2 = []

    for l in track(lam_vals, description='Calculating Likelihoods...'):
        # Px = pygemma_model.compute_Pc(eigenVals, U, np.c_[W,x.reshape(-1,1)], l)
        # lik.append(0.5*(n-c-1)*np.log((n-c-1)/(2*np.pi)) - 0.5*(n-c-1) + 0.5*np.linalg.slogdet(W_x_star.T @ W_x_star)[1] - 0.5*np.sum(np.log(l*eigenVals + 1.0)) - 0.5*np.linalg.slogdet(W_x_star.T @ (1.0/(l*eigenVals + 1.0)[:,np.newaxis] * W_x_star))[1] - 0.5*(n-c-1)*np.log(Y.T @ Px @ Y))
        precompute_mat = lmm.precompute_mat(l, eigenVals, W_x_star, Y_star)
        #print(l, precompute_mat['yt_Pi_y'][W_x_star.shape[1]], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_W'],precompute_mat['logdet_Wt_H_inv_W'])
        lik.append(lmm.likelihood_restricted_lambda_overload(l, n, W_x_star.shape[1], precompute_mat['yt_Pi_y'][W_x_star.shape[1]], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_W'],precompute_mat['logdet_Wt_H_inv_W']))
        lik_der1.append(lmm.likelihood_derivative1_restricted_lambda_overload(l, n, W_x_star.shape[1], precompute_mat['yt_Pi_y'][W_x_star.shape[1]], precompute_mat['yt_Pi_Pi_y'][W_x_star.shape[1]], precompute_mat['tr_Pi'][W_x_star.shape[1]]))
        #lik_der1.append(lmm.likelihood_derivative1_restricted_lambda(l, eigenVals, Y_star, W_x_star))
        lik_der2.append(lmm.likelihood_derivative2_restricted_lambda_overload(l, n, W_x_star.shape[1], precompute_mat['yt_Pi_y'][W_x_star.shape[1]], precompute_mat['yt_Pi_Pi_y'][W_x_star.shape[1]], precompute_mat['yt_Pi_Pi_Pi_y'][W_x_star.shape[1]], precompute_mat['tr_Pi'][W_x_star.shape[1]], precompute_mat['tr_Pi_Pi'][W_x_star.shape[1]]))

    print('Best Lambda: ', lam_temp)
    print('Best Likelihood: ', lmm.likelihood_restricted_lambda(lam_temp, eigenVals, Y_star, W_x_star))
    precompute_mat = lmm.precompute_mat(lam_temp, eigenVals, W_x_star, Y_star)
    print('Best Likelihood Precompute: ', lmm.likelihood_restricted_lambda_overload(lam_temp, n, W_x_star.shape[1], precompute_mat['yt_Pi_y'][W_x_star.shape[1]], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_W'],precompute_mat['logdet_Wt_H_inv_W']))
    print('Results: ', lmm.calc_beta_vg_ve_restricted(eigenVals, W_star, x_star, lam_temp, Y_star))
    print('Likelihood Min and Max: ', np.min(lik), np.max(lik))
    print('Likelihood Derivative 1 Min and Max: ', np.min(lik_der1), np.max(lik_der1))
    print('Likelihood Derivative 1 Min and Max above 1e2: ', np.min(np.array(lik_der1)[lam_vals > 1e2]), np.max(np.array(lik_der1)[lam_vals > 1e2]))
    print('Likelihood Derivative 2 Min and Max: ', np.min(lik_der2), np.max(lik_der2))
    plt.scatter(x=lam_vals, y=lik)
    plt.axvline(x=lam_temp, color='blue')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "likelihood.png"))
    plt.clf()
    plt.scatter(x=lam_vals, y=lik_der1, c='red')
    plt.axvline(x=lam_temp, color='blue')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "likelihood_derivative.png"))
    plt.clf()
    plt.scatter(x=lam_vals, y=lik_der2, c='red')
    plt.axvline(x=lam_temp, color='blue')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "likelihood_derivative2.png"))
    plt.clf()

    # for l in [0.000010,
    #             0.000100,
    #             0.001000,
    #             0.010000,
    #             0.100000,
    #             1.000000,
    #             10.000000,
    #             100.000000,
    #             1000.000000,
    #             10000.000000]:
    #     precompute_mat = lmm.precompute_mat(l, eigenVals, W_x_star, Y_star)
    #     Px = pygemma_model.compute_Pc(eigenVals, U, np.c_[W,x.reshape(-1,1)], l)
    #     print(l, float(Y.T @ Px @ Y))
    #     print(l, precompute_mat['yt_Pi_y'][W_x_star.shape[1]], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_H_inv_W']-precompute_mat['logdet_Wt_W'], lmm.likelihood_restricted_lambda_overload(l, n, W_x_star.shape[1], precompute_mat['yt_Pi_y'][W_x_star.shape[1]], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_W'], precompute_mat['logdet_Wt_H_inv_W']),
    #             np.log(np.linalg.det(W_x_star.T @ (1.0/(l*eigenVals + 1.0)[:,np.newaxis] * W_x_star)) / np.linalg.det(W_x_star.T @ W_x_star)))
    #     #print(l, precompute_mat['wjt_Pi_wk'], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_W'],precompute_mat['logdet_Wt_H_inv_W'], precompute_mat['logdet_Wt_H_inv_W']-precompute_mat['logdet_Wt_W'])
    #     #print(l, precompute_mat['yt_Pi_y'][W_x_star.shape[1]], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_W'],precompute_mat['logdet_Wt_H_inv_W'], precompute_mat['logdet_Wt_H_inv_W']-precompute_mat['logdet_Wt_W'])
    #     #print(l, precompute_mat['yt_Pi_y'][-1], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_W'],precompute_mat['logdet_Wt_H_inv_W'], precompute_mat['logdet_Wt_H_inv_W']-precompute_mat['logdet_Wt_W'])
    #     #print(l, precompute_mat['yt_Pi_y'], precompute_mat['logdet_H'], precompute_mat['logdet_Wt_W'],precompute_mat['logdet_Wt_H_inv_W'], precompute_mat['logdet_Wt_H_inv_W']-precompute_mat['logdet_Wt_W'])

    for pheno_name in pheno.columns:
        p_vals_dict = {'SNP' : snps['SNP'].values[sample]}

        Y = pheno[pheno_name].values.reshape(-1,1).astype(np.float32)
        Y = qnorm.quantile_normalize(Y, axis=1)
        Y = (Y-np.mean(Y))/np.std(Y)
        Y = Y.reshape(-1,1)
        
        # Run GEMMA
        print('Running GEMMA')
        data_results, total_time = gemma_utils.run_gemma('gemma_run',
                                                            pd.DataFrame(X, columns=snps['SNP'].values[sample]),
                                                            #pd.DataFrame(X[:,0:1], columns=snps['SNP'].values[0:1]),
                                                            Y,
                                                            W,
                                                            K)

        p_vals_dict['gemma'] = data_results['p_wald'].values
        
        print('GEMMA Run Time:', total_time, 's')
        print(data_results.head(10))
        theoretical = np.linspace((1/len(data_results)) + (1e-300),1.0,len(data_results))
        pvals = np.sort(data_results['p_wald'])
        
        plt.scatter(y=-np.log10(pvals+1e-300), x=-np.log10(theoretical))
        plt.axline((0,0), slope=1, color='red')
        plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        plt.ylabel(r'Observed: $-\log_{10}(p)$')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_gemma_wald_qq.png"))
        plt.clf()

        # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        results_df = pd.DataFrame(
            {
            'pos'  : snps['POS'].values[sample],
            'pval' : -np.log10(data_results['p_wald']+1e-300),
            'chr' : snps['CHR'].values[sample]
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
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_gemma_wald_manhatten.png"))
        plt.clf()

        # # Run EMMA
        # print('Running EMMA')
        # data_results, total_time = gemma_utils.run_emma('emma_run',
        #                                                     pd.DataFrame(X, columns=snps['SNP'].values[sample]),
        #                                                     #pd.DataFrame(X[:,0:1], columns=snps['SNP'].values[0:1]),
        #                                                     Y,
        #                                                     W,
        #                                                     K)

        # p_vals_dict['emma'] = data_results['p_wald'].values
        
        # print('EMMA Run Time:', total_time, 's')
        # print(data_results.head(10))
        # theoretical = np.linspace((1/len(data_results)) + (1e-300),1.0,len(data_results))
        # pvals = np.sort(data_results['p_wald'])
        
        # plt.scatter(y=-np.log10(pvals+1e-300), x=-np.log10(theoretical))
        # plt.axline((0,0), slope=1, color='red')
        # plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        # plt.ylabel(r'Observed: $-\log_{10}(p)$')
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_emma_wald_qq.png"))
        # plt.clf()

        # # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        # results_df = pd.DataFrame(
        #     {
        #     'pos'  : snps['POS'].values[sample],
        #     'pval' : -np.log10(data_results['p_wald']+1e-300),
        #     'chr' : snps['CHR'].values[sample]
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
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_emma_wald_manhatten.png"))
        # plt.clf()
        
        #data_results = lmm.pygemma(Y - H @ Y, X - H @ X, np.ones(shape=(n, 1)), K, snps=snps['SNP'].values[sample], verbose=1)
        nproc = 1 if os.environ.get('SLURM_CPUS_PER_TASK') is None else int(os.environ.get('SLURM_CPUS_PER_TASK'))
        print(f"Using {nproc} processors")
        #data_results = jax_pygemma.pygemma_jax(Y, X, W, K, snps=snps['SNP'].values[sample], verbose=1, nproc=nproc)
        start = time.time()
        data_results = lmm.pygemma(Y, X, W, K, snps=snps['SNP'].values[sample], verbose=1, nproc=nproc)
        print('PyGemma Run Time:', time.time() - start, 's')
        print(data_results.head(10))

        p_vals_dict['pygemma'] = data_results['p_wald'].values

        theoretical = np.linspace((1/len(data_results)) + (1e-300),1.0,len(data_results))
        pvals = np.sort(data_results['p_wald'])
        
        plt.scatter(y=-np.log10(pvals+1e-300), x=-np.log10(theoretical))
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
            'pval' : -np.log10(data_results['p_wald']+1e-300),
            'chr' : snps['CHR'].values[sample]
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
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten.png"))
        plt.clf()

        #############
        # LRT Plots #
        #############

        # pvals = np.sort(data_results['p_lrt'])

        # plt.scatter(y=-np.log10(pvals+1e-300), x=-np.log10(theoretical))
        # plt.axline((0,0), slope=1, color='red')
        # plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
        # plt.ylabel(r'Observed: $-\log_{10}(p)$')
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_lrt_qq.png"))
        # plt.clf()

        # # Manhatten plot adapted from https://stackoverflow.com/a/66062857
        # results_df = pd.DataFrame(
        #     {
        #     'pos'  : snps['POS'].values[sample],
        #     'pval' : -np.log10(data_results['p_lrt']+1e-300),
        #     'chr' : snps['CHR'].values[sample]
        #     }
        # )

        # results_df = results_df.sort_values(['chr', 'pos'])
        # results_df.reset_index(inplace=True, drop=True)
        # results_df['i'] = results_df.index

        # alpha = -np.log10(0.05/len(pvals))
        # with sns.color_palette():
        #    sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'], linewidth=0)
        # plt.axline((0,alpha), slope=0, color='red')
        # chrom_df=results_df.groupby('chr')['i'].median()
        # plt.xlabel('chr') 
        # plt.xticks(chrom_df,chrom_df.index)
        # plt.ylabel(r'$-\log_{10}(p)$')
        # plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_lrt_manhatten.png"))
        # plt.clf()

        # plt.scatter(y=-np.log10(data_results['p_wald']+1e-300), x=-np.log10(data_results['p_lrt']+1e-300))
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
        print(data_results.head(10))

        p_vals_dict['linear_model'] = data_results['p_wald'].values
        p_vals_dict['linear_model_gc'] = data_results['p_wald_gc'].values

        theoretical = np.linspace((1/len(data_results)) + (1e-300),1.0,len(data_results))
        pvals = np.sort(data_results['p_wald'])
        
        plt.scatter(y=-np.log10(pvals+1e-300), x=-np.log10(theoretical))
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
            'pval' : -np.log10(data_results['p_wald']+1e-300),
            'chr' : snps['CHR'].values[sample]
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
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten_fixed.png"))
        plt.clf()

        # GC
        theoretical = np.linspace((1/len(data_results)) + (1e-300),1.0,len(data_results))
        pvals = np.sort(data_results['p_wald_gc'])
        
        plt.scatter(y=-np.log10(pvals+1e-300), x=-np.log10(theoretical))
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
            'pval' : -np.log10(data_results['p_wald_gc']+1e-300),
            'chr' : snps['CHR'].values[sample]
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
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_wald_manhatten_fixed_gc.png"))
        plt.clf()

        # Convert p_vals_dict to dataframe and save
        p_vals_df = pd.DataFrame(p_vals_dict)

        p_vals_df.to_csv(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_pvals.csv"), index=False)

        # Grid of plots for methods
        # For gemma, emma, and pygemma plot pairwise the p-values off diagonal
        # Plot the QQ plot on the diagonal
        
        #method_list = ['gemma', 'emma', 'pygemma']
        method_list = ['gemma', 'pygemma']

        figure, axes = plt.subplots(nrows=len(method_list), ncols=len(method_list), figsize=(20,20))

        # for idx_1, method1 in enumerate(['gemma', 'emma', 'pygemma']):
        #     for idx_2, method2 in enumerate(['gemma', 'emma', 'pygemma']):
        for idx_1, method1 in enumerate(method_list):
            for idx_2, method2 in enumerate(method_list):
                if idx_1 == idx_2:
                    # Plot QQ plot for method
                    pvals = np.sort(p_vals_df[method1])
                    theoretical = np.linspace(1/len(pvals),1.0,len(pvals))

                    # Seaborn qq plot
                    sns.scatterplot(x=-np.log10(pvals+1/len(p_vals_df)), y=-np.log10(theoretical), ax=axes[idx_1, idx_2])
                    axes[idx_1, idx_2].axline((0,0), slope=1, color='red')
                    axes[idx_1, idx_2].set_xlabel(r'Theoretical: $-\log_{10}(p)$')
                    axes[idx_1, idx_2].set_ylabel(r'Observed: $-\log_{10}(p)$')
                    
                    # Set title
                    axes[idx_1, idx_2].set_title(f"{method1} QQ Plot")
                    
                else:
                    # Plot p-value scatter plot with seaborn
                    #sns.scatterplot(x=-np.log10(np.maximum(p_vals_df[method1], 1e-25)), y=-np.log10(np.maximum(p_vals_df[method2], 1e-25)), ax=axes[idx_1, idx_2])
                    sns.scatterplot(x=-np.log10(p_vals_df[method1]), y=-np.log10(p_vals_df[method2]), ax=axes[idx_1, idx_2])
                    axes[idx_1, idx_2].set_xlabel(f"{method1}")
                    axes[idx_1, idx_2].set_ylabel(f"{method2}")

                    # Plot diagonal line
                    axes[idx_1, idx_2].axline((0,0), slope=1, color='red')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"{dataset_name}_{pheno_name}_pval_comparison.png"))
        figure.clf()
        plt.close(figure)
        plt.clf()

        # Reset plotting
        plt.rcParams.update(plt.rcParamsDefault)
        sns.set_theme()


        
