import time
import os

import numpy as np
import pandas as pd
import qnorm

import argparse

from rich.console import Console

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from pysnptools.snpreader import Bed

import warnings

sns.set_theme()

from pygemma import lmm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--snps", dest="snps", help="Path to snps", type=str, default="snps.csv")
    parser.add_argument("-p", "--phenotype", dest="phenotype", help="Path to phenotype", type=str, default="phenotypes.csv")
    parser.add_argument("-c", "--covars", dest="covars", help="Path to covars", type=str, default=None)
    parser.add_argument("-pcs", "--pcs", dest="pcs", help="Number of PCs", type=int, default=2)
    parser.add_argument("-pcf", "--pcfile", dest="pcfile", help="File containing PCs", type=str, default=None)
    parser.add_argument('-k', '--kinship', dest='kinship', help='Path to kinship matrix', type=str, default=None)
    parser.add_argument("-n", '--nproc', dest='nproc', help='Number of processes', type=int, default=1)
    parser.add_argument("-o", "--output", dest="output", help="Path to output directory", type=str, default="output_dir")
    args = parser.parse_args()

    # Set seed
    np.random.seed(42)

    # Make output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read in SNPs
    print('Reading in SNPs...')
    # Read PLINK files (.bed, .fam, .bim)
    chromosome = 20
    phenotype_idx = 0
    plink_data = Bed("/net/fantasia/home/borang/Robert/UKB_AFR/Geno/AFR/chr_{}.bed".format(chromosome), count_A1=False)

    # ID info which can be matched up eid
    geno_id = pd.Series(plink_data.iid[:, 1])

    # Sample individuals
    individuals = range(len(plink_data.iid))
    #individuals = np.random.choice(range(len(plink_data.iid)), 100, replace=False)
    geno_id = geno_id[individuals]

    sample = range(len(plink_data.sid))
    #sample = np.random.choice(range(len(plink_data.sid)), 1000, replace=False)

    # Genotype
    genotypes = plink_data.read(dtype=np.float32)

    X = genotypes.val[individuals, :].astype(np.float32)
    X = X[:, sample]

    # rs3131965 to chr:3131965:X:Y
    snps = [f'{chromosome}:{rs[2:]}:X:Y' for rs in plink_data.sid[sample]]
    p = X.shape[1]

    # Impute missing SNPs with sklearn
    print('Imputing missing SNPs...')
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)
    #X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Read phenotypes
    print('Reading in phenotypes...')
    phenotype_df = pd.read_csv(args.phenotype)
    Y = phenotype_df[phenotype_df.columns[1:][phenotype_idx]].values.reshape(-1,1)[individuals,:].astype(np.float32)
    
    # Impute missing phenotypes with sklearn
    print('Imputing missing phenotypes...')
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    Y = imp.fit_transform(Y).reshape(-1,1)


    # Y = phenotype_df['Exp_Value'].values.reshape(-1,1)
    Y = qnorm.quantile_normalize(Y, axis=1)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    Y = Y.reshape(-1,1).astype(np.float32)
    del phenotype_df

    W = np.ones(shape=(X.shape[0], 1), dtype=np.float32)

    if args.kinship:
        print('Reading in kinship matrix...')
        kinship_df = pd.read_csv(args.kinship, sep='\t', header=None)
        K = kinship_df.values
        del kinship_df
    else:
        print('Computing kinship matrix...')
        K = (X - X.mean(axis=0)) / X.std(axis=0)
        K = K @ K.T / p

    # Read covariates
    if args.covars:
        print('Reading in covariates...')
        covars_df = pd.read_csv(args.covars, sep=' ')

        # Female is 1, Male is 0
        gender_indicator = (covars_df['Inferred.Gender'].values == 'F').reshape(-1,1).astype(np.float32)[individuals,:]
        W = np.c_[W, gender_indicator]
        
        if int(args.pcs) > 0:
            if 'PC1' in covars_df.columns:
                print('Reading in PCs...')
                pcs = covars_df[[f'PC{i}' for i in range(1, int(args.pcs)+1)]].values[individuals,:].astype(np.float32)
                pcs = (pcs - pcs.mean(axis=0)) / pcs.std(axis=0)
                W = np.c_[W, pcs]
                del pcs
            else:
                print('Running PCA...')
                pca = PCA(n_components=int(args.pcs))

                pcs = pca.fit_transform(X)

                W = np.c_[W, pcs]

        del covars_df

    #############
    # DEBUGGING #
    #############

    # import scipy

    # eigenVals, U = scipy.linalg.eigh(K)

    # eigenVals = eigenVals.astype(np.float32)
    # U = U.astype(np.float32)

    # eigenVals = np.maximum(0, eigenVals)

    # assert (eigenVals >= 0).all()

    # X = U.T @ X[:,0:1]
    # Y = U.T @ Y
    # W = U.T @ W

    # X = X.reshape(-1,1)

    # # print dtypes
    # print(f'X: {X.dtype}')
    # print(f'Y: {Y.dtype}')
    # print(f'W: {W.dtype}')
    # print(f'K: {K.dtype}')
    # print(f'eigenVals: {eigenVals.dtype}')


    # lambda_restricted = lmm.calc_lambda_restricted(eigenVals, Y, np.c_[W, X])

    # print(f'lambda_restricted: {lambda_restricted}')

    # #beta, beta_vec, se_beta, tau = lmm.calc_beta_vg_ve_restricted(eigenVals, W, X.reshape(-1,1), lambda_restricted, Y)
    # lam = lambda_restricted
    # MIN_VAL = 1e-20
    # x = X.reshape(-1,1)
    # W_x = np.c_[W,x]

    # n = W.shape[0]
    # c = W.shape[1]
    
    # mod_eig = lam*eigenVals + 1.0

    # W_x_t_H_inv = ((1.0/mod_eig)[:,np.newaxis] * W_x).T
    # #print(f'W_x_t_H_inv: {W_x_t_H_inv}')
    
    # beta_vec = np.linalg.inv(W_x_t_H_inv @ W_x) @ (W_x_t_H_inv @ Y)
    
    # #cdef np.float32_t beta = beta_vec[c,0] #compute_at_Pi_b(lam, c, mod_eig, W, x, Y) / max(compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL) # Double check this property holds, but I think it does bc positive symmetric
    # beta = lmm.compute_at_Pi_b(lam, c, mod_eig, W, x, Y) / max(lmm.compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL) # Double check this property holds, but I think it does bc positive symmetric

    # ytPxy = max(lmm.compute_at_Pi_b(lam, c+1, mod_eig, W_x, Y, Y), MIN_VAL)

    # se_beta = np.sqrt(ytPxy) / (np.sqrt(max(lmm.compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL)) * np.sqrt((n - c - 1)))

    # tau = (n-c-1)/ytPxy
    # print(f'beta: {beta}, beta_vec: {beta_vec}, se_beta: {se_beta}, tau: {tau}')
    # exit(0)
    #################
    # END DEBUGGING #
    #################

    print('Launching pyGEMMA...')
    data_results = lmm.pygemma(Y, X, W, K, snps=snps, verbose=1, nproc=args.nproc)

    print('Writing results to csv...')
    data_results.to_csv(os.path.join(args.output, f'pygemma_results_chr{chromosome}_pheno{str(phenotype_idx)}.csv'), index=False)

    print('Creating Q-Q plot...')
    theoretical = np.linspace(1/len(data_results),1.0,len(data_results))
    pvals = np.sort(data_results['p_wald'])

    plt.scatter(y=-np.log10(np.maximum(pvals, 1.0/len(data_results))), x=-np.log10(theoretical))
    plt.axline((0,0), slope=1, color='red')
    plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
    plt.ylabel(r'Observed: $-\log_{10}(p)$')
    plt.title(f'QQ Plot - {os.path.splitext(os.path.basename(args.phenotype))[0][:-5]}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f"chr{chromosome}_pheno{str(phenotype_idx)}_wald_qq.png"))
    plt.clf()

    print('Creating Manhattan plot...')
    # Parse SNP column where SNP is in the format: chr:pos:ref:alt
    snps_values = data_results['SNPs'].str.split(':', expand=True)
    snps_values = snps_values[snps_values.columns[0:3]]
    snps_values.columns = ['CHR', 'POS', 'REF', 'ALT']


    # Manhattan plot adapted from https://stackoverflow.com/a/66062857
    results_df = pd.DataFrame(
        {
        'pos'  : snps_values['POS'].values,
        'pval' : -np.log10(np.maximum(data_results['p_wald'], 1.0/len(data_results))),
        'chr' : snps_values['CHR'].values
        }
    )

    results_df = results_df.sort_values(['chr', 'pos'])
    results_df.reset_index(inplace=True, drop=True)
    results_df['i'] = results_df.index

    alpha = -np.log10(0.05/len(pvals))
    print(f'alpha: {alpha}')
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
    plt.title(f'Manhattan Plot - {os.path.splitext(os.path.basename(args.phenotype))[0][:-5]}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f"chr{chromosome}_pheno{str(phenotype_idx)}_wald_Manhattan.png"))
    plt.clf()
    