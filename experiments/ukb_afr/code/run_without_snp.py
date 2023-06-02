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

from statsmodels.regression.linear_model import OLS

import warnings

sns.set_theme()

from pygemma import lmm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phenotype", dest="phenotype", help="Path to phenotype", type=str, default="phenotypes.csv")
    parser.add_argument("-c", "--covars", dest="covars", help="Path to covars", type=str, default=None)
    parser.add_argument("-pcs", "--pcs", dest="pcs", help="Number of PCs", type=int, default=2)
    parser.add_argument("-pcf", "--pcfile", dest="pcfile", help="File containing PCs", type=str, default=None)
    parser.add_argument("-o", "--output", dest="output", help="Path to output directory", type=str, default="output_dir")
    args = parser.parse_args()

    # Set seed
    np.random.seed(42)

    phenotype_idx = 1

    # Make output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read phenotypes
    print('Reading in phenotypes...')
    phenotype_df = pd.read_csv(args.phenotype)
    Y = phenotype_df[phenotype_df.columns[1:][phenotype_idx]].values.reshape(-1,1).astype(np.float32)
    
    # Impute missing phenotypes with sklearn
    print('Imputing missing phenotypes...')
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    Y = imp.fit_transform(Y).reshape(-1,1)
    Y = np.log(Y + 0.01)

    # Y = phenotype_df['Exp_Value'].values.reshape(-1,1)
    # Y = qnorm.quantile_normalize(Y, axis=1)
    # Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    # Y = Y.reshape(-1,1).astype(np.float32)
    del phenotype_df

    W = np.ones(shape=(Y.shape[0], 1), dtype=np.float32)

    # Read covariates
    if args.covars:
        print('Reading in covariates...')
        covars_df = pd.read_csv(args.covars, sep=' ')

        # Male is 1, Female is 0
        gender_indicator = (covars_df['Inferred.Gender'].values == 'M').reshape(-1,1).astype(np.float32)
        W = np.c_[W, gender_indicator]
        
        if int(args.pcs) > 0:
            if 'PC1' in covars_df.columns:
                print('Reading in PCs...')
                pcs = covars_df[[f'PC{i}' for i in range(1, int(args.pcs)+1)]].values.astype(np.float32)
                pcs = (pcs - pcs.mean(axis=0)) / pcs.std(axis=0)
                W = np.c_[W, pcs]
                del pcs
            else:
                print('Error: PCs not found in covars file')

        del covars_df

    # Fit linear fixed effects model with statmodels package
    print('Fitting linear model...')
    start = time.time()
    model = OLS(Y, W)
    results = model.fit()
    end = time.time()
    print(f'Finished fitting linear model in {end-start} seconds')

    # Perform regression diagnostics
    print('Performing regression diagnostics...')
    print(results.summary())

    # Plot histogram of residuals with seaborn
    print('Plotting histogram of residuals...')
    sns.histplot(results.resid)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'residuals.png'))
    plt.clf()

    # Plot residuals vs fitted values with seaborn
    print('Plotting residuals vs fitted values...')
    sns.scatterplot(x=results.fittedvalues.reshape(-1), y=results.resid.reshape(-1))
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'residuals_vs_fitted.png'))
    plt.clf()

    