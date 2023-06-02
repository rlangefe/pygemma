import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

filepath = '/net/mulan/home/rlangefe/gemma_work/modified_gemma/log_out.csv'

OUTDIR = '/net/mulan/home/rlangefe/gemma_work/modified_gemma/plots'

# make output dir if doesn't exist
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

df = pd.read_csv(filepath, header=None)

df.columns = ['lambda', 'likelihood', 'likelihood_d1', 'likelihood_d2']

# lambda vs likelihood
sns.scatterplot(x='lambda', y='likelihood', data=df)
plt.savefig(os.path.join(OUTDIR, 'lambda_vs_likelihood.png'))
plt.clf()

# lambda vs likelihood_d1
sns.scatterplot(x='lambda', y='likelihood_d1', data=df)
plt.savefig(os.path.join(OUTDIR, 'lambda_vs_likelihood_d1.png'))
plt.clf()

# lambda vs likelihood_d2
sns.scatterplot(x='lambda', y='likelihood_d2', data=df)
plt.savefig(os.path.join(OUTDIR, 'lambda_vs_likelihood_d2.png'))
plt.clf()

