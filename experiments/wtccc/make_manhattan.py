import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

import os

sns.set_theme()

print('Loading data...')
df = pd.read_csv('/net/mulan/home/rlangefe/gemma_work/pygemma/experiments/wtccc/6_pcs_output/RA_RA_pygemma_results.csv',
                    engine='c')

print('Loading SNP info...')
snp_info = pd.read_csv('/net/mulan/data/WTCCC/processed/WTCCC_plink_chrannot.cat.txt.gz', sep='\t')

print('Merging data...')
df = df.merge(snp_info, left_on='SNPs', right_on='SNP')

n = 4798

print('Calculating p-values...')
df['p_wald'] = stats.f.sf(np.power(df['beta'].values.astype(np.float64)/df['se_beta'].values.astype(np.float64), 2.0),
                            dfn=1,
                            dfd=n-7-1)

print('Making Manhattan plot...')
results_df = pd.DataFrame(
                        {
                        'pos'  : df['BP'].values,
                        'pval' : -np.log10(df['p_wald'].values + 1e-300),
                        'chr' : df['CHR'].values
                        }
                    )

results_df = results_df.sort_values(['chr', 'pos'])
results_df.reset_index(inplace=True, drop=True)
results_df['i'] = results_df.index

with sns.color_palette():
    sns.scatterplot(x=results_df['i'], y=results_df['pval'], hue=results_df['chr'], linewidth=0)

chrom_df=results_df.groupby('chr')['i'].median()
plt.xlabel('chr') 
plt.xticks(chrom_df,chrom_df.index)
plt.ylabel(r'$-log_{10}(p)$')

# Remove legend
plt.legend([],[], frameon=False)

# Rotate xticks 90 degrees and make them smaller
plt.xticks(rotation=90, fontsize=8)


plt.tight_layout()
plt.savefig('mod_ra_manhattan.png')
plt.clf()

print('Making QQ plot...')
sns.set_theme()
theoretical = np.linspace(1/len(df),1.0,len(df))
pvals = np.sort(df['p_wald'])

plt.scatter(y=-np.log10(pvals+1e-300), x=-np.log10(theoretical))
plt.axline((0,0), slope=1, color='red')
plt.xlabel(r'Theoretical: $-\log_{10}(p)$')
plt.ylabel(r'Observed: $-\log_{10}(p)$')
plt.tight_layout()
plt.savefig('mod_ra_qq.png')
plt.clf()

