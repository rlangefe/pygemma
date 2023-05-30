import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

if __name__ == '__main__':
    PLOTPATH = '/net/mulan/home/rlangefe/gemma_work/pygemma/tests/output'
    PYGEMMAPLOTPATH = '/net/mulan/home/rlangefe/gemma_work/pygemma/tests/output'

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    for i, pheno in enumerate(range(1,4)):
        
        pygemma_plot = plt.imread(os.path.join(PYGEMMAPLOTPATH, f'Homework3_Pheno{pheno}_wald_manhatten.png'))
        #pygemma_plot = plt.imread(os.path.join(PLOTPATH, f'Homework3_Pheno{pheno}_wald_manhatten.png'))
        ols_plot = plt.imread(os.path.join(PLOTPATH, f'Homework3_Pheno{pheno}_wald_manhatten_fixed.png'))
        gemma_plot = plt.imread(os.path.join(PLOTPATH, f'Homework3_Pheno{pheno}_gemma_wald_manhatten.png'))

        ax[i, 0].imshow(pygemma_plot)
        ax[i, 1].imshow(ols_plot)
        ax[i, 2].imshow(gemma_plot)

        ax[i, 0].set_title(f'PyGemma Pheno{pheno}')
        ax[i, 1].set_title(f'OLS Pheno{pheno}')
        ax[i, 2].set_title(f'GEMMA Pheno{pheno}')

        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTPATH, 'Homework3_manhatten_summary.png'))
    plt.close(fig)
    plt.clf()


    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    for i, pheno in enumerate(range(1,4)):
        
        pygemma_plot = plt.imread(os.path.join(PYGEMMAPLOTPATH, f'Homework3_Pheno{pheno}_wald_qq.png'))
        #pygemma_plot = plt.imread(os.path.join(PLOTPATH, f'Homework3_Pheno{pheno}_wald_qq.png'))
        ols_plot = plt.imread(os.path.join(PLOTPATH, f'Homework3_Pheno{pheno}_wald_qq_fixed.png'))
        gemma_plot = plt.imread(os.path.join(PLOTPATH, f'Homework3_Pheno{pheno}_gemma_wald_qq.png'))

        ax[i, 0].imshow(pygemma_plot)
        ax[i, 1].imshow(ols_plot)
        ax[i, 2].imshow(gemma_plot)

        ax[i, 0].set_title(f'PyGemma Pheno{pheno}')
        ax[i, 1].set_title(f'OLS Pheno{pheno}')
        ax[i, 2].set_title(f'GEMMA Pheno{pheno}')

        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTPATH, 'Homework3_qq_summary.png'))
    plt.close(fig)
    plt.clf()


