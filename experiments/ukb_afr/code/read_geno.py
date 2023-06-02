import pandas as pd
import numpy as np
from pysnptools.snpreader import Bed

# Read PLINK files (.bed, .fam, .bim)
chr = 1
plink_data = Bed("/net/fantasia/home/borang/Robert/UKB_AFR/Geno/AFR/chr_{}.bed".format(chr), count_A1=False)

# ID info which can be matched up eid
geno_id = pd.Series(plink_data.iid[:, 1])

# Genotype
genotypes = plink_data.read(dtype=np.float32)