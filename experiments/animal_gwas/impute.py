import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import os

df_file = os.path.join('imputed_data', 'mouse_hs1940.fam')

df = pd.read_csv(df_file, sep='\t', header=None)

# Impute columns 5 to 10 (including 10)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

df.iloc[:, 5:11] = imputer.fit_transform(df.iloc[:, 5:11])

# Save the imputed data
df.to_csv(os.path.join('imputed_data', 'mouse_hs1940_imputed.fam'), sep='\t', index=False, header=False)