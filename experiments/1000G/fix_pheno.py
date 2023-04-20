import numpy as np
import pandas as pd
import qnorm

import argparse
import os

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Path to input file", type=str, default=None)
    parser.add_argument("-o", "--output", dest="output", help="Path to output file", type=str, default="pheno.tsv")

    args = parser.parse_args()

    # Read in input file
    df = pd.read_csv(args.input)

    # Normalize phenotype
    Y = df['Exp_Value'].values.reshape(-1,1)
    #Y = Y - Y.mean()
    Y = qnorm.quantile_normalize(Y, axis=1)

    # Write to output file (as tsv)
    pd.DataFrame(Y).to_csv(args.output, 
                           sep='\t', 
                           index=False, 
                           header=False, float_format='%.15f')
