import numpy as np
import pandas as pd

import argparse
import os

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Path to input file", type=str, default="input.tsv")
    parser.add_argument("-o", "--output", dest="output", help="Path to output file", type=str, default="geno.tsv")

    args = parser.parse_args()

    # Read in input file
    df = pd.read_csv(args.input)
    gene_names = list(df.columns)

    # Normalize dataframe columns
    df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    #df = df.apply(lambda x: (x - x.mean()), axis=0)

    # Transpose df
    df = df.transpose().reset_index()

    df.columns = ['geneID'] + [f'sample{i}' for i in df.columns[1:]]

    # Write to output file (as tsv)
    df.to_csv(args.output, sep='\t', index=False, float_format='%f')


