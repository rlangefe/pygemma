import time
import os

import numpy as np
import pandas as pd

import pygemma

from rich.console import Console

console = Console()

def run_function_test(function, parameters):
    failed = False

    start = time.time()
    
    try:
        result = function(*parameters)
    except Exception as e:
        print(e)
        failed = True
        
    diff = str(round(time.time() - start, 4))
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


def generate_test_matrices(n=1000, covars=10):
    K = np.random.uniform(size=(n, n))
    K = np.abs(np.tril(K) + np.tril(K, -1).T)
    K = np.dot(K, K.T)
    eigenVals, U = np.linalg.eig(K)
    W = np.random.rand(n, covars)
    x = np.random.choice([0,1,2], 
                        size=(n, 1),
                        replace=True)
    Y = np.random.rand(n, 1)
    lam = 5
    tau = 10
    beta = np.random.rand(covars+1, 1)

    return x, Y, W, eigenVals, U, lam, beta, tau

with console.status("[bold green]Running pyGEMMA Function Run Tests...") as status:

    # Seed tests
    np.random.seed(42)

    n = 1000
    covars = 10

    # Initializing parameters for tests
    x, Y, W, eigenVals, U, lam, beta, tau = generate_test_matrices(n=n, covars=covars)
    
    console.log(f'Test Parameters: n={n}, lam={lam}, tau={tau}')
    
    functions_and_args = [
                            (pygemma.compute_Px, [eigenVals, U, W, x, lam]),
                            (pygemma.likelihood_lambda, [lam, eigenVals, U, Y, W, x]),
                            (pygemma.likelihood_derivative1_lambda, [lam, eigenVals, U, Y, W, x]),
                            (pygemma.likelihood_derivative2_lambda, [lam, eigenVals, U, Y, W, x]),
                            (pygemma.calc_lambda, [eigenVals, U, Y, W, x]),
                            (pygemma.calc_lambda_restricted, [eigenVals, U, Y, W, x]),
                            (pygemma.likelihood, [lam, tau, beta, eigenVals, U, Y, W, x]),
                            (pygemma.likelihood_restricted, [lam, tau, eigenVals, U, Y, W, x]),
                            (pygemma.likelihood_restricted_lambda, [lam, eigenVals, U, Y, W, x]),
                            (pygemma.likelihood_derivative1_restricted_lambda, [lam, eigenVals, U, Y, W, x]),
                            (pygemma.likelihood_derivative2_restricted_lambda, [lam, eigenVals, U, Y, W, x]),
                          ]
    
    run_test_list(functions_and_args)

DATADIR = r"C:\Users\lange\Desktop\School\Michigan\MS Year 1\Spring\BIOSTAT 666\Final Project\pygemma\data"

dataset_list = [
        {
            'name'   : 'mouse_hs1940',
            'snps'   : os.path.join(DATADIR, "mouse_hs1940_snps.txt"),
            'covars' : None,
            'pheno'  : os.path.join(DATADIR, "mouse_hs1940.txt"),
            
        }
    ]

exit(0)

for dataset in dataset_list:
    dataset_name = dataset['name']

    Y = 

    with console.status("[bold green]Running pyGEMMA Function Tests - {dataset_name}...") as status:

        # Seed tests
        np.random.seed(42)

        # Initializing parameters for tests
        x, Y, W, eigenVals, U, lam = generate_test_matrices(n=1000, covars=10)
        
        console.log(f'Test Parameters: n={n}, lam={lam}')
        
        functions_and_args = [
                                (pygemma.compute_Px, [eigenVals, U, W, x, lam]),
                                (pygemma.likelihood_lambda, [lam, eigenVals, U, Y, W, x]),
                                (pygemma.likelihood_derivative1_lambda, [lam, eigenVals, U, Y, W, x]),
                                (pygemma.likelihood_derivative2_lambda, [lam, eigenVals, U, Y, W, x]),
                                (pygemma.calc_lambda, [eigenVals, U, Y, W, x])
                            ]
        
        run_test_list(functions_and_args)

