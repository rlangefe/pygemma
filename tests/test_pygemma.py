import numpy as np
from rich.console import Console
import time

import pygemma

console = Console()

def run_function_test(function, parameters):
    failed = False

    start = time.time()
    
    try:
        result = function(*parameters)
    except Exception as e:
        print(e)
        failed = True
        
    diff = str(round(time.time() - start))
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


with console.status("[bold green]Running pyGEMMA Function Tests...") as status:

    # Seed tests
    np.random.seed(42)

    # Initializing parameters for tests
    n = 1000
    covars = 10
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

    console.log(f'Test Parameters: n={n}, lam={lam}')
    
    functions_and_args = [
                            (pygemma.compute_Px, [eigenVals, U, W, x, lam]),
                            (pygemma.likelihood_lambda, [lam, eigenVals, U, Y, W, x]),
                            (pygemma.likelihood_derivative1_lambda, [lam, eigenVals, U, Y, W, x]),
                            (pygemma.likelihood_derivative2_lambda, [lam, eigenVals, U, Y, W, x]),
                            (pygemma.CalcLambda, [eigenVals, U, Y, W, x])
                          ]
    
    run_test_list(functions_and_args)    

