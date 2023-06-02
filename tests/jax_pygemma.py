import jax.numpy as jnp
import jax
from jax import grad, jit, vmap

import numpy as np

from functools import partial

import pandas as pd

from scipy import optimize, stats

from rich.console import Console
from rich.progress import track

import multiprocessing

import time

MIN_VAL = 1e-15

def pygemma_jax(Y, X, W, K, snps=None, verbose=0, nproc=1):
    if True: # Fix compatability issues with Python<=3.6.9 later (issue with rich package)
        console = Console()
    else:
        verbose = 0

    Y = jnp.array(Y)
    X = jnp.array(X)
    W = jnp.array(W)
    K = jnp.array(K)

    Y = Y.astype(jnp.float32).reshape(-1,1)
    W = W.astype(jnp.float32)
    X = X.astype(jnp.float32)

    Y = (Y - jnp.mean(Y, axis=0)) / jnp.std(Y, axis=0)

    results_dict = {
                        'beta'       : [],
                        'se_beta'    : [],
                        'tau'        : [],
                        'lambda'     : [],
                        #'D_lrt'      : [],
                        #'p_lrt'      : [],
                        'F_wald'     : [],
                        'p_wald'     : [],
                        #'likelihood' : []
                    }
    if verbose > 0:
        with console.status(f"[bold green]Running null model...") as status:

            start = time.time()
            eigenVals, U = jnp.linalg.eig(K) # Perform eigendecomposition
            eigenVals = eigenVals.astype(jnp.float32)
            U = U.astype(jnp.float32)

            eigenVals = jnp.maximum(0, eigenVals)

            assert (eigenVals >= 0).all()
            console.log(f"[green]Eigendecomposition computed - {round(time.time() - start,3)} s")

            start = time.time()
            X = (X - jnp.mean(X, axis=0))/jnp.std(X, axis=0)
            console.log(f"[green]Genotype matrix centered - {round(time.time() - start,3)} s")

            console.log(f"[green]Running {X.shape[1]} SNPs with {Y.shape[0]} individuals...")

            # Calculate under null
            n, c = W.shape
    else:
        eigenVals, U = jnp.linalg.eig(K) # Perform eigendecomposition
        eigenVals = eigenVals.astype(jnp.float32)
        U = U.astype(jnp.float32)

        eigenVals = jnp.maximum(0, eigenVals)

        assert (eigenVals >= 0).all()

        X = (X - jnp.mean(X, axis=0))/jnp.std(X, axis=0)

        # Calculate under null
        n, c = W.shape    

    X = U.T @ X
    Y = U.T @ Y
    W = U.T @ W

    if verbose > 0:
        progress_bar = range(X.shape[1])
    else:
        progress_bar = range(X.shape[1])

    with multiprocessing.Pool(nproc) as pool:
        total = X.shape[1]
        
        results = []
        done = 0
        for r in track(pool.imap(calculate, SampleIter(X, Y, W, eigenVals)), description='Testing SNPs...', total=total):
            # update the progress bar
            done += 1
            #progress.update(task, completed=done)
            
            results.append(r)

    # results = []
    # done = 0

    # for t in track(SampleIter(X, Y, W, eigenVals), description='Testing SNPs...', total=X.shape[1]):
    #     results.append(calculate(t))            

    results_df = pd.DataFrame.from_dict(results)
    

    if snps is not None:
        results_df['SNPs'] = snps

    return results_df

# Define functions
def calculate(t):
    eigenVals, Y, W, X = t
    try:
        lambda_restricted = calc_lambda_restricted(eigenVals, Y, jnp.c_[W, X])
        beta, beta_vec, se_beta, tau = calc_beta_vg_ve_restricted(eigenVals, W, X.reshape(-1,1), lambda_restricted, Y)
        F_wald = jnp.power(beta/se_beta, 2.0)

        n = Y.shape[0]
        c = W.shape[1]

        return {
            'beta': beta,
            'se_beta': se_beta,
            'tau': tau,
            'lambda': lambda_restricted,
            'F_wald': F_wald,
            'p_wald': 1-stats.f.cdf(x=F_wald, dfn=1, dfd=n-c-1),
        }
    #except np.linalg.LinAlgError as e:
    except Exception as e:
        print(e)
        return {
            'beta': jnp.nan,
            'se_beta': jnp.nan,
            'tau': jnp.nan,
            'lambda': jnp.nan,
            'F_wald': jnp.nan,
            'p_wald': jnp.nan,
        }

def calc_beta_vg_ve_restricted(eigenVals,
                                W, 
                                x, 
                                lam, 
                                Y):
    W_x = jnp.c_[W,x]

    n = W.shape[0]
    c = W.shape[1]
    
    mod_eig = lam*eigenVals + 1.0

    W_x_t_H_inv = ((1.0/mod_eig)[:,jnp.newaxis] * W_x).T
    
    beta_vec = jnp.linalg.inv(W_x_t_H_inv @ W_x) @ (W_x_t_H_inv @ Y)
    
    beta = beta_vec[c,0]

    ytPxy = max(compute_at_Pi_b(mod_eig, W_x, Y, Y), MIN_VAL)

    se_beta = jnp.sqrt(ytPxy) / (jnp.sqrt(max(compute_at_Pi_b(mod_eig, W, x, x), MIN_VAL)) * jnp.sqrt((n - c - 1)))

    tau = (n-c-1)/ytPxy

    return jnp.float32(beta), beta_vec, jnp.float32(se_beta), jnp.float32(tau)

def calc_lambda_restricted(eigenVals,  
                            Y, 
                            W):
    # Loop over intervals and find where likelihood changes signs with respect to lambda
    step = 1.0

    lambda_pow_low = -5.0
    lambda_pow_high = 5.0

    roots = [10.0 ** lambda_pow_low, 10.0 ** lambda_pow_high]
    likelihood_list = [likelihood_restricted_lambda(roots[0], eigenVals, Y, W), likelihood_restricted_lambda(roots[1], eigenVals, Y, W)]

    maxiter = 5000
    
    lambda_possible = jnp.arange(lambda_pow_low,lambda_pow_high,step, dtype=jnp.float32)
    likelihood_lambda1 = 0.0

    for idx in range(lambda_possible.shape[0]):
        lambda_idx = lambda_possible[idx]

        lambda0 = 10.0 ** (lambda_idx)
        lambda1 = 10.0 ** (lambda_idx + step)

        # If it's the first iteration
        if idx == 0:
            # Compute lower lambda
            likelihood_lambda0 = likelihood_derivative1_restricted_lambda(lambda0, eigenVals, Y, W)

        else:
            # Reuse lambda likelihood from previous iteration
            likelihood_lambda0 = likelihood_lambda1

        
        likelihood_lambda1 = likelihood_derivative1_restricted_lambda(lambda1, eigenVals, Y, W)

        if jnp.sign(likelihood_lambda0) * jnp.sign(likelihood_lambda1) < 0:
            lambda_min = optimize.brentq(f=likelihood_derivative1_restricted_lambda, 
                                        a=lambda0,
                                        b=lambda1,
                                        rtol=0.1,
                                        maxiter=maxiter,
                                        args=(eigenVals, Y, W),
                                        disp=False)

            lambda_min = newton(lambda_min, eigenVals, Y, W)

            roots.append(lambda_min)

            likelihood_list.append(likelihood_restricted_lambda(lambda_min, eigenVals, Y, W))
    
    return roots[np.argmax(likelihood_list)]

def newton(lam,
            eigenVals, 
            Y, 
            W):

    lambda_min = lam
    iter = 0
    r_eps = 0.0

    while True:
        d1 = likelihood_derivative1_restricted_lambda(lambda_min, eigenVals, Y, W)
        d2 = likelihood_derivative2_restricted_lambda(lambda_min, eigenVals, Y, W)

        ratio = d1/d2

        if jnp.sign(ratio) * jnp.sign(d1) * jnp.sign(d2) <= 0.0:
            break

        lambda_new = lambda_min - ratio
        r_eps = jnp.abs(lambda_new - lambda_min) / jnp.abs(lambda_min)

        if lambda_new < 0.0 or jnp.isnan(lambda_new) or jnp.isinf(lambda_new):
            break

        lambda_min = lambda_new

        if r_eps < 1e-5 or iter > 100:
            break

        iter += 1

    return lambda_min

#@partial(jax.jit, static_argnames=['eigenVals', 'Y'])
def likelihood_restricted_lambda(lam, 
                                eigenVals, 
                                Y, 
                                W):
    n = W.shape[0]
    c = W.shape[1]
    
    mod_eig = lam*eigenVals + 1.0
    Wt_eig_W = W.T @ (W / mod_eig[:, jnp.newaxis])

    result = 0.5*(n - c)*jnp.log(0.5*(n - c)/jnp.pi)
    result = result - 0.5*(n - c)
    result = result + 0.5*jnp.linalg.slogdet(W.T @ W)[1]
    result = result - 0.5 * jnp.sum(jnp.log(mod_eig))

    result = result - 0.5*jnp.linalg.slogdet(Wt_eig_W)[1]

    result = result - 0.5*(n - c)*jnp.log(max(compute_at_Pi_b(mod_eig, W, Y, Y), MIN_VAL))

    return result

#@partial(jax.jit, static_argnames=['eigenVals'])
def compute_at_Pi_b(eigenVals,
                    W,
                    a,
                    b):
    result = 0.0
    result2 = 0.0

    # a.T @ (H_inv - H_inv @ W @ (W.T @ H_inv @ W)^-1 @ W.T @ H_inv ) @ b

    # a.T @ H_inv @ b
    result = jnp.sum(jnp.divide(jnp.multiply(a, b), eigenVals[:, None]))

    # H_inv @ W
    inv = jnp.divide(W, eigenVals[:, None])

    # W.T @ H_inv @ W
    inv = jnp.dot(W.T, inv)

    # (W.T @ H_inv @ W)^-1
    inv = jnp.linalg.inv(inv)

    # H_inv @ b
    temp_vec = b.copy()
    temp_vec = jnp.divide(temp_vec, eigenVals[:, None])

    # (H_inv @ a).T
    temp_vec2 = a.copy()
    temp_vec2 = jnp.divide(temp_vec2, eigenVals[:, None]).T

    # W.T @ H_inv @ b
    #temp_vec = mat_vec_mult(W.T, temp_vec)
    temp_vec = jnp.dot(W.T, temp_vec)

    # a.T @ H_inv @ W
    temp_vec2 = jnp.dot(temp_vec2, W)

    # (W.T @ H_inv @ W)^-1 @ W.T @ H_inv @ b
    temp_vec = jnp.dot(inv, temp_vec)

    result2 = jnp.dot(temp_vec2, temp_vec)
    
    return (result - result2)[0,0]

likelihood_derivative1_restricted_lambda = grad(likelihood_restricted_lambda, argnums=0)
likelihood_derivative2_restricted_lambda = grad(likelihood_derivative1_restricted_lambda, argnums=0)

class SampleIter:
    def __init__ (self, X, Y, W, eigenVals):
        self.X = X
        self.Y = Y
        self.W = W
        self.eigenVals = eigenVals
        self.g = 0
        self.c = X.shape[1]
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.g < self.c:
            self.g += 1
            return (self.eigenVals, self.Y, self.W, self.X[:,self.g-1])
        else:
            raise StopIteration

