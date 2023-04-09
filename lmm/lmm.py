import numpy as np
import pandas as pd

try:
    from rich.progress import track
    from rich.console import Console
except ModuleNotFoundError:
    print('Issues with rich for progress tracking')
    

from scipy import optimize, stats

import time

def calc_lambda(eigenVals, U, Y, W):
    # Loop over intervals and find where likelihood changes signs with respect to lambda
    step = 1.0

    lambda_pow_low = -5.0
    lambda_pow_high = 5.0

    lambda_possible = [(np.power(10.0, i), np.power(10.0, i+step)) for i in np.arange(lambda_pow_low,lambda_pow_high,step)]

    roots = [np.power(10.0, lambda_pow_low), np.power(10.0, lambda_pow_high)]
    
    for lambda0, lambda1 in lambda_possible:
    
        likelihood_lambda0 = likelihood_derivative1_lambda(lambda0, eigenVals, U, Y, W)
        likelihood_lambda1 = likelihood_derivative1_lambda(lambda1, eigenVals, U, Y, W)

        if np.sign(likelihood_lambda0) * np.sign(likelihood_lambda1) < 0:
            lambda_min = optimize.brentq(f=lambda l: likelihood_derivative1_lambda(l, eigenVals, U, Y, W), 
                                                a=lambda0, 
                                                b=lambda1,
                                                rtol=0.1,
                                                maxiter=5000,
                                                disp=False)
            
            lambda_min = optimize.newton(func=lambda l: likelihood_derivative1_lambda(l, eigenVals, U, Y, W), 
                                        x0=lambda_min,
                                        rtol=1e-5,
                                        fprime=lambda l: likelihood_derivative2_lambda(l, eigenVals, U, Y, W),
                                        maxiter=10,
                                        disp=False)
            
            roots.append(lambda_min)
            


    likelihood_list = [likelihood_lambda(lam, eigenVals, U, Y, W) for lam in roots]

    return roots[np.argmax(likelihood_list)]

def calc_lambda_restricted(eigenVals, U, Y, W):
    # Loop over intervals and find where likelihood changes signs with respect to lambda
    step = 1.0

    lambda_pow_low = -5.0
    lambda_pow_high = 5.0

    lambda_possible = [(np.power(10.0, i), np.power(10.0, i+step)) for i in np.arange(lambda_pow_low,lambda_pow_high,step)]

    roots = [np.power(10.0, lambda_pow_low), np.power(10.0, lambda_pow_high)]
    
    for lambda0, lambda1 in lambda_possible:
        likelihood_lambda0 = likelihood_derivative1_restricted_lambda(lambda0, eigenVals, U, Y, W)
        likelihood_lambda1 = likelihood_derivative1_restricted_lambda(lambda1, eigenVals, U, Y, W)


        if np.sign(likelihood_lambda0) * np.sign(likelihood_lambda1) < 0:
            lambda_min = optimize.brentq(f=lambda l: likelihood_derivative1_restricted_lambda(l, eigenVals, U, Y, W), 
                                                a=lambda0,
                                                b=lambda1,
                                                rtol=0.1,
                                                maxiter=5000,
                                                disp=False)
            
            
            # TODO: Deal with lack of convergence
            lambda_min = optimize.newton(func=lambda l: likelihood_derivative1_restricted_lambda(l, eigenVals, U, Y, W), 
                                    x0=lambda_min,
                                    rtol=1e-5,
                                    fprime=lambda l: likelihood_derivative2_restricted_lambda(l, eigenVals, U, Y, W),
                                    maxiter=10,
                                    disp=False)


            roots.append(lambda_min)

    likelihood_list = [likelihood_restricted_lambda(lam, eigenVals, U, Y, W) for lam in roots]

    return roots[np.argmax(likelihood_list)]




def pygemma(Y, X, W, K, snps=None, verbose=0):
    if True: # Fix compatability issues with Python<=3.6.9 later (issue with rich package)
        console = Console()
    else:
        verbose = 0

    Y = Y.astype(np.float32).reshape(-1,1)
    W = W.astype(np.float32)
    X = X.astype(np.float32)

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
            eigenVals, U = np.linalg.eig(K) # Perform eigendecomposition
            eigenVals = eigenVals.astype(np.float32)
            U = U.astype(np.float32)

            eigenVals = np.maximum(0, eigenVals)

            assert (eigenVals >= 0).all()
            console.log(f"[green]Eigendecomposition computed - {round(time.time() - start,3)} s")

            start = time.time()
            X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
            console.log(f"[green]Genotype matrix centered - {round(time.time() - start,3)} s")

            console.log(f"[green]Running {X.shape[1]} SNPs with {Y.shape[0]} individuals...")

            # Calculate under null
            n, c = W.shape

            # start = time.time()
            # lambda_null = calc_lambda(eigenVals, U, Y, W)
            # console.log(f"[green]Null lambda computed: {round(lambda_null, 5)} - {round(time.time() - start,3)} s")

            # start = time.time()
            # Pc = compute_Pc(eigenVals, U, W, lambda_null)
    
            # Wt_Pc = W.T @ Pc
            # beta_vec_null = np.linalg.inv(Wt_Pc @ W) @ (Wt_Pc @ Y)
            # tau_null = float(n / (Y.T @ Pc @ Y))
            # console.log(f"[green]Null tau computed: {round(tau_null, 5)} - {round(time.time() - start,3)} s")

            # start = time.time()
            # l_null = likelihood(lambda_null, tau_null, beta_vec_null, eigenVals, U, Y, W)
            # console.log(f"[green]Null likelihood computed: {round(l_null, 5)} - {round(time.time() - start,3)} s")
    else:
        eigenVals, U = np.linalg.eig(K) # Perform eigendecomposition
        eigenVals = eigenVals.astype(np.float32)
        U = U.astype(np.float32)

        eigenVals = np.maximum(0, eigenVals)

        assert (eigenVals >= 0).all()

        X = (X - np.mean(X, axis=0))#/np.std(X, axis=0)

        # Calculate under null
        n, c = W.shape

        # lambda_null = calc_lambda(eigenVals, U, Y, W)

        # Pc = compute_Pc(eigenVals, U, W, lambda_null)
    
        # Wt_Pc = W.T @ Pc
        # beta_vec_null = np.linalg.inv(Wt_Pc @ W) @ (Wt_Pc @ Y)
        # tau_null = float(n / (Y.T @ Pc @ Y))

        # l_null = likelihood(lambda_null, tau_null, beta_vec_null, eigenVals, U, Y, W)


    if verbose > 0:
        progress_bar = track(range(X.shape[1]), description='Testing SNPs...')
    else:
        progress_bar = range(X.shape[1])

    for g in progress_bar:
        try:
            lambda_restricted = calc_lambda_restricted(eigenVals, U, Y, np.c_[W, X[:,g]])
            beta, beta_vec, se_beta, tau = calc_beta_vg_ve_restricted(eigenVals, U, W, X[:,g], lambda_restricted, Y)

            F_wald = np.power(beta/se_beta, 2.0)

            #lambda_alt = calc_lambda(eigenVals, U, Y, np.c_[W, X[:,g]])
            #beta, beta_vec, se_beta, tau = calc_beta_vg_ve(eigenVals, U, W, X[:,g], lambda_alt, Y)

            # #Fix these calculations later
            #l_alt = likelihood(lambda_alt, tau, beta_vec, eigenVals, U, Y, np.c_[W, X[:,g]])
            #D_lrt = 2 * (l_alt - l_null)
        except np.linalg.LinAlgError as e:
            beta = np.nan
            se_beta = np.nan
            tau = np.nan
            lambda_restricted = np.nan
            F_wald = np.nan
            D_lrt = np.nan

        # Store values
        results_dict['beta'].append(beta)
        results_dict['se_beta'].append(se_beta)
        results_dict['tau'].append(tau)
        results_dict['lambda'].append(lambda_restricted)
        results_dict['F_wald'].append(F_wald)
        results_dict['p_wald'].append(1-stats.f.cdf(x=F_wald, dfn=1, dfd=n-c-1))

        #results_dict['D_lrt'].append(D_lrt)
        #results_dict['p_lrt'].append(1-stats.chi2.cdf(x=D_lrt, df=1))

        #l_alt = likelihood_restricted(lambda_restricted, tau, eigenVals, U, Y, np.c_[W, X[:,g]])
        #results_dict['likelihood'].append(l_alt)


    results_df = pd.DataFrame.from_dict(results_dict)

    if snps is not None:
        results_df['SNPs'] = snps

    return results_df

try:
    from pygemma.pygemma_model import *
    print("Using Cython version of pyGEMMA")

except ImportError:
    print("Cython not available, using Python version of pyGEMMA")

    def compute_Pc(eigenVals, U, W, lam):
        H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T
        W_x = W

        return H_inv - H_inv @ W_x @ np.linalg.inv(W_x.T @ H_inv @ W_x) @ W_x.T @ H_inv
    
    def calc_beta_vg_ve(eigenVals, U, W, x, lam, Y):
        W_x = np.c_[W,x]
        Px = compute_Pc(eigenVals, U, W_x, lam)
        
        n, c = W.shape


        W_xt_Px = W_x.T @ Px
        beta_vec = np.linalg.inv(W_xt_Px @ W_x) @ (W_xt_Px @ Y)
        beta = beta_vec[-1]

        ytPxy = Y.T @ Px @ Y

        se_beta = (1/np.sqrt((n - c - 1))) * np.sqrt(ytPxy)/np.sqrt(x.T @ Px @ x)

        tau = n/ytPxy

        return beta, se_beta, tau

    def calc_beta_vg_ve_restricted(eigenVals, U, W, x, lam, Y):
        W_x = np.c_[W,x]

        Px = compute_Pc(eigenVals, U, W_x, lam)
        Pc = compute_Pc(eigenVals, U, W, lam)

        n, c = W.shape

        W_xt_Pc = W_x.T @ Pc
        beta_vec = np.linalg.inv(W_xt_Pc @ W_x) @ (W_xt_Pc @ Y)
        beta = beta_vec[-1]

        ytPxy = Y.T @ Px @ Y

        se_beta = (1/np.sqrt((n - c - 1))) * np.sqrt(ytPxy)/np.sqrt(x.T @ Pc @ x)

        tau = (n-c-1)/ytPxy

        return float(beta), float(se_beta), float(tau)

    def likelihood_lambda(lam, eigenVals, U, Y, W):
        n = Y.shape[0]

        result = (n/2)*np.log(n/(2*np.pi))

        result = result - n/2

        result = result - 0.5*np.sum(np.log(lam*eigenVals + 1.0))

        result = result - (n/2)*np.log(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)

        return float(result)

    def likelihood_derivative1_lambda(lam, eigenVals, U, Y, W):
        n = Y.shape[0]

        result = -0.5*((n-np.sum(1/(lam*eigenVals + 1.0)))/lam)

        Px = compute_Pc(eigenVals, U, W, lam)

        yT_Px_y = Y.T @ Px @ Y

        yT_Px_G_Px_y = (yT_Px_y - (Y.T @ Px) @ (Px @ Y))/lam

        result = result - (n/2)*yT_Px_G_Px_y/yT_Px_y

        return float(result)

    def likelihood_derivative2_lambda(lam, eigenVals, U, Y, W): 
        n = Y.shape[0]

        result = 0.5*(n + np.sum(np.power(lam*eigenVals + 1.0, -2)) + 2*np.sum(np.power(lam*eigenVals + 1.0,-1)))

        Px = compute_Pc(eigenVals, U, W, lam)

        yT_Px_y = Y.T @ Px @ Y

        yT_Px_Px_y = (Y.T @ Px) @ (Px @ Y)

        yT_Px_G_Px_G_Px_y = (yT_Px_y + (Y.T @ Px) @ Px @ (Px @ Y) - 2*yT_Px_Px_y)/(lam*lam)

        yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

        result = result - 0.5 * n * (2 * yT_Px_G_Px_G_Px_y * yT_Px_y - yT_Px_G_Px_y * yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)

        return float(result)

    def likelihood_derivative1_restricted_lambda(lam, eigenVals, U, Y, W):
        n = W.shape[0]
        c = W.shape[1]

        Px = compute_Pc(eigenVals, U, W, lam)

        result = -0.5*((n-c - np.trace(Px))/lam)

        yT_Px_y = Y.T @ Px @ Y

        yT_Px_G_Px_y = (yT_Px_y - (Y.T @ Px) @ (Px @ Y))/lam

        result = result - 0.5*(n - c)*yT_Px_G_Px_y/yT_Px_y

        return float(result)

    def likelihood_derivative2_restricted_lambda(lam, eigenVals, U, Y, W): 
        n = Y.shape[0]
        c = W.shape[1]

        Px = compute_Pc(eigenVals, U, W, lam)

        yT_Px_y = Y.T @ Px @ Y

        yT_Px_Px_y = (Y.T @ Px) @ (Px @ Y)

        yT_Px_G_Px_G_Px_y = (yT_Px_y + (Y.T @ Px) @ Px @ (Px @ Y) - 2*yT_Px_Px_y)/(lam*lam)

        yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

        result = 0.5*(n - c + np.trace(Px @ Px) - 2*np.trace(Px))/(lam*lam)

        result = result - 0.5 * (n - c) * (2 * yT_Px_G_Px_G_Px_y @ yT_Px_y - yT_Px_G_Px_y @ yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)

        return float(result)

    def likelihood(lam, tau, beta, eigenVals, U, Y, W):
        n = W.shape[0]

        H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T

        result = (n/2)*np.log(tau) 
        result = result - (n/2)*np.log(2*np.pi)
        
        result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

        y_W_beta = Y - W @ beta

        result = result - 0.5 * tau * y_W_beta.T @ H_inv @ y_W_beta


        return float(result)

    def likelihood_restricted(lam, tau, eigenVals, U, Y, W):
        n = W.shape[0]
        c = W.shape[1]

        H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T

        result = 0.5*(n - c)*np.log(tau)
        result = result - 0.5*(n - c)*np.log(2*np.pi)
        _, logdet = np.linalg.slogdet(W.T @ W)
        result = result + 0.5*logdet

        result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

        #result = result - 0.5*np.log(np.linalg.det(W_x.T @ H_inv @ W_x)) # Causing NAN
        _, logdet = np.linalg.slogdet(W.T @ H_inv @ W)
        result = result - 0.5*logdet # Causing NAN

        result = result - 0.5*(n - c)*np.log(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)

        return float(result)

    def likelihood_restricted_lambda(lam, eigenVals, U, Y, W):
        n, c = W.shape

        H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T

        result = 0.5*(n - c)*np.log(0.5*(n - c)/np.pi)
        result = result - 0.5*(n - c)
        _, logdet = np.linalg.slogdet(W.T @ W)
        result = result + 0.5*logdet
        
        result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

        #result = result - 0.5*np.log(np.linalg.det(W_x.T @ H_inv @ W_x)) # Causing NAN
        _, logdet = np.linalg.slogdet(W.T @ H_inv @ W)
        result = result - 0.5*logdet # Causing NAN

        result = result - 0.5*(n - c)*np.log(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)

        return float(result)


