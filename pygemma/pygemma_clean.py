import numpy as np
import pandas as pd
from rich.progress import track

from scipy import optimize, stats

#from pygemma import pygemma_model

def pygemma(Y, X, W, K, verbose=0):
    results_dict = {
                        'beta'    : [],
                        'se_beta' : [],
                        'tau'     : [],
                        'lambda'  : [],
                        #'D_lrt'   : [],
                        #'p_lrt'   : [],
                        'F_wald'  : [],
                        'p_wald'  : []
                    }

    eigenVals, U = np.linalg.eig(K) # Perform eigendecomposition

    eigenVals = np.maximum(0, eigenVals)

    assert (eigenVals >= 0).all()

    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    # Calculate under null
    #lambda_null = calc_lambda_restricted(eigenVals, U, Y, W, None)
    #_, _, tau = calc_beta_vg_ve_restricted(eigenVals, U, W, None, lambda_null, Y)
    #l_null = likelihood_restricted(lambda_null, tau, eigenVals, U, Y, W, None)

    n, c = W.shape

    if verbose > 0:
        progress_bar = track(range(X.shape[1]), description='Fitting Models...')
    else:
        progress_bar = range(X.shape[1])

    for g in progress_bar:
        lambda_restricted = calc_lambda_restricted(eigenVals, U, Y, np.c_[W, X[:,g]])
        beta, se_beta, tau = calc_beta_vg_ve_restricted(eigenVals, U, W, X[:,g], lambda_restricted, Y)

        # Fix these calculations later
        l_alt = likelihood_restricted(lambda_restricted, tau, eigenVals, U, Y, np.c_[W, X[:,g]])

        F_wald = np.power(beta/se_beta, 2.0)
        #D_lrt = 2 * np.log10(l_alt/l_null)

        # Store values
        results_dict['beta'].append(beta)
        results_dict['se_beta'].append(se_beta)
        results_dict['tau'].append(tau)
        results_dict['lambda'].append(lambda_restricted)
        results_dict['F_wald'].append(F_wald)
        results_dict['p_wald'].append(1-stats.chi2.cdf(x=F_wald, df=1))
        #results_dict['D_lrt'].append(D_lrt)
        #results_dict['p_lrt'].append(1-stats.f.cdf(x=D_lrt, dfn=1, dfd=n-c-1))


    results_df = pd.DataFrame.from_dict(results_dict)

    return results_df

def calc_beta_vg_ve(eigenVals, U, W, x, lam, Y):
    Px = compute_Pc(eigenVals, U, W, lam)
    
    n, c = W.shape

    W_x = np.c_[W,x]

    W_xt_Px = W_x.T @ Px
    beta_vec = np.linalg.inv(W_xt_Px @ W_x) @ (W_xt_Px @ Y)
    beta = beta_vec[-1]

    ytPxy = Y.T @ Px @ Y

    se_beta = np.sqrt(ytPxy/((n - c - 1) * (x.T @ Px @ x)))

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

    se_beta = np.sqrt(ytPxy/((n - c - 1) * (x.T @ Pc @ x)))

    tau = (n-c-1)/ytPxy

    return float(beta), float(se_beta), float(tau)

def compute_Pc(eigenVals, U, W, lam):
    H_inv = U.T @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U
    W_x = W

    return H_inv - H_inv @ W_x @ np.linalg.inv(W_x.T @ H_inv @ W_x) @ W_x.T @ H_inv

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

    result = result - n * (yT_Px_G_Px_G_Px_y @ yT_Px_y - yT_Px_G_Px_y @ yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)

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

    result = result - (n - c) * (yT_Px_G_Px_G_Px_y @ yT_Px_y - yT_Px_G_Px_y @ yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)

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
    result = result - 0.5*(n - c - 1)*np.log(2*np.pi)
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
    result = result - 0.5*(n - c - 1)
    _, logdet = np.linalg.slogdet(W.T @ W)
    result = result + 0.5*logdet
    
    result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

    #result = result - 0.5*np.log(np.linalg.det(W_x.T @ H_inv @ W_x)) # Causing NAN
    _, logdet = np.linalg.slogdet(W.T @ H_inv @ W)
    result = result - 0.5*logdet # Causing NAN

    result = result - 0.5*(n - c)*np.log(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)

    return float(result)


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
                                                xtol=0.1,
                                                maxiter=5000)
            
            lambda_min = optimize.newton(func=lambda l: likelihood_derivative1_lambda(l, eigenVals, U, Y, W), 
                                         x0=lambda_min,
                                         fprime=lambda l: likelihood_derivative2_lambda(l, eigenVals, U, Y, W),
                                         maxiter=100)
            
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
                                                tol=0.1,
                                                maxiter=5000)
        
            lambda_min = optimize.newton(func=lambda l: likelihood_derivative1_restricted_lambda(l, eigenVals, U, Y, W), 
                                         x0=lambda_min,
                                         tol=0.1,
                                         fprime=lambda l: likelihood_derivative2_restricted_lambda(l, eigenVals, U, Y, W),
                                         maxiter=100)
            
            roots.append(lambda_min)
        
        

    likelihood_list = [likelihood_restricted_lambda(lam, eigenVals, U, Y, W, x) for lam in roots]

    return roots[np.argmax(likelihood_list)]


