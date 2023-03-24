import numpy as np
from scipy import optimize

#from pygemma import pygemma_model

# TODO: Implement GEMMA model call
# See https://github.com/genetics-statistics/GEMMA/blob/master/src/lmm.cpp#L2208
def pygemma(Y, X, W, K):
    eigenVals, U = np.linalg.eig(K) # Perform eigendecomposition



    for g in X.shape[1]:
        lambda_full = calc_lambda(eigenVals, U, Y, W, X[:,g])

    # TODO: Optimize by precomputing Px


def compute_Px(eigenVals, U, W, x, lam):
    H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T
    W_x = np.c_[W, x]

    return H_inv - H_inv @ W_x @ np.linalg.inv(W_x.T @ H_inv @ W_x) @ W_x.T @ H_inv

def likelihood_lambda(lam, eigenVals, U, Y, W, x):
    n = Y.shape[0]

    result = (n/2)*np.log(n/(2*np.pi))

    result = result - n/2

    result = result - 0.5*np.sum(np.log(lam*eigenVals + 1.0))

    result = result - (n/2)*np.log(Y.T @ compute_Px(eigenVals, U, W, x, lam) @ Y)

    return np.float32(result)

def likelihood_derivative1_lambda(lam, eigenVals, U, Y, W, x):
    n = Y.shape[0]

    result = -0.5*((n-np.sum(1/(lam*eigenVals + 1.0)))/lam)

    Px = compute_Px(eigenVals, U, W, x, lam)

    yT_Px_y = Y.T @ Px @ Y

    yT_Px_G_Px_y = (yT_Px_y - (Y.T @ Px) @ (Px @ Y))/lam

    result = result - (n/2)*yT_Px_G_Px_y/yT_Px_y

    return np.float32(result)

def likelihood_derivative2_lambda(lam, eigenVals, U, Y, W, x): 
    n = Y.shape[0]

    result = 0.5*(n + np.sum(np.power(lam*eigenVals + 1.0, -2)) + 2*np.sum(np.power(lam*eigenVals + 1.0,-1)))

    Px = compute_Px(eigenVals, U, W, x, lam)

    yT_Px_y = Y.T @ Px @ Y

    yT_Px_Px_y = (Y.T @ Px) @ (Px @ Y)

    yT_Px_G_Px_G_Px_y = (yT_Px_y + (Y.T @ Px) @ Px @ (Px @ Y) - 2*yT_Px_Px_y)/(lam*lam)

    yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

    result = result - n * (yT_Px_G_Px_G_Px_y @ yT_Px_y - yT_Px_G_Px_y @ yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)

    return np.float32(result)

def likelihood_derivative1_restricted_lambda(lam, eigenVals, U, Y, W, x):
    n = W.shape[0]
    c = W.shape[1] + 1

    Px = compute_Px(eigenVals, U, W, x, lam)

    result = -0.5*((n-c-1 - np.trace(Px))/lam)

    yT_Px_y = Y.T @ Px @ Y

    yT_Px_G_Px_y = (yT_Px_y - (Y.T @ Px) @ (Px @ Y))/lam

    result = result - 0.5*(n - c - 1)*yT_Px_G_Px_y/yT_Px_y

    return np.float32(result)

def likelihood_derivative2_restricted_lambda(lam, eigenVals, U, Y, W, x): 
    n = Y.shape[0]
    c = W.shape[1] + 1

    Px = compute_Px(eigenVals, U, W, x, lam)

    yT_Px_y = Y.T @ Px @ Y

    yT_Px_Px_y = (Y.T @ Px) @ (Px @ Y)

    yT_Px_G_Px_G_Px_y = (yT_Px_y + (Y.T @ Px) @ Px @ (Px @ Y) - 2*yT_Px_Px_y)/(lam*lam)

    yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

    result = 0.5*(n - c - 1 + np.trace(Px @ Px) - 2*np.trace(Px))/(lam*lam)

    result = result - (n - c - 1) * (yT_Px_G_Px_G_Px_y @ yT_Px_y - yT_Px_G_Px_y @ yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)

    return np.float32(result)

def likelihood(lam, tau, beta, eigenVals, U, Y, W, x):
    n = W.shape[0]

    W_x = np.c_[W,x]

    H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T

    result = (n/2)*np.log(tau) 
    result = result - (n/2)*np.log(2*np.pi)
    
    result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

    y_Wx_beta = Y - W_x @ beta

    result = result - 0.5 * tau * y_Wx_beta.T @ H_inv @ y_Wx_beta


    return np.float32(result)

def likelihood_restricted(lam, tau, eigenVals, U, Y, W, x):
    W_x = np.c_[W,x]

    n, c = W_x.shape

    H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T

    result = 0.5*(n - c - 1)*np.log(tau)
    result = result - 0.5*(n - c - 1)*np.log(2*np.pi)
    result = result + 0.5*np.log(np.linalg.det(W_x.T @ W_x))

    result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

    result = result - 0.5*np.log(np.linalg.det(W_x.T @ H_inv @ W_x)) # Possible bottleneck

    result = result - 0.5*(n - c - 1)*np.log(Y.T @ compute_Px(eigenVals, U, W, x, lam) @ Y)

    return np.float32(result)

def likelihood_restricted_lambda(lam, eigenVals, U, Y, W, x):
    n, c = W.shape

    W_x = np.c_[W,x]

    H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T

    result = 0.5*(n - c - 1)*np.log(0.5*(n - c - 1)/np.pi)
    result = result - 0.5*(n - c - 1)
    result = result + 0.5*np.log(np.linalg.det(W_x.T @ W_x))

    result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

    result = result - 0.5*np.log(np.linalg.det(W_x.T @ H_inv @ W_x)) # Possible bottleneck

    result = result - 0.5*(n - c - 1)*np.log(Y.T @ compute_Px(eigenVals, U, W, x, lam) @ Y)

    return np.float32(result)


def calc_lambda(eigenVals, U, Y, W, x):
    # Loop over intervals and find where likelihood changes signs with respect to lambda
    lambda0 = np.power(10.0, -5.0)
    lambda1 = np.power(10.0, 5.0)

    likelihood_lambda0 = likelihood_derivative1_lambda(lambda0, eigenVals, U, Y, W, x)
    likelihood_lambda1 = likelihood_derivative1_lambda(lambda1, eigenVals, U, Y, W, x)

    if likelihood_lambda0*likelihood_lambda1 < 0:
        lambda_min, _ = optimize.brentq(f=lambda l: likelihood_derivative1_lambda(l, eigenVals, U, Y, W, x), 
                                            a=lambda0, 
                                            b=lambda1,
                                            xtol=0.1,
                                            maxiter=1000)
    elif likelihood_lambda0 < 0:
        lambda_min = lambda0
    else:
        lambda_min = lambda1

    return lambda_min

def calc_lambda_restricted(eigenVals, U, Y, W, x):
    # Loop over intervals and find where likelihood changes signs with respect to lambda
    # TODO: Create range of values and loop over the brackets
    lambda0 = np.power(10.0, -5.0)
    lambda1 = np.power(10.0, 5.0)

    likelihood_lambda0 = likelihood_derivative1_restricted_lambda(lambda0, eigenVals, U, Y, W, x)
    likelihood_lambda1 = likelihood_derivative1_restricted_lambda(lambda1, eigenVals, U, Y, W, x)

    if likelihood_lambda0*likelihood_lambda1 < 0:
        lambda_min, _ = optimize.brentq(f=lambda l: likelihood_derivative1_restricted_lambda(l, eigenVals, U, Y, W, x), 
                                            a=lambda0, 
                                            b=lambda1,
                                            xtol=0.1,
                                            maxiter=5000)
        
        # TODO: Add Newton-Raphson iterations (scipy.optimize.newton) and catch error for no convergance on brent
        

    elif likelihood_lambda0 < 0:
        lambda_min = lambda0
    else:
        lambda_min = lambda1

    return lambda_min

# TODO: Implement unrestricted lambda estimates like with the calc_lambda_restricted function


