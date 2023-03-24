import numpy as np
from scipy import optimize

#from pygemma import pygemma_model

# TODO: Implement GEMMA model call
# See https://github.com/genetics-statistics/GEMMA/blob/master/src/lmm.cpp#L2208
def pygemma(Y, X, W, K):
    eigenVals, U = np.linalg.eig(K)




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

    result = result - (n*n/2) * (yT_Px_G_Px_G_Px_y @ yT_Px_y - yT_Px_G_Px_y @ yT_Px_G_Px_y) / (yT_Px_y @ yT_Px_y)

    return np.float32(result)


def CalcLambda(eigenVals, U, Y, W, x):
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