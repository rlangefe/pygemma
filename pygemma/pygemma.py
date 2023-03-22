import numpy as np
from scipy import optimize

from pygemma import pygemma_model

# TODO: Implement GEMMA model call
# See https://github.com/genetics-statistics/GEMMA/blob/master/src/lmm.cpp#L2208
def pygemma():
    UtW = np.dot(U.T W)
    UtY = np.dot(U.T, Y)

    pygemma_model.CalcLambda('L', eval, UtW, &UtY_col.vector, cPar.l_min, cPar.l_max,
                   cPar.n_region, cPar.l_mle_null, cPar.logl_mle_H0)
    
    pygemma_model.CalcLmmVgVeBeta(eval, UtW, &UtY_col.vector, cPar.l_mle_null,
                        cPar.vg_mle_null, cPar.ve_mle_null, &beta.vector,
                        &se_beta.vector)
    
def compute_aT_Px_b(a, H_inv, b, W):
    pass

def compute_aT_Px_Px_b(a, H_inv, b, W):
    pass

def compute_aT_Px_Px_Px_b(a, H_inv, b, W):
    pass

def likelihood(lam, eigenVals, U, Y, W):
    n = Y.shape[0]

    H_inv = np.linalg.inv(lam*eigenVals + 1.0)

    result = (n/2)*np.log(n/(2*np.pi))

    result = result - n/2

    result = result - 0.5*np.log(np.prod(lam*eigenVals + 1.0))

    result = result - (n/2)*np.log(compute_aT_Px_b(Y, H_inv, Y, W))

    return result

def likelihood_derivative1(lam, eigenVals, U, Y, W):
    n = Y.shape[0]

    H_inv = np.linalg.inv(lam*eigenVals + 1.0)

    result = -0.5*((n-np.sum(1/(lam*eigenVals + 1.0)))/lam)

    yT_Px_y = compute_aT_Px_b(Y, H_inv, Y, W)

    yT_Px_G_Px_y = (yT_Px_y - compute_aT_Px_Px_b(Y, H_inv, Y, W))/lam

    result = result - (n/2)*yT_Px_G_Px_y/yT_Px_y

    return result

def likelihood_derivative2(lam, eigenVals, U, Y, W): 
    n = Y.shape[0]

    H_inv = np.linalg.inv(lam*eigenVals + 1.0)

    result = 0.5*(n + np.sum(np.pow(lam*eigenVals + 1.0, -2)) + 2*np.sum(np.pow(lam*eigenVals + 1.0,-1)))

    yT_Px_y = compute_aT_Px_b(Y, H_inv, Y, W)

    yT_Px_Px_y = compute_aT_Px_Px_b(Y, H_inv, Y, W)

    yT_Px_G_Px_G_Px_y = (yT_Px_y + compute_aT_Px_Px_Px_b(Y, H_inv, Y, W) - 2*yT_Px_Px_y)/(lam*lam)

    yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

    result = result - (n*n/2) * (yT_Px_G_Px_G_Px_y*yT_Px_y - yT_Px_G_Px_y*yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)

    return result


def CalcLambda():
    # Loop over intervals and find where likelihood changes signs with respect to lambda
    lambda_sign_change = []
    
    likelihood_evals = []
    roots = []
    for lambda0, lambda1 in lambda_sign_change:
        roots.append(optimize.newton(f=lambda l: likelihood_derivative1(l, eigenVals, U, Y, W), 
                                     x0=lambda0, 
                                     fprime=likelihood_derivative2(l, eigenVals, U, Y, W)))
        
        likelihood_evals.append(likelihood(roots[-1], eigenVals, U, Y, W))

    return roots[np.argmin(likelihood_evals)]