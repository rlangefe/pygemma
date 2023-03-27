import cython
import numpy as np
cimport numpy as np

from scipy import optimize, stats

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def compute_Pc(
                np.ndarray[np.float32_t, ndim=1] eigenVals, 
                np.ndarray[np.float32_t, ndim=2] U, 
                np.ndarray[np.float32_t, ndim=2] W, 
                np.float32_t lam
                ):
    
    cdef np.ndarray[np.float32_t, ndim=2] H_inv = U.T @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U

    return H_inv - H_inv @ W @ np.linalg.inv(W.T @ H_inv @ W) @ W.T @ H_inv

##################
# Reimplementing #
##################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def compute_Pc_cython(
                np.ndarray[np.float32_t, ndim=1] eigenVals, 
                np.ndarray[np.float32_t, ndim=2] U, 
                np.ndarray[np.float32_t, ndim=2] W, 
                np.float32_t lam
                ):
    
    cdef np.ndarray[np.float32_t, ndim=2] H_inv = U.T @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U

    return H_inv - H_inv @ W @ np.linalg.inv(W.T @ H_inv @ W) @ W.T @ H_inv

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def calc_beta_vg_ve(np.ndarray[np.float32_t, ndim=1] eigenVals,
                    np.ndarray[np.float32_t, ndim=2] U, 
                    np.ndarray[np.float32_t, ndim=2] W, 
                    np.ndarray[np.float32_t, ndim=1] x, 
                    np.float32_t lam, 
                    np.ndarray[np.float32_t, ndim=2] Y):
    cdef np.ndarray[np.float32_t, ndim=2] W_x = np.c_[W,x]
    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W_x, lam)
    
    cdef int n, c
    n = W.shape[0]
    c = W.shape[1]


    cdef np.ndarray[np.float32_t, ndim=2] W_xt_Px = W_x.T @ Px
    cdef np.ndarray[np.float32_t, ndim=1] beta_vec = np.linalg.inv(W_xt_Px @ W_x) @ (W_xt_Px @ Y)
    beta = float(beta_vec[-1])

    cdef np.float32_t ytPxy = Y.T @ Px @ Y

    se_beta = np.sqrt(ytPxy/((n - c - 1) * (x.T @ Px @ x)))

    tau = n/ytPxy

    return beta, se_beta, tau

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def calc_beta_vg_ve_restricted(np.ndarray[np.float32_t, ndim=1] eigenVals,
                    np.ndarray[np.float32_t, ndim=2] U, 
                    np.ndarray[np.float32_t, ndim=2] W, 
                    np.ndarray[np.float32_t, ndim=1] x, 
                    np.float32_t lam, 
                    np.ndarray[np.float32_t, ndim=2] Y):
    cdef np.ndarray[np.float32_t, ndim=2] W_x = np.c_[W,x]

    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W_x, lam)
    cdef np.ndarray[np.float32_t, ndim=2] Pc = compute_Pc_cython(eigenVals, U, W, lam)

    cdef int n, c
    n = W.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] W_xt_Pc = W_x.T @ Pc
    cdef np.ndarray[np.float32_t, ndim=2] beta_vec = np.linalg.inv(W_xt_Pc @ W_x) @ (W_xt_Pc @ Y)
    beta = float(beta_vec[c-1])

    cdef np.float32_t ytPxy = Y.T @ Px @ Y

    se_beta = np.sqrt(ytPxy/((n - c - 1) * (x.T @ Pc @ x)))

    tau = (n-c-1)/ytPxy

    return np.float32(beta), np.float32(se_beta), np.float32(tau)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def likelihood_lambda(np.float32_t lam,
                      np.ndarray[np.float32_t, ndim=1] eigenVals, 
                      np.ndarray[np.float32_t, ndim=2] U, 
                      np.ndarray[np.float32_t, ndim=2] Y, 
                      np.ndarray[np.float32_t, ndim=2] W):
    n = Y.shape[0]

    cdef np.float32_t result = (n/2)*np.log(n/(2*np.pi))

    result = result - n/2

    result = result - 0.5*np.sum(np.log(lam*eigenVals + 1.0))

    result = result - (n/2)*np.log(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)

    return np.float32(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def likelihood_derivative1_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                  np.ndarray[np.float32_t, ndim=2] U, 
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W):
    n = Y.shape[0]

    cdef np.float32_t result = -0.5*((n-np.sum(1/(lam*eigenVals + 1.0)))/lam)

    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W, lam)
    cdef np.ndarray[np.float32_t, ndim=2] yt_Px = Y.T @ Px

    cdef np.float32_t yT_Px_y = yt_Px @ Y

    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yt_Px @ yt_Px.T)/lam

    result = result - (n/2)*yT_Px_G_Px_y/yT_Px_y

    return np.float32(result)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def likelihood_derivative2_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                  np.ndarray[np.float32_t, ndim=2] U, 
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W): 
    n = Y.shape[0]

    cdef np.float32_t result = 0.5*(n + np.sum(np.power(lam*eigenVals + 1.0, -2)) + 2*np.sum(np.power(lam*eigenVals + 1.0,-1)))

    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W, lam)
    cdef np.ndarray[np.float32_t, ndim=2] yt_Px = Y.T @ Px

    cdef np.float32_t yT_Px_y = yt_Px @ Y

    cdef np.float32_t yT_Px_Px_y = yt_Px @ yt_Px.T

    cdef np.float32_t yT_Px_G_Px_G_Px_y = (yT_Px_y + yt_Px @ Px @ yt_Px.T - 2*yT_Px_Px_y)/(lam*lam)

    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

    result = result - n * (yT_Px_G_Px_G_Px_y * yT_Px_y - yT_Px_G_Px_y * yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)

    return np.float32(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def likelihood_derivative1_restricted_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                  np.ndarray[np.float32_t, ndim=2] U, 
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W):
    n = W.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W, lam)
    cdef np.ndarray[np.float32_t, ndim=2] yt_Px = Y.T @ Px

    cdef np.float32_t result = -0.5*((n-c - np.trace(Px))/lam)

    cdef np.float32_t yT_Px_y = yt_Px @ Y

    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yt_Px @ yt_Px.T)/lam

    result = result - 0.5*(n - c)*yT_Px_G_Px_y/yT_Px_y

    return np.float32(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def likelihood_derivative2_restricted_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                  np.ndarray[np.float32_t, ndim=2] U, 
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W): 
    n = Y.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W, lam)
    cdef np.ndarray[np.float32_t, ndim=2] yt_Px = Y.T @ Px

    cdef np.float32_t yT_Px_y = yt_Px @ Y

    cdef np.float32_t yT_Px_Px_y = yt_Px @ yt_Px.T

    cdef np.float32_t yT_Px_G_Px_G_Px_y = (yT_Px_y + yt_Px @ Px @ yt_Px.T - 2*yT_Px_Px_y)/(lam*lam)

    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

    cdef np.float32_t result = 0.5*(n - c + np.trace(Px @ Px) - 2*np.trace(Px))/(lam*lam)

    result = result - (n - c) * (yT_Px_G_Px_G_Px_y * yT_Px_y - yT_Px_G_Px_y * yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)

    return np.float32(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def likelihood(np.float32_t lam, 
               np.float32_t tau, 
               np.ndarray[np.float32_t, ndim=2] beta, 
               np.ndarray[np.float32_t, ndim=1] eigenVals, 
               np.ndarray[np.float32_t, ndim=2] U, 
               np.ndarray[np.float32_t, ndim=2] Y, 
               np.ndarray[np.float32_t, ndim=2] W):
    n = W.shape[0]

    cdef np.ndarray[np.float32_t, ndim=2] H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T

    cdef np.float32_t result = (n/2)*np.log(tau) 
    result = result - (n/2)*np.log(2*np.pi)
    
    result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

    cdef np.ndarray[np.float32_t, ndim=2] y_W_beta = Y - W @ beta

    result = result - 0.5 * tau * y_W_beta.T @ H_inv @ y_W_beta


    return np.float32(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def likelihood_restricted(np.float32_t lam, 
               np.float32_t tau,
               np.ndarray[np.float32_t, ndim=1] eigenVals, 
               np.ndarray[np.float32_t, ndim=2] U, 
               np.ndarray[np.float32_t, ndim=2] Y, 
               np.ndarray[np.float32_t, ndim=2] W):
    n = W.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T

    cdef np.float32_t result = 0.5*(n - c)*np.log(tau)
    result = result - 0.5*(n - c)*np.log(2*np.pi)
    _, logdet = np.linalg.slogdet(W.T @ W)
    result = result + 0.5*logdet

    result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

    #result = result - 0.5*np.log(np.linalg.det(W_x.T @ H_inv @ W_x)) # Causing NAN
    _, logdet = np.linalg.slogdet(W.T @ H_inv @ W)
    result = result - 0.5*logdet # Causing NAN

    result = result - 0.5*(n - c)*np.log(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)

    return np.float32(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def likelihood_restricted_lambda(np.float32_t lam, 
                                 np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                 np.ndarray[np.float32_t, ndim=2] U, 
                                 np.ndarray[np.float32_t, ndim=2] Y, 
                                 np.ndarray[np.float32_t, ndim=2] W):
    n = W.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] H_inv = U @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U.T

    cdef np.float32_t result = 0.5*(n - c)*np.log(0.5*(n - c)/np.pi)
    result = result - 0.5*(n - c)
    _, logdet = np.linalg.slogdet(W.T @ W)
    result = result + 0.5*logdet
    
    result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

    #result = result - 0.5*np.log(np.linalg.det(W_x.T @ H_inv @ W_x)) # Causing NAN
    _, logdet = np.linalg.slogdet(W.T @ H_inv @ W)
    result = result - 0.5*logdet # Causing NAN

    result = result - 0.5*(n - c)*np.log(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)

    return np.float32(result)







