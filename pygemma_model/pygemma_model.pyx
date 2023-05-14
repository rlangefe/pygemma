# cython: infer_types=True
import cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, log

from scipy import optimize, stats
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas

from numpy cimport float_t, ndarray

#from scipy.optimize.cython_optimize cimport brentq

import time

MIN_VAL=1e-20

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
        print(f"[red]Failed {function.__name__} - {diff} s")
        return result
    else:
        print(f"[green]Passed {function.__name__} - {diff} s")
        return result
    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef calc_lambda_restricted(np.ndarray[np.float32_t, ndim=1] eigenVals,  
                             np.ndarray[np.float32_t, ndim=2] Y, 
                             np.ndarray[np.float32_t, ndim=2] W):
    # Loop over intervals and find where likelihood changes signs with respect to lambda
    cdef np.float32_t step = 1.0

    cdef np.float32_t lambda_pow_low = -5.0
    cdef np.float32_t lambda_pow_high = 5.0

    roots = [10.0 ** lambda_pow_low, 10.0 ** lambda_pow_high]
    likelihood_list = [likelihood_restricted_lambda(roots[0], eigenVals, Y, W), likelihood_restricted_lambda(roots[1], eigenVals, Y, W)]

    cdef float lambda0, lambda1, lambda_min, likelihood_lambda0, likelihood_lambda1 = 0.0

    cdef int maxiter = 100
    
    for lambda_idx in np.arange(lambda_pow_low,lambda_pow_high,step):
        lambda0 = 10.0 ** (lambda_idx) #np.power(10.0, lambda_idx)
        lambda1 = 10.0 ** (lambda_idx + step) #np.power(10.0, lambda_idx+step)
        
        likelihood_lambda0 = likelihood_derivative1_restricted_lambda(lambda0, eigenVals, Y, W)
        likelihood_lambda1 = likelihood_derivative1_restricted_lambda(lambda1, eigenVals, Y, W)



        if np.sign(likelihood_lambda0) * np.sign(likelihood_lambda1) < 0:
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


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef logdet(float[:,::1] mat):
    cdef int n = mat.shape[0]
    cdef int i, j, k, sign = 0
    cdef float det = 1.0
    cdef int[:] ipiv = np.zeros(n, dtype=np.int32)
    cdef float[:,:] lu = np.zeros((n, n), dtype=np.float32)
    cdef float* data = &mat[0, 0]
    cdef float* lu_data = &lu[0, 0]

    for i in range(n):
        ipiv[i] = i

    for i in range(n):
        max_val = 0.0
        max_idx = i

        for j in range(i, n):
            if fabs(data[j * n + i]) > max_val:
                max_val = fabs(data[j * n + i])
                max_idx = j

        if max_val == 0.0:
            return -np.inf

        if max_idx != i:
            sign += 1
            ipiv[i], ipiv[max_idx] = ipiv[max_idx], ipiv[i]
            for j in range(n):
                lu_data[i * n + j], lu_data[max_idx * n + j] = lu_data[max_idx * n + j], lu_data[i * n + j]
            det = -det

        for j in range(i + 1, n):
            lu_data[j * n + i] = data[j * n + i] / data[i * n + i]
            for k in range(i + 1, n):
                data[j * n + k] -= lu_data[j * n + i] * data[i * n + k]

    for i in range(n):
        det *= lu_data[i * n + i]

    if sign % 2 == 1:
        det = -det

    return log(fabs(det))


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef float newton(float lam,
                   np.ndarray[np.float32_t, ndim=1] eigenVals, 
                   np.ndarray[np.float32_t, ndim=2] Y, 
                   np.ndarray[np.float32_t, ndim=2] W):

    cdef float lambda_min = lam
    cdef int iter = 0
    cdef float r_eps = 0.0
    cdef float lambda_new, ratio, d1, d2

    while True:
        d1 = likelihood_derivative1_restricted_lambda(lambda_min, eigenVals, Y, W)
        d2 = likelihood_derivative2_restricted_lambda(lambda_min, eigenVals, Y, W)

        with cython.nogil:
            ratio = d1/d2

        if ratio * d1 * d2 <= 0.0:
            break

        lambda_new = lambda_min - ratio
        r_eps = fabs(lambda_new - lambda_min) / fabs(lambda_min)

        if lambda_new < 0.0 or np.isnan(lambda_new) or np.isinf(lambda_new):
            break

        lambda_min = lambda_new

        if r_eps < 1e-5 or iter > 100:
            break

        iter += 1

    return lambda_min

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
@cython.cdivision(True)
def compute_Pc(
                np.ndarray[np.float32_t, ndim=1] eigenVals, 
                np.ndarray[np.float32_t, ndim=2] U, 
                np.ndarray[np.float32_t, ndim=2] W, 
                np.float32_t lam
                ):
    
    cdef np.ndarray[np.float32_t, ndim=2] H_inv = compute_H_inv(lam, eigenVals, U)

    cdef np.ndarray[np.float32_t, ndim=2] H_inv_W = H_inv @ W

    return H_inv - H_inv_W @ np.linalg.inv(W.T @ H_inv_W) @ H_inv_W.T

##################
# Reimplementing #
##################

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
@cython.cdivision(True)
cdef compute_Pc_cython(
                np.ndarray[np.float32_t, ndim=1] eigenVals, 
                np.ndarray[np.float32_t, ndim=2] U, 
                np.ndarray[np.float32_t, ndim=2] W, 
                np.float32_t lam
                ):
    
    cdef np.ndarray[np.float32_t, ndim=2] H_inv = compute_H_inv(lam, eigenVals, U)

    cdef np.ndarray[np.float32_t, ndim=2] H_inv_W = H_inv @ W
    
    return H_inv - H_inv_W @ np.linalg.inv(W.T @ H_inv_W) @ W.T @ H_inv #H_inv_W.T

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef calc_beta_vg_ve(np.ndarray[np.float32_t, ndim=1] eigenVals,
                    np.ndarray[np.float32_t, ndim=2] W, 
                    np.ndarray[np.float32_t, ndim=1] x, 
                    np.float32_t lam, 
                    np.ndarray[np.float32_t, ndim=2] Y):
    cdef np.ndarray[np.float32_t, ndim=2] W_x = np.c_[W,x]
    
    cdef int n, c
    n = W.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0
    
    cdef np.ndarray[np.float32_t, ndim=2] W_x_t_H_inv = ((1.0/mod_eig)[:,np.newaxis] * W_x).T

    cdef np.ndarray[np.float32_t, ndim=2] beta_vec = np.linalg.inv(W_x_t_H_inv @ W_x) @ (W_x_t_H_inv @ Y)

    cdef np.float32_t beta = beta_vec[c,0]

    cdef np.float32_t ytPxy = max(compute_at_Pi_b(lam, c+1, mod_eig, W_x, Y, Y), MIN_VAL)

    # NOTE: x.T @ Px @ x is negative. We should fix this, but for now, we're throwing it away bc
    # it's not needed for anything
    #cdef np.float32_t se_beta = (1/np.sqrt((n - c - 1))) * np.sqrt(ytPxy)/np.sqrt(x.T @ Pc @ x)


    cdef np.float32_t tau = n/ytPxy

    return beta, beta_vec, None, tau

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef calc_beta_vg_ve_restricted(np.ndarray[np.float32_t, ndim=1] eigenVals,
                    np.ndarray[np.float32_t, ndim=2] W, 
                    np.ndarray[np.float32_t, ndim=2] x, 
                    np.float32_t lam, 
                    np.ndarray[np.float32_t, ndim=2] Y):
    cdef np.ndarray[np.float32_t, ndim=2] W_x = np.c_[W,x]

    cdef int n, c
    n = W.shape[0]
    c = W.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0

    cdef np.ndarray[np.float32_t, ndim=2] W_x_t_H_inv = ((1.0/mod_eig)[:,np.newaxis] * W_x).T
    
    cdef np.ndarray[np.float32_t, ndim=2] beta_vec = np.linalg.inv(W_x_t_H_inv @ W_x) @ (W_x_t_H_inv @ Y)
    
    cdef np.float32_t beta = beta_vec[c,0] #compute_at_Pi_b(lam, c, mod_eig, W, x, Y) / max(compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL) # Double check this property holds, but I think it does bc positive symmetric

    cdef np.float32_t ytPxy = max(compute_at_Pi_b(lam, c+1, mod_eig, W_x, Y, Y), MIN_VAL)

    cdef np.float32_t se_beta = np.sqrt(ytPxy) / (np.sqrt(max(compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL)) * np.sqrt((n - c - 1)))

    cdef np.float32_t tau = (n-c-1)/ytPxy

    return np.float32(beta), beta_vec, np.float32(se_beta), np.float32(tau)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_lambda(np.float32_t lam,
                      np.ndarray[np.float32_t, ndim=1] eigenVals,
                      np.ndarray[np.float32_t, ndim=2] Y, 
                      np.ndarray[np.float32_t, ndim=2] W):
    cdef int n, c
    n = W.shape[0]
    c = W.shape[1]
    cdef np.float32_t result
    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0

    result = (n/2)*log(n/(2*np.pi))

    result = result - n/2

    result = result - 0.5*np.sum(np.log(mod_eig))

    result = result - (n/2)*log(max(compute_at_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL))

    return np.float32(result)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative1_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals,
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W):
    cdef int n, c
    n = W.shape[0]
    c = W.shape[1]
    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0
    cdef np.float32_t result = -0.5*((n-np.sum(1.0/(mod_eig)))/lam)
    cdef np.float32_t num, denom
    num = max(compute_at_Pi_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL)
    denom = max(compute_at_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL)

    result = result + (n/2)*(1.0 - num / denom)/lam

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative2_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals,
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W):
    cdef int n,c
    n = W.shape[0]
    c = W.shape[1]
    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0
    cdef np.float32_t yT_Px_y = max(compute_at_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL)
    cdef np.float32_t yT_Px_Px_y = max(compute_at_Pi_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL)
    
    cdef np.float32_t yT_Px_G_Px_G_Px_y = (yT_Px_y + max(compute_at_Pi_Pi_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL) - 2*yT_Px_Px_y)/(lam*lam)
    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

    cdef np.float32_t result = 0.5*(n + np.sum(np.power(mod_eig, -2.0)) - 2*np.sum(np.power(mod_eig,-1.0)))/np.power(lam,2)
    result = result - 0.5 * n * (2 * yT_Px_G_Px_G_Px_y - yT_Px_G_Px_y * yT_Px_G_Px_y / yT_Px_y) / (yT_Px_y)

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative1_restricted_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals,
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W):
    cdef int n, c
    n = W.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0

    cdef np.float32_t yT_Px_y = max(compute_at_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL)
    cdef np.float32_t yT_Px_Px_y = max(compute_at_Pi_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL)

    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam
    cdef np.float32_t result = -0.5*((n - c - trace_Pi(lam, c, mod_eig, W))/lam) # -0.5*tr(Px @ G)

    result = result + 0.5*(n - c)*yT_Px_G_Px_y/yT_Px_y # 0.5 * Y.T @ Px @ G @ Px @ Y / (Y.T @ Px @ Y)

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative2_restricted_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals,
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W):
    cdef int n,c
    n = Y.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0

    cdef np.float32_t yT_Px_y = max(compute_at_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL)

    cdef np.float32_t yT_Px_Px_y = max(compute_at_Pi_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL)

    cdef np.float32_t yT_Px_Px_Px_y = max(compute_at_Pi_Pi_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL)

    cdef np.float32_t yT_Px_G_Px_G_Px_y = (yT_Px_y + yT_Px_Px_Px_y - 2*yT_Px_Px_y)/(lam*lam)

    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam
    
    cdef np.float32_t result = 0.5*(n - c + trace_Pi_Pi(lam, c, mod_eig, W) - 2*trace_Pi(lam, c, mod_eig, W))/(lam*lam)

    result = result - (n - c) * ((yT_Px_G_Px_G_Px_y * yT_Px_y) - 0.5 * yT_Px_G_Px_y*yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)
    
    return np.float32(result)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood(np.float32_t lam, 
               np.float32_t tau, 
               np.ndarray[np.float32_t, ndim=2] beta, 
               np.ndarray[np.float32_t, ndim=1] eigenVals,
               np.ndarray[np.float32_t, ndim=2] Y, 
               np.ndarray[np.float32_t, ndim=2] W):
    cdef int n = W.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0
    cdef np.float32_t result = 0.5 * n *log(tau) 
    result = result - 0.5*n*log(2*np.pi)
    
    result = result - 0.5 * np.sum(np.log(mod_eig))

    cdef np.ndarray[np.float32_t, ndim=2] y_W_beta = Y - W @ beta

    result = result - 0.5 * tau * y_W_beta.T @ ((1.0/mod_eig[:, np.newaxis]) * y_W_beta)


    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_restricted(np.float32_t lam, 
               np.float32_t tau,
               np.ndarray[np.float32_t, ndim=1] eigenVals, 
               np.ndarray[np.float32_t, ndim=2] Y, 
               np.ndarray[np.float32_t, ndim=2] W):
    cdef int n,c
    n = W.shape[0]
    c = W.shape[1]
    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0
    cdef np.ndarray[np.float32_t, ndim=2] y_eig = (1.0/mod_eig[:, np.newaxis] * Y)
    cdef np.ndarray[np.float32_t, ndim=2] Wt_eig_W = W.T @ ((1.0/mod_eig[:, np.newaxis]) * W)

    cdef np.float32_t result = 0.5*(n - c)*log(tau)
    result = result - 0.5*(n - c)*log(2*np.pi)

    result = result + 0.5*np.linalg.slogdet(W.T @ W)[1]

    result = result - 0.5 * np.sum(np.log(mod_eig))
    
    result = result - 0.5*np.linalg.slogdet(Wt_eig_W)[1]

    result = result - 0.5*tau*(max(compute_at_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL))

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_restricted_lambda(float lam, 
                                 np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                 np.ndarray[np.float32_t, ndim=2] Y, 
                                 np.ndarray[np.float32_t, ndim=2] W):
    cdef int n,c
    n = W.shape[0]
    c = W.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0
    cdef np.ndarray[np.float32_t, ndim=2] Wt_eig_W = W.T @ (W / mod_eig[:, np.newaxis])

    cdef np.float32_t result = 0.5*(n - c)*log(0.5*(n - c)/np.pi)
    result = result - 0.5*(n - c)
    result = result + 0.5*np.linalg.slogdet(W.T @ W)[1]
    result = result - 0.5 * np.sum(np.log(mod_eig))

    result = result - 0.5*(n - c)*log(max(compute_at_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL))

    result = result - 0.5*np.linalg.slogdet(Wt_eig_W)[1]

    return np.float32(result) 

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef compute_H_inv(np.float32_t lam,
                   np.ndarray[np.float32_t, ndim=1] eigenVals,
                   np.ndarray[np.float32_t, ndim=2] U):            

    return U.T @ (1.0/(lam*eigenVals + 1.0)[:, np.newaxis] * U)

# GEMMA lmm.c 1093
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef trace_Pi(float lam,
              int i,
              float[:] eigenVals,
              float[:,:] W):
    
    cdef int j
    cdef float result = 0.0
    cdef float[:,:] W_i
    cdef float num, denom

    while i > 0:
        if i == 1:
            with nogil:
                num = 0.0
                denom = 0.0
                for j in range(eigenVals.shape[0]):
                    num += W[j, 0] * W[j, 0] / (eigenVals[j] **2)
                    denom += W[j, 0] * W[j, 0] / eigenVals[j]

                result -= num / denom
        else:
            W_i = W[:, (i-1):i]
            result -= compute_at_Pi_Pi_b(lam, i-1, eigenVals, W[:,0:(i-1)], W_i, W_i) / compute_at_Pi_b(lam, i-1, eigenVals, W[:,0:(i-1)], W_i, W_i) 
        i -= 1
    
    with nogil:
        for j in range(eigenVals.shape[0]):
            result += 1.0/eigenVals[j]
    
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef trace_Pi_recursive(np.float32_t lam,
              int i,
              np.ndarray[np.float32_t, ndim=1] eigenVals,
              np.ndarray[np.float32_t, ndim=2] W):

    if i == 0:
        # Return tr(Pi_0) = tr(H^-1) = sum(1/(lam*eigenVals + 1))
        return np.sum(np.power(eigenVals, -1.0))

    else:
        return trace_Pi(lam, i-1, eigenVals, W) - compute_at_Pi_Pi_b(lam, i-1, eigenVals, W[:,0:(i-1)], W[:, (i-1):i], W[:, (i-1):i]) / compute_at_Pi_b(lam, i-1, eigenVals, W[:,0:(i-1)], W[:, (i-1):i], W[:, (i-1):i])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float trace_Pi_Pi(float lam,
              int i,
              float[:] eigenVals,
              float[:,:] W):
    
    cdef int j
    cdef float result = 0.0
    cdef float[:,:] W_i, W_0_im1
    cdef float wi_Pim1_wi, wi_Pim1_Pim1_wi, wi_Pim1_Pim1_Pim1_wi

    while i > 0:
        if i == 1:
            with nogil:
                wi_Pim1_wi = 0.0
                wi_Pim1_Pim1_wi = 0.0
                wi_Pim1_Pim1_Pim1_wi = 0.0

                for j in range(eigenVals.shape[0]):
                    wi_Pim1_wi += W[j, 0] * W[j, 0] / eigenVals[j]
                    wi_Pim1_Pim1_wi += W[j, 0] * W[j, 0] / (eigenVals[j] ** 2.0)
                    wi_Pim1_Pim1_Pim1_wi += W[j, 0] * W[j, 0] / (eigenVals[j] ** 3.0)

                result += ((wi_Pim1_Pim1_wi / wi_Pim1_wi) ** 2.0) 
                result -= 2.0 * wi_Pim1_Pim1_Pim1_wi / wi_Pim1_wi
        else:
            W_i = W[:, (i-1):i]
            W_0_im1 = W[:,0:(i-1)]
            wi_Pim1_wi = compute_at_Pi_b(lam, i-1, eigenVals, W_0_im1, W_i, W_i)

            result += ((compute_at_Pi_Pi_b(lam, i-1, eigenVals, W_0_im1, W_i, W_i) / wi_Pim1_wi) ** 2.0)
            result -= 2.0 * compute_at_Pi_Pi_Pi_b(lam, i-1, eigenVals, W_0_im1, W_i, W_i) / wi_Pim1_wi
            
        i -= 1

    with nogil:    
        for j in range(eigenVals.shape[0]):
            result += 1.0/(eigenVals[j]**2.0)

    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float trace_Pi_Pi_recursive(np.float32_t lam,
                int i,
                np.ndarray[np.float32_t, ndim=1] eigenVals,
                np.ndarray[np.float32_t, ndim=2] W):

    cdef np.float32_t wi_Pim1_wi

    if i == 0:
        # Return tr(Pi_0) = tr(H^-1) = sum(1/(lam*eigenVals + 1))
        return np.sum(np.power(eigenVals, -2.0))

    else:
        wi_Pim1_wi = compute_at_Pi_b(lam, i-1, eigenVals, W, W[:, (i-1):i], W[:, (i-1):i])

        return trace_Pi_Pi(lam, i-1, eigenVals, W) \
            + np.power(compute_at_Pi_Pi_b(lam, i-1, eigenVals, W[:,0:(i-1)], W[:, (i-1):i], W[:, (i-1):i]) / wi_Pim1_wi, 2.0) \
            - 2.0 * compute_at_Pi_Pi_Pi_b(lam, i-1, eigenVals, W[:,0:(i-1)], W[:, (i-1):i], W[:, (i-1):i]) / wi_Pim1_wi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef mat_vec_mult(float[:, :] A, float[:, :] b):
    cdef int i, j, k
    cdef float[:, :] result = np.zeros((A.shape[0], b.shape[1]), dtype=np.float32)

    with nogil:
        for i in range(A.shape[0]):
            for j in range(b.shape[1]):
                for k in range(A.shape[1]):
                    result[i, j] += A[i, k] * b[k, j]

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef invert_matrix(float [:,:] A):
    cdef int n = A.shape[0]
    cdef int i, j, k
    cdef float [:,:] B = np.zeros((n,n), dtype=np.float32)
    cdef float [:,:] C = np.zeros((n,n), dtype=np.float32)
    cdef float temp = 0.0

    with nogil:
        # Copy A into B and initialize C to the identity matrix
        for i in range(n):
            for j in range(n):
                B[i,j] = A[i,j]
                C[i,j] = 0.0
            C[i,i] = 1.0

        # Perform Gaussian elimination with partial pivoting
        for i in range(n):
            # Find the pivot row
            max_idx = i
            for j in range(i+1, n):
                if fabs(B[j,i]) > fabs(B[max_idx,i]):
                    max_idx = j

            # Swap the pivot row with the current row
            if max_idx != i:
                for k in range(n):
                    temp = B[i,k]
                    B[i,k] = B[max_idx,k]
                    B[max_idx,k] = temp
                    temp = C[i,k]
                    C[i,k] = C[max_idx,k]
                    C[max_idx,k] = temp

            # Divide the current row by the pivot element
            temp = B[i,i]
            for k in range(n):
                B[i,k] /= temp
                C[i,k] /= temp

            # Subtract the pivot row from all other rows
            for j in range(n):
                if j != i:
                    temp = B[j,i]
                    for k in range(n):
                        B[j,k] -= temp * B[i,k]
                        C[j,k] -= temp * C[i,k]

    return C


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef mat_mat_mult(float[:, :] A, float[:, :] B):
    cdef int i, j, k
    cdef float[:, :] result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float compute_at_Pi_b(float lam,
                     int i,
                      float[:] eigenVals,
                      float[:, :] W,
                      float[:, :] a,
                      float[:, :] b):

    cdef int j,k
    cdef float result, result2
    cdef float[:, :] inv, temp_vec
    result = 0.0
    result2 = 0.0

    for j in range(a.shape[0]):
        result += a[j, 0] * b[j, 0] / eigenVals[j]

    inv = W.copy()
    for j in range(W.shape[0]):
        for k in range(W.shape[1]):
            inv[j, k] /= eigenVals[j]

    inv = mat_mat_mult(W.T, inv)
    inv = invert_matrix(inv)

    temp_vec = b.copy()
    for j in range(b.shape[0]):
        temp_vec[j, 0] /= eigenVals[j]

    temp_vec = mat_vec_mult(W.T, temp_vec)
    temp_vec = mat_vec_mult(inv, temp_vec)
    temp_vec = mat_vec_mult(W, temp_vec)

    for j in range(a.shape[0]):
        result2 += a[j, 0] * temp_vec[j, 0] / eigenVals[j]

    #if fabs((float(np.asarray(a.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(b)) - np.asarray(a.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(W)) @ np.linalg.inv(np.asarray(W.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(W))) @ np.asarray(W.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(b)))) - (result - result2)) > 1e-3:
        #print(lam, float(np.asarray(a.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(b)) - np.asarray(a.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(W)) @ np.linalg.inv(np.asarray(W.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(W))) @ np.asarray(W.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(b))), result - result2)

    # if (result - result2) < 0:
    #     print(lam, result - result2, result, result2, 
    #           float(np.asarray(a.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(b))), 
    #           float(np.asarray(a.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(W)) @ np.linalg.inv(np.asarray(W.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(W))) @ np.asarray(W.T) @ (1.0/np.asarray(eigenVals)[:,np.newaxis] * np.asarray(b))))

    return result - result2



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float compute_at_Pi_b_recursive(float lam,
                      int i,
                      float[:] eigenVals,
                      float[:, :] W,
                      float[:, :] a,
                      float[:, :] b):

    cdef int j
    cdef float[:, :] W_i
    cdef float result = 0.0

    if i == 0:
        for j in range(a.shape[0]):
            result += a[j, 0] * b[j, 0] / eigenVals[j]
        return result
    else:
        W_i = W[:, (i-1):i]
        return compute_at_Pi_b(lam, i-1, eigenVals, W, a, b) - compute_at_Pi_b(lam, i-1, eigenVals, W, b, W_i) * compute_at_Pi_b(lam, i-1, eigenVals, W, a, W_i)/compute_at_Pi_b(lam, i-1, eigenVals, W, W_i, W_i) 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float compute_at_Pi_Pi_b(float lam,
                              int i,
                              float[:] eigenVals,
                              float[:,:] W,
                              float[:,:] a,
                              float[:,:] b):
    cdef int j,k
    cdef float result = 0.0
    cdef float[:, :] inv, temp_vec, temp_vec2

    inv = W.copy()
    for j in range(W.shape[0]):
        for k in range(W.shape[1]):
            inv[j, k] /= eigenVals[j]

    inv = mat_mat_mult(W.T, inv)
    inv = invert_matrix(inv)

    temp_vec = b.copy()
    temp_vec2 = a.copy()
    for j in range(b.shape[0]):
        temp_vec[j, 0] /= eigenVals[j]
        temp_vec2[j, 0] /= eigenVals[j]

    temp_vec = mat_vec_mult(W.T, temp_vec)
    temp_vec = mat_vec_mult(inv, temp_vec)
    temp_vec = mat_vec_mult(W, temp_vec)

    temp_vec2 = mat_vec_mult(W.T, temp_vec2)
    temp_vec2 = mat_vec_mult(inv, temp_vec2)
    temp_vec2 = mat_vec_mult(W, temp_vec2)

    for j in range(temp_vec.shape[0]):
        temp_vec[j, 0] = (b[j, 0] - temp_vec[j, 0]) / eigenVals[j]
        temp_vec2[j, 0] = (a[j, 0] - temp_vec2[j, 0]) / eigenVals[j]
        result += temp_vec[j, 0] * temp_vec2[j, 0]

    return result #max(result, MIN_VAL)


    #return compute_at_Pi_b(lam, i, eigenVals, W, a, temp_vec)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef compute_at_Pi_Pi_b_recursive(np.float32_t lam,
                        int i,
                        np.ndarray[np.float32_t, ndim=1] eigenVals,
                        np.ndarray[np.float32_t, ndim=2] W,
                        np.ndarray[np.float32_t, ndim=2] a,
                        np.ndarray[np.float32_t, ndim=2] b):
    
    cdef np.float32_t at_Pim1_wi, bt_Pim1_wi, wi_Pi_wi


    if i == 0:
        return a.T @ ((np.power(eigenVals, 2.0)[:, np.newaxis]) * b)
    else:
        at_Pim1_wi = compute_at_Pi_b(lam, i-1, eigenVals, W, a, W[:, (i-1):i])
        bt_Pim1_wi = compute_at_Pi_b(lam, i-1, eigenVals, W, b, W[:, (i-1):i])
        wi_Pi_wi = compute_at_Pi_b(lam, i-1, eigenVals, W, W[:, (i-1):i], W[:, (i-1):i])

        return compute_at_Pi_Pi_b_recursive(lam, i-1, eigenVals, W, a, b) \
            + at_Pim1_wi*bt_Pim1_wi*compute_at_Pi_Pi_b_recursive(lam, i-1, eigenVals, W, W[:, (i-1):i], W[:, (i-1):i])/compute_at_Pi_Pi_b_recursive(lam, i-1, eigenVals, W, W[:, (i-1):i], W[:, (i-1):i]) \
            - at_Pim1_wi * compute_at_Pi_Pi_b_recursive(lam, i-1, eigenVals, W, b, W[:, (i-1):i])/wi_Pi_wi \
            - bt_Pim1_wi * compute_at_Pi_Pi_b_recursive(lam, i-1, eigenVals, W, a, W[:, (i-1):i])/wi_Pi_wi

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef float compute_at_Pi_Pi_Pi_b(float lam,
                        int i,
                        float[:] eigenVals,
                        float[:,:] W,
                        float[:,:] a,
                        float[:,:] b):

    cdef int j,k
    cdef float result, result2 = 0.0
    cdef float[:, :] inv, temp_vec, temp_vec2, temp_vec3

    inv = W.copy()
    for j in range(W.shape[0]):
        for k in range(W.shape[1]):
            inv[j, k] /= eigenVals[j]

    inv = mat_mat_mult(W.T, inv)
    inv = invert_matrix(inv)

    temp_vec = b.copy()
    temp_vec2 = a.copy()
    for j in range(b.shape[0]):
        temp_vec[j, 0] /= eigenVals[j]
        temp_vec2[j, 0] /= eigenVals[j]

    temp_vec = mat_vec_mult(W.T, temp_vec)
    temp_vec = mat_vec_mult(inv, temp_vec)
    temp_vec = mat_vec_mult(W, temp_vec)

    temp_vec2 = mat_vec_mult(W.T, temp_vec2)
    temp_vec2 = mat_vec_mult(inv, temp_vec2)
    temp_vec2 = mat_vec_mult(W, temp_vec2)

    for j in range(temp_vec.shape[0]):
        temp_vec[j, 0] = (b[j, 0] - temp_vec[j, 0]) / eigenVals[j]
        temp_vec2[j, 0] = (a[j, 0] - temp_vec2[j, 0]) / eigenVals[j]

    return compute_at_Pi_b(lam, i, eigenVals, W, temp_vec2, temp_vec)

    # for j in range(a.shape[0]):
    #     result += temp_vec2[j, 0] * (temp_vec[j, 0] / eigenVals[j])

    # temp_vec3 = temp_vec2.copy()
    # for j in range(temp_vec2.shape[0]):
    #     temp_vec3[j, 0] /= eigenVals[j]

    # temp_vec3 = mat_vec_mult(W.T, temp_vec3)
    # temp_vec3 = mat_vec_mult(inv, temp_vec3)
    # temp_vec3 = mat_vec_mult(W, temp_vec3)

    # for j in range(temp_vec3.shape[0]):
    #     result2 += temp_vec2[j, 0] * (temp_vec3[j, 0] / eigenVals[j])

    # return result - result2

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float compute_at_Pi_Pi_Pi_b_recursive(np.float32_t lam,
                        int i,
                        np.ndarray[np.float32_t, ndim=1] eigenVals,
                        np.ndarray[np.float32_t, ndim=2] W,
                        np.ndarray[np.float32_t, ndim=2] a,
                        np.ndarray[np.float32_t, ndim=2] b):

    cdef np.float32_t at_Pim1_wi, bt_Pim1_wi, wi_Pim1_wi, wi_Pim1_Pim1_wi, at_Pim1_Pim1_wi, bt_Pim1_Pim1_wi
    
    if i == 0:
        return a.T @ ((np.power(1.0/(lam*eigenVals + 1.0), 3.0)[:, np.newaxis]) * b)
    else:
        at_Pim1_wi = compute_at_Pi_b(lam, i-1, eigenVals, W, a, W[:, (i-1):i])
        bt_Pim1_wi = compute_at_Pi_b(lam, i-1, eigenVals, W, b, W[:, (i-1):i])
        wi_Pim1_wi = compute_at_Pi_b(lam, i-1, eigenVals, W, W[:, (i-1):i], W[:, (i-1):i])
        wi_Pim1_Pim1_wi = compute_at_Pi_Pi_b(lam, i-1, eigenVals, W, W[:, (i-1):i], W[:, (i-1):i])
        at_Pim1_Pim1_wi = compute_at_Pi_Pi_b(lam, i-1, eigenVals, W, a, W[:, (i-1):i])
        bt_Pim1_Pim1_wi = compute_at_Pi_Pi_b(lam, i-1, eigenVals, W, b, W[:, (i-1):i])

        return compute_at_Pi_Pi_Pi_b(lam, i-1, eigenVals, W, a, b) \
            - at_Pim1_wi * bt_Pim1_wi * wi_Pim1_Pim1_wi * wi_Pim1_Pim1_wi / (wi_Pim1_wi * wi_Pim1_wi * wi_Pim1_wi) \
            - at_Pim1_wi * compute_at_Pi_Pi_Pi_b(lam, i-1, eigenVals, W, b, W[:, (i-1):i]) / wi_Pim1_wi \
            - bt_Pim1_wi * compute_at_Pi_Pi_Pi_b(lam, i-1, eigenVals, W, a, W[:, (i-1):i]) / wi_Pim1_wi \
            - at_Pim1_Pim1_wi * bt_Pim1_Pim1_wi / wi_Pim1_wi \
            + at_Pim1_wi * bt_Pim1_Pim1_wi * wi_Pim1_Pim1_wi / (wi_Pim1_wi * wi_Pim1_wi) \
            + bt_Pim1_wi * at_Pim1_Pim1_wi * wi_Pim1_Pim1_wi / (wi_Pim1_wi * wi_Pim1_wi) \
            + at_Pim1_wi * bt_Pim1_wi * compute_at_Pi_Pi_Pi_b(lam, i-1, eigenVals, W, W[:, (i-1):i], W[:, (i-1):i]) / (wi_Pim1_wi * wi_Pim1_wi)