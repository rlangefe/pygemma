# cython: infer_types=True
import cython
import numpy as np
cimport numpy as np

from scipy import optimize, stats

#from scipy.optimize.cython_optimize cimport brentq

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef calc_lambda_restricted(np.ndarray[np.float32_t, ndim=1] eigenVals, 
                             np.ndarray[np.float32_t, ndim=2] U, 
                             np.ndarray[np.float32_t, ndim=2] Y, 
                             np.ndarray[np.float32_t, ndim=2] W):
    # Loop over intervals and find where likelihood changes signs with respect to lambda
    cdef np.float32_t step = 1.0

    cdef np.float32_t lambda_pow_low = -5.0
    cdef np.float32_t lambda_pow_high = 5.0

    #np.ndarray[np.float32_t, ndim=1] lambda_possible = [(np.power(10.0, i), np.power(10.0, i+step)) for i in np.arange(lambda_pow_low,lambda_pow_high,step)]

    roots = [np.power(10.0, lambda_pow_low), np.power(10.0, lambda_pow_high)]

    cdef np.float32_t lambda0, lambda1, lambda_min, likelihood_lambda0, likelihood_lambda1

    for lambda_idx in np.arange(lambda_pow_low,lambda_pow_high,step):
        lambda0 = np.power(10.0, lambda_idx)
        lambda1 = np.power(10.0, lambda_idx+step)

        likelihood_lambda0 = likelihood_derivative1_restricted_lambda(lambda0, eigenVals, U, Y, W)
        likelihood_lambda1 = likelihood_derivative1_restricted_lambda(lambda1, eigenVals, U, Y, W)



        if np.sign(likelihood_lambda0) * np.sign(likelihood_lambda1) < 0:
            lambda_min = optimize.brentq(f=likelihood_derivative1_restricted_lambda, 
                                    a=lambda0,
                                    b=lambda1,
                                    rtol=0.1,
                                    maxiter=5000,
                                    args=(eigenVals, U, Y, W),
                                    disp=False)
            
            
            # TODO: Deal with lack of convergence
            # lambda_min = optimize.newton(func=lambda l: likelihood_derivative1_restricted_lambda(l, eigenVals, U, Y, W), 
            #                         x0=lambda_min,
            #                         rtol=1e-5,
            #                         fprime=lambda l: likelihood_derivative2_restricted_lambda(l, eigenVals, U, Y, W),
            #                         maxiter=10,
            #                         disp=False)

            # iter = 0
            
            # while True:
            #     d1 = likelihood_derivative1_restricted_lambda(lambda_min, eigenVals, U, Y, W)
            #     d2 = likelihood_derivative2_restricted_lambda(lambda_min, eigenVals, U, Y, W)
            #     ratio = d1/d2

            #     if np.sign(ratio) != np.sign(d1)*np.sign(d2):
            #         break

            #     lambda_new = lambda_min - ratio

            #     r_eps = np.abs(lambda_new-lambda_min)/np.abs(lambda_min)

            #     if (lambda_new < lambda_smallest) or (lambda_new > lambda_biggest):
            #         break

            #     lambda_min = lambda_new

            #     if (r_eps < 1e-5) or (iter >= 100):
            #         break
            #     iter = iter + 1

            lambda_min = newton(lambda_min, eigenVals, U, Y, W)

            roots.append(lambda_min)

    likelihood_list = np.array([likelihood_restricted_lambda(lam, eigenVals, U, Y, W) for lam in roots],dtype=np.float32)

    # Added relative maximum
    likelihood_list = [0.0] + list((likelihood_list[:likelihood_list.shape[0]-1] - likelihood_list[1:])/np.abs(likelihood_list[1:]))

    return roots[np.argmax(likelihood_list)]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef newton(np.float32_t lam,
                np.ndarray[np.float32_t, ndim=1] eigenVals, 
                np.ndarray[np.float32_t, ndim=2] U, 
                np.ndarray[np.float32_t, ndim=2] Y, 
                np.ndarray[np.float32_t, ndim=2] W):

    cdef np.float32_t lambda_min = lam

    cdef int iter = 0

    cdef np.float32_t r_eps = 0

    cdef np.float32_t lambda_new, ratio, d1, d2
            
    while True:
        d1 = likelihood_derivative1_restricted_lambda(lambda_min, eigenVals, U, Y, W)
        d2 = likelihood_derivative2_restricted_lambda(lambda_min, eigenVals, U, Y, W)
        ratio = d1/d2

        if np.sign(ratio) != np.sign(d1)*np.sign(d2):
            break

        lambda_new = lambda_min - ratio

        r_eps = np.abs(lambda_new-lambda_min)/np.abs(lambda_min)

        if lambda_new < 0:
            break

        lambda_min = lambda_new

        if (r_eps < 1e-5) or (iter >= 10):
            break
        iter = iter + 1

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


    #cdef np.ndarray[np.float32_t, ndim=2] W_xt_Px = W_x.T @ Px
    cdef np.ndarray[np.float32_t, ndim=2] H_inv = compute_H_inv(lam, eigenVals, U)
    
    #cdef np.ndarray[np.float32_t, ndim=2] beta_vec = np.linalg.inv(W_xt_Px @ W_x) @ (W_xt_Px @ Y)
    cdef np.ndarray[np.float32_t, ndim=2] beta_vec = np.linalg.inv(W_x.T @ H_inv @ W_x) @ (W_x.T @ H_inv @ Y)

    cdef np.float32_t beta = beta_vec[c,0]

    cdef np.float32_t ytPxy = Y.T @ Px @ Y

    # NOTE: x.T @ Px @ x is negative. We should fix this, but for now, we're throwing it away bc
    # it's not needed for anything
    #cdef np.float32_t se_beta = (1/np.sqrt((n - c - 1))) * np.sqrt(ytPxy)/np.sqrt(x.T @ Pc @ x)


    cdef np.float32_t tau = n/ytPxy

    return beta, beta_vec, None, tau

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef calc_beta_vg_ve_restricted(np.ndarray[np.float32_t, ndim=1] eigenVals,
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

    #cdef np.ndarray[np.float32_t, ndim=2] W_xt_Pc = W_x.T @ Pc
    
    #cdef np.ndarray[np.float32_t, ndim=2] beta_vec = np.linalg.inv(W_xt_Pc @ W_x) @ (W_xt_Pc @ Y)

    cdef np.ndarray[np.float32_t, ndim=2] H_inv = compute_H_inv(lam, eigenVals, U)
    
    cdef np.ndarray[np.float32_t, ndim=2] beta_vec = np.linalg.inv(W_x.T @ H_inv @ W_x) @ (W_x.T @ H_inv @ Y)

    cdef np.float32_t beta = (x.T @ Pc @ Y) / (x.T @ Pc @ x) #beta_vec[c,0]

    cdef np.float32_t ytPxy = Y.T @ Px @ Y

    cdef np.float32_t se_beta = np.sqrt(np.abs(ytPxy)) / np.sqrt(np.abs(x.T @ Pc @ x)) / np.sqrt((n - c - 1))

    cdef np.float32_t tau = (n-c-1)/ytPxy

    return np.float32(beta), beta_vec, np.float32(se_beta), np.float32(tau)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_lambda(np.float32_t lam,
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

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative1_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                  np.ndarray[np.float32_t, ndim=2] U, 
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W):
    n = Y.shape[0]

    cdef np.float32_t result = -0.5*((n-np.sum(1/(lam*eigenVals + 1.0)))/lam)

    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W, lam)

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_y = Y.T @ Px @ Y

    #cdef np.ndarray[np.float32_t, ndim=2] yT_Px_G_Px_y = (yT_Px_y - (Y.T @ Px) @ (Px @ Y))/lam

    #result = result + (n/2)*(yT_Px_G_Px_y/yT_Px_y)
    result = result + (n/2)*(1.0 - (Y.T @ Px) @ (Px @ Y) / yT_Px_y)/lam

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative2_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                  np.ndarray[np.float32_t, ndim=2] U, 
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W): 
    n = Y.shape[0]

    cdef np.float32_t result = 0.5*(n + np.sum(np.power(lam*eigenVals + 1.0, -2.0)) - 2*np.sum(np.power(lam*eigenVals + 1.0,-1.0)))/np.power(lam,2)

    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W, lam)

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_y = Y.T @ Px @ Y

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_Px_y = (Y.T @ Px) @ (Px @ Y)

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_G_Px_G_Px_y = (yT_Px_y + (Y.T @ Px) @ Px @ (Px @ Y) - 2*yT_Px_Px_y)/(lam*lam)

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

    result = result - 0.5 * n * (2 * yT_Px_G_Px_G_Px_y - yT_Px_G_Px_y * yT_Px_G_Px_y / yT_Px_y) / (yT_Px_y)

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative1_restricted_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                  np.ndarray[np.float32_t, ndim=2] U, 
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W):
    n = W.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W, lam)

    cdef np.float32_t result = -0.5*((n - c - np.trace(Px))/lam)

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_y = Y.T @ Px @ Y

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_G_Px_y = (yT_Px_y - (Y.T @ Px) @ (Px @ Y))/lam

    result = result + 0.5*(n - c)*yT_Px_G_Px_y/yT_Px_y

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative2_restricted_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                  np.ndarray[np.float32_t, ndim=2] U, 
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W): 
    cdef int n,c
    n = Y.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] Px = compute_Pc_cython(eigenVals, U, W, lam)

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_y = Y.T @ Px @ Y

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_Px_y = (Y.T @ Px) @ (Px @ Y)

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_G_Px_G_Px_y = (yT_Px_y + (Y.T @ Px) @ Px @ (Px @ Y) - 2*yT_Px_Px_y)/(lam*lam)

    cdef np.ndarray[np.float32_t, ndim=2] yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

    cdef np.float32_t result = 0.5*(n - c + np.trace(Px @ Px) - 2*np.trace(Px))/(lam*lam)

    #result = result - 0.5 * (n - c) * ((2 * yT_Px_G_Px_G_Px_y - yT_Px_G_Px_y * yT_Px_G_Px_y / yT_Px_y) / yT_Px_y) 
    result = result - (n - c) * ((yT_Px_G_Px_G_Px_y * yT_Px_y) - 0.5 * yT_Px_G_Px_y*yT_Px_G_Px_y) / (yT_Px_y * yT_Px_y)
    
    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood(np.float32_t lam, 
               np.float32_t tau, 
               np.ndarray[np.float32_t, ndim=2] beta, 
               np.ndarray[np.float32_t, ndim=1] eigenVals, 
               np.ndarray[np.float32_t, ndim=2] U, 
               np.ndarray[np.float32_t, ndim=2] Y, 
               np.ndarray[np.float32_t, ndim=2] W):
    cdef int n = W.shape[0]

    cdef np.ndarray[np.float32_t, ndim=2] H_inv = compute_H_inv(lam, eigenVals, U)

    cdef np.float32_t result = 0.5 * n *np.log(tau) 
    result = result - 0.5*n*np.log(2*np.pi)
    
    result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

    cdef np.ndarray[np.float32_t, ndim=2] y_W_beta = Y - W @ beta

    result = result - 0.5 * tau * y_W_beta.T @ H_inv @ y_W_beta


    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_restricted(np.float32_t lam, 
               np.float32_t tau,
               np.ndarray[np.float32_t, ndim=1] eigenVals, 
               np.ndarray[np.float32_t, ndim=2] U, 
               np.ndarray[np.float32_t, ndim=2] Y, 
               np.ndarray[np.float32_t, ndim=2] W):
    cdef int n,c
    n = W.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] H_inv = compute_H_inv(lam, eigenVals, U)

    cdef np.float32_t result = 0.5*(n - c)*np.log(tau)
    result = result - 0.5*(n - c)*np.log(2*np.pi)

    result = result + 0.5*np.linalg.slogdet(W.T @ W)[1]

    result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))

    #result = result - 0.5*np.log(np.linalg.det(W_x.T @ H_inv @ W_x)) # Causing NAN
    #result = result - 0.5*np.linalg.slogdet(W.T @ H_inv @ W)[1]
    result = result - 0.5*np.linalg.slogdet(compute_H_inv(lam, eigenVals, W.T @ U))[1]

    #result = result - 0.5*tau*np.log(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)
    result = result - 0.5*tau*(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_restricted_lambda(np.float32_t lam, 
                                 np.ndarray[np.float32_t, ndim=1] eigenVals, 
                                 np.ndarray[np.float32_t, ndim=2] U, 
                                 np.ndarray[np.float32_t, ndim=2] Y, 
                                 np.ndarray[np.float32_t, ndim=2] W):
    cdef int n,c
    n = W.shape[0]
    c = W.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] H_inv = compute_H_inv(lam, eigenVals, U)

    cdef np.float32_t result = 0.0 #0.5*(n - c)*np.log(0.5*(n - c)/np.pi)
    #result = result - 0.5*(n - c)
    #result = result + 0.5*np.linalg.slogdet(W.T @ W)[1]
    
    ##result = result - 0.5 * np.sum(np.log(lam*eigenVals + 1.0))
    result = result - 0.5 * np.sum(np.log(eigenVals + 1.0/lam)) - 0.5*n*np.log(lam)

    result = result - 0.5*(n - c)*np.log(Y.T @ compute_Pc(eigenVals, U, W, lam) @ Y)

    result = result - 0.5*np.linalg.slogdet(W.T @ H_inv @ W)[1]

    return np.float32(result) 

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef compute_H_inv(np.float32_t lam,
                   np.ndarray[np.float32_t, ndim=1] eigenVals,
                   np.ndarray[np.float32_t, ndim=2] U):            

    return U @ (1.0/(lam*eigenVals + 1.0)[:, np.newaxis] * U.T)




