# cython: infer_types=True
import cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, log
from cython cimport parallel

from scipy import optimize, stats
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas

from numpy cimport float_t, ndarray

import pandas as pd

#from scipy.optimize.cython_optimize cimport brentq

import time

MIN_VAL=1e-35

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
                             np.ndarray[np.float32_t, ndim=2] W,
                             bint precompute=True):

    # Loop over intervals and find where likelihood changes signs with respect to lambda
    cdef np.float32_t step = 1.0

    cdef np.float32_t lambda_pow_low = -5.0
    cdef np.float32_t lambda_pow_high = 5.0

    cdef float lambda0, lambda1, lambda_min, likelihood_lambda0, likelihood_lambda1 = 0.0

    cdef int maxiter = 100
    
    cdef float[:] lambda_possible = np.arange(lambda_pow_low,lambda_pow_high,step, dtype=np.float32)

    cdef n = W.shape[0]
    cdef c = W.shape[1] - 1

    if precompute:
        roots = [10.0 ** lambda_pow_low, 10.0 ** lambda_pow_high]

        precomp_low = precompute_mat(roots[0], eigenVals, W, Y, full=False)
        precomp_high = precompute_mat(roots[1], eigenVals, W, Y, full=False)

        likelihood_list = [likelihood_restricted_lambda_overload(roots[0], n, c+1, precomp_low['yt_Pi_y'][c+1], precomp_low['logdet_H'], precomp_low['logdet_Wt_W'], precomp_low['logdet_Wt_H_inv_W']), 
                           likelihood_restricted_lambda_overload(roots[1], n, c+1, precomp_high['yt_Pi_y'][c+1], precomp_high['logdet_H'], precomp_high['logdet_Wt_W'], precomp_high['logdet_Wt_H_inv_W'])]

        for idx in range(lambda_possible.shape[0]):
            lambda_idx = lambda_possible[idx]

            lambda0 = 10.0 ** (lambda_idx)
            lambda1 = 10.0 ** (lambda_idx + step)

            # If it's the first iteration
            if idx == 0:
                # Compute lower lambda
                likelihood_lambda0 = wrapper_likelihood_derivative1_restricted_lambda(lambda0, eigenVals, Y, W)

            else:
                # Reuse lambda likelihood from previous iteration
                likelihood_lambda0 = likelihood_lambda1

            
            likelihood_lambda1 = wrapper_likelihood_derivative1_restricted_lambda(lambda1, eigenVals, Y, W)

            if np.sign(likelihood_lambda0) * np.sign(likelihood_lambda1) < 0:
                
                # lambda_min = optimize.brentq(f=wrapper_likelihood_derivative1_restricted_lambda, 
                #                             a=lambda0,
                #                             b=lambda1,
                #                             rtol=1e-5,#rtol=0.1,
                #                             maxiter=maxiter,
                #                             args=(eigenVals, Y, W),
                #                             disp=False)
                
                lambda_min = optimize.brenth(f=wrapper_likelihood_derivative1_restricted_lambda, 
                                            a=lambda0,
                                            b=lambda1,
                                            rtol=0.1,
                                            maxiter=maxiter,
                                            args=(eigenVals, Y, W),
                                            disp=False)
                lambda_min = newton(lambda_min, eigenVals, Y, W, precompute=True, lambda_min=lambda0, lambda_max=lambda1)

                roots.append(lambda_min)

                precompute_dict = precompute_mat(lambda_min, eigenVals, W, Y, full=False)

                likelihood_list.append(likelihood_restricted_lambda_overload(lambda_min, n, c+1, precompute_dict['yt_Pi_y'][c+1], precompute_dict['logdet_H'], precompute_dict['logdet_Wt_W'], precompute_dict['logdet_Wt_H_inv_W']))
        
        #print(pd.DataFrame({'lambda': roots, 'likelihood': likelihood_list}))
        return roots[np.argmax(likelihood_list)]

    else:
        roots = [10.0 ** lambda_pow_low, 10.0 ** lambda_pow_high]
        likelihood_list = [likelihood_restricted_lambda(roots[0], eigenVals, Y, W), likelihood_restricted_lambda(roots[1], eigenVals, Y, W)]

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

            if np.sign(likelihood_lambda0) * np.sign(likelihood_lambda1) < 0:
                lambda_min = optimize.brentq(f=likelihood_derivative1_restricted_lambda, 
                                            a=lambda0,
                                            b=lambda1,
                                            rtol=0.1,
                                            maxiter=maxiter,
                                            args=(eigenVals, Y, W),
                                            disp=False)

                lambda_min = newton(lambda_min, eigenVals, Y, W, lambda_min=lambda0, lambda_max=lambda1)

                roots.append(lambda_min)

                likelihood_list.append(likelihood_restricted_lambda(lambda_min, eigenVals, Y, W))

        
        return roots[np.argmax(likelihood_list)]

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# @cython.cdivision(True)
# cpdef precompute_mat(float lam,
#                       np.ndarray[np.float32_t, ndim=1] eigenVals,
#                       np.ndarray[np.float32_t, ndim=2] W,
#                       np.ndarray[np.float32_t, ndim=2] Y,
#                       bint full=True):

#     cdef int i,j,k

#     cdef int c = W.shape[1]

#     cdef np.ndarray[np.float32_t, ndim=2] W_star = np.c_[W, Y]
    
#     # Note: W is now a matrix of shape (n, c+1), where n is the number of samples and c is the number of covariates
#     # W[:,c+1] is the phenotype vector

#     cdef np.ndarray[np.float32_t, ndim=1] Hi_eval = 1.0 / (lam*eigenVals + 1.0)

#     cdef np.ndarray[np.float32_t, ndim=3] wjt_Pi_wk = np.zeros((W_star.shape[1], 
#                                                                 c+1, 
#                                                                 W_star.shape[1]),
#                                                                 dtype=np.float32)

#     cdef np.ndarray[np.float32_t, ndim=3] wjt_Pi_Pi_wk
#     cdef np.ndarray[np.float32_t, ndim=3] wjt_Pi_Pi_Pi_wk

#     cdef np.ndarray[np.float32_t, ndim=1] tr_Pi
#     cdef np.ndarray[np.float32_t, ndim=1] tr_Pi_Pi

#     cdef np.ndarray[np.uint32_t, ndim=1] indices = np.arange(c, dtype=np.uint32)

#     cdef float logdet_Wt_H_inv_W #= 0.0 #np.linalg.slogdet(W.T @ (Hi_eval[:,np.newaxis] * W))[1]

#     #start = time.time()

#     # Pi Computations
#     #start = time.time()
#     for i in range(c+1): # Loop for Pi
#         if i == 0:
#             #wjt_Pi_wk[:,i,:] = W_star.T @ (Hi_eval[:,np.newaxis] * W_star)
#             wjt_Pi_wk[:,:,:] = (W_star.T @ (Hi_eval[:,np.newaxis] * W_star))[:,np.newaxis,:]
#         else:
#             #wjt_Pi_wk[:,i,:] = wjt_Pi_wk[:,i-1,:] - np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) / max(wjt_Pi_wk[i-1,i-1,i-1], MIN_VAL)
#             wjt_Pi_wk[:,i:,:] -= (np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]))[:,np.newaxis,:]
#         wjt_Pi_wk[i,i,i] = max(wjt_Pi_wk[i,i,i], MIN_VAL)
   
#     #print(f"Pi: {round(time.time() - start, 4)} s")

#     #print(np.min(wjt_Pi_wk[indices,indices,indices]))

#     logdet_Wt_H_inv_W = np.sum(np.log(wjt_Pi_wk[indices,indices,indices]))


#     wjt_Pi_Pi_wk = np.zeros((W_star.shape[1], 
#                             c+1, 
#                             W_star.shape[1]), 
#                             dtype=np.float32)

#     tr_Pi = np.zeros((c+1,), dtype=np.float32)

#     # Pi @ Pi Computations
#     #start = time.time()
#     for i in range(c+1): # Loop for Pi
#         if i == 0:
#             wjt_Pi_Pi_wk[:,:,:] = (W_star.T @ ((Hi_eval ** 2.0)[:,np.newaxis] * W_star))[:,np.newaxis,:]
#         else:
#             # wjt_Pi_Pi_wk[:,i:,:] +=  ((np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_wk[:,i-1,i-1])) \
#             #                                 - (transpose_sum(np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]))))[:,np.newaxis,:]
#             wjt_Pi_Pi_wk[:,i:,:] +=  ((np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_wk[:,i-1,i-1])) \
#                                             - np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]) \
#                                             - np.outer(wjt_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1]))[:,np.newaxis,:]
#         wjt_Pi_Pi_wk[i,i,i] = max(wjt_Pi_Pi_wk[i,i,i], MIN_VAL)

#     #print(f"Pi @ Pi: {round(time.time() - start, 4)} s")

#     #start = time.time()
#     tr_Pi[0] = np.sum(Hi_eval)

#     # Use np.cumsum to compute trace
#     if c > 0:
#         tr_Pi[1:] = - wjt_Pi_Pi_wk[indices,indices,indices] / wjt_Pi_wk[indices,indices,indices]
#         tr_Pi = np.cumsum(tr_Pi)
#     #print(f"Trace Pi: {round(time.time() - start, 4)} s")


#     # If not performing full computation
#     # Skips computations if we only need first derivative (meaning brent)
#     if not full:
#         #start = time.time()
#         precompute_dict = {
#                         'wjt_Pi_wk'                 : wjt_Pi_wk[:c, :, :c],
#                         'wjt_Pi_Pi_wk'              : wjt_Pi_Pi_wk[:c, :, :c],
#                         'yt_Pi_y'                   : wjt_Pi_wk[c, :, c].reshape(-1),
#                         'yt_Pi_Pi_y'                : wjt_Pi_Pi_wk[c, :, c].reshape(-1),
#                         'tr_Pi'                     : tr_Pi,
#                         'logdet_Wt_W'               : np.linalg.slogdet(W.T @ W)[1],
#                         'logdet_Wt_H_inv_W'         : logdet_Wt_H_inv_W,
#                         'logdet_H'                  : np.sum(np.log(lam*eigenVals + 1.0)),
#                         }
#         #print(f"Precompute: {round(time.time() - start, 4)} s")

#         return precompute_dict

#     else:
#         wjt_Pi_Pi_Pi_wk = np.zeros((W_star.shape[1], 
#                                     c+1, 
#                                     W_star.shape[1]), 
#                                     dtype=np.float32)
        
#         tr_Pi_Pi = np.zeros((c+1,), dtype=np.float32)
        
#         # Pi @ Pi @ Pi Computations
#         #start = time.time()
#         for i in range(c+1): # Loop for Pi
#             if i == 0:
#                 #wjt_Pi_Pi_Pi_wk[:,i,:] = W_star.T @ ((Hi_eval ** 3.0)[:,np.newaxis] * W_star)
#                 wjt_Pi_Pi_Pi_wk[:,i,:] = (W_star.T @ ((Hi_eval ** 3.0)[:,np.newaxis] * W_star))#[:,np.newaxis,:]
#             else:
#                 # wjt_Pi_Pi_Pi_wk[:,i,:] = wjt_Pi_Pi_Pi_wk[:,i-1,:] \
#                 #                                     - np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) * ((wjt_Pi_Pi_wk[i-1,i-1,i-1] ** 2.0) / (wjt_Pi_wk[i-1,i-1,i-1] ** 3.0)) \
#                 #                                     + ( - np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_Pi_wk[:,i-1,i-1]) \
#                 #                                     - np.outer(wjt_Pi_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) \
#                 #                                     - np.outer(wjt_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1])) / wjt_Pi_wk[i-1,i-1,i-1] \
#                 #                                     + (np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]) \
#                 #                                     + np.outer(wjt_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1])) * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)) \
#                 #                                     + np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) * (wjt_Pi_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0))
#                 # wjt_Pi_Pi_Pi_wk[:,i,:] = wjt_Pi_Pi_Pi_wk[:,i-1,:] \
#                 #                                     - np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) * ((wjt_Pi_Pi_wk[i-1,i-1,i-1] ** 2.0) / (wjt_Pi_wk[i-1,i-1,i-1] ** 3.0)) \
#                 #                                     - (transpose_sum(np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_Pi_wk[:,i-1,i-1])) \
#                 #                                     + np.outer(wjt_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1])) / wjt_Pi_wk[i-1,i-1,i-1] \
#                 #                                     + transpose_sum(np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1])) * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)) \
#                 #                                     + np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) * (wjt_Pi_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0))
#                 # wjt_Pi_Pi_Pi_wk[:,i,:] = wjt_Pi_Pi_Pi_wk[:,i-1,:] \
#                 #                                     - np.outer(wjt_Pi_wk[:,i-1,i-1] * ((wjt_Pi_Pi_wk[i-1,i-1,i-1] ** 2.0) / (wjt_Pi_wk[i-1,i-1,i-1] ** 3.0)), wjt_Pi_wk[:,i-1,i-1]) \
#                 #                                     - transpose_sum(np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_Pi_wk[:,i-1,i-1])) \
#                 #                                     - np.outer(wjt_Pi_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]) \
#                 #                                     + transpose_sum(np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]) * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0))) \
#                 #                                     + np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_wk[:,i-1,i-1])
#                 # wjt_Pi_Pi_Pi_wk[:,i,:] = wjt_Pi_Pi_Pi_wk[:,i-1,:] \
#                 #                                     - np.outer(wjt_Pi_wk[:,i-1,i-1] * ((wjt_Pi_Pi_wk[i-1,i-1,i-1] ** 2.0) / (wjt_Pi_wk[i-1,i-1,i-1] ** 3.0)), wjt_Pi_wk[:,i-1,i-1]) \
#                 #                                     - transpose_sum(np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_Pi_wk[:,i-1,i-1])) \
#                 #                                     - np.outer(wjt_Pi_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]) \
#                 #                                     + transpose_sum(np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_Pi_wk[:,i-1,i-1])) \
#                 #                                     + np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_wk[:,i-1,i-1])
#                 wjt_Pi_Pi_Pi_wk[:,i,:] = wjt_Pi_Pi_Pi_wk[:,i-1,:] \
#                                                     - np.outer(wjt_Pi_wk[:,i-1,i-1] * ((wjt_Pi_Pi_wk[i-1,i-1,i-1] ** 2.0) / (wjt_Pi_wk[i-1,i-1,i-1] ** 3.0)), wjt_Pi_wk[:,i-1,i-1]) \
#                                                     - np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_Pi_wk[:,i-1,i-1]) \
#                                                     - np.outer(wjt_Pi_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1]) \
#                                                     - np.outer(wjt_Pi_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]) \
#                                                     + np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_Pi_wk[:,i-1,i-1]) \
#                                                     + np.outer(wjt_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0))) \
#                                                     + np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_wk[:,i-1,i-1])
#             wjt_Pi_Pi_Pi_wk[i,i,i] = max(wjt_Pi_Pi_Pi_wk[i,i,i], MIN_VAL)

#                 # wjt_Pi_Pi_Pi_wk[:,i:,:] += (- np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) * ((wjt_Pi_Pi_wk[i-1,i-1,i-1] ** 2.0) / (wjt_Pi_wk[i-1,i-1,i-1] ** 3.0)) \
#                 #                                     + ( - np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_Pi_wk[:,i-1,i-1]) \
#                 #                                     - np.outer(wjt_Pi_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) \
#                 #                                     - np.outer(wjt_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1])) / wjt_Pi_wk[i-1,i-1,i-1] \
#                 #                                     + (np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]) \
#                 #                                     + np.outer(wjt_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1])) * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)) \
#                 #                                     + np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) * (wjt_Pi_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)))[:,np.newaxis,:]
                
#         #print(f"Pi @ Pi @ Pi: {round(time.time() - start, 4)} s")


#         # Use np.cumsum to compute trace
#         #start = time.time()
#         tr_Pi_Pi[0] = np.sum(Hi_eval ** 2.0)

#         if c > 0:
#             tr_Pi_Pi[1:] = ((wjt_Pi_Pi_wk[indices,indices,indices] / wjt_Pi_wk[indices,indices,indices]) ** 2.0) \
#                             - 2 * (wjt_Pi_Pi_Pi_wk[indices,indices,indices] / wjt_Pi_wk[indices,indices,indices])
#             tr_Pi_Pi = np.cumsum(tr_Pi_Pi)
#         #print(f"Trace Pi @ Pi: {round(time.time() - start, 4)} s")

#         #start = time.time()
#         precompute_dict = {
#                             'wjt_Pi_wk'                 : wjt_Pi_wk[:c, :, :c],
#                             'wjt_Pi_Pi_wk'              : wjt_Pi_Pi_wk[:c, :, :c],
#                             'wjt_Pi_Pi_Pi_wk'           : wjt_Pi_Pi_Pi_wk[:c, :, :c],
#                             'tr_Pi'                     : tr_Pi,
#                             'tr_Pi_Pi'                  : tr_Pi_Pi,
#                             'yt_Pi_y'                   : wjt_Pi_wk[c, :, c].reshape(-1),
#                             'yt_Pi_Pi_y'                : wjt_Pi_Pi_wk[c, :, c].reshape(-1),
#                             'yt_Pi_Pi_Pi_y'             : wjt_Pi_Pi_Pi_wk[c, :, c].reshape(-1),
#                             'logdet_Wt_W'               : np.linalg.slogdet(W.T @ W)[1],
#                             'logdet_Wt_H_inv_W'         : logdet_Wt_H_inv_W,
#                             'logdet_H'                  : np.sum(np.log(lam*eigenVals + 1.0)),
#                             }
#         #print(f"Precompute: {round(time.time() - start, 4)} s")
        
#         #print(f"Runtime: {round(time.time() - start, 4)} s")

#         return precompute_dict

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef precompute_mat(float lam,
                      np.ndarray[np.float32_t, ndim=1] eigenVals,
                      np.ndarray[np.float32_t, ndim=2] W,
                      np.ndarray[np.float32_t, ndim=2] Y,
                      bint full=True):

    cdef int i,j,k

    cdef int c = W.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2] W_star = np.c_[W.astype(np.float64), Y.astype(np.float64)]
    
    # Note: W is now a matrix of shape (n, c+1), where n is the number of samples and c is the number of covariates
    # W[:,c+1] is the phenotype vector

    cdef np.ndarray[np.float64_t, ndim=1] Hi_eval = 1.0 / (np.float64(lam)*eigenVals.astype(np.float64) + 1.0)
    #print(lam, Hi_eval[:5])

    cdef np.ndarray[np.float64_t, ndim=3] wjt_Pi_wk = np.zeros((W_star.shape[1], 
                                                                c+1, 
                                                                W_star.shape[1]),
                                                                dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=3] wjt_Pi_Pi_wk
    cdef np.ndarray[np.float64_t, ndim=3] wjt_Pi_Pi_Pi_wk

    cdef np.ndarray[np.float64_t, ndim=1] tr_Pi
    cdef np.ndarray[np.float64_t, ndim=1] tr_Pi_Pi

    cdef np.ndarray[np.uint32_t, ndim=1] indices = np.arange(c, dtype=np.uint32)

    cdef float logdet_Wt_H_inv_W #= 0.0 #np.linalg.slogdet(W.T @ (Hi_eval[:,np.newaxis] * W))[1]

    #start = time.time()

    # Pi Computations
    #start = time.time()
    for i in range(c+1): # Loop for Pi
        if i == 0:
            #wjt_Pi_wk[:,i,:] = (W_star.T @ (Hi_eval[:,np.newaxis] * W_star)).T
            wjt_Pi_wk[:,:,:] = (W_star.T @ (Hi_eval[:,np.newaxis] * W_star))[:,np.newaxis,:]
            # for j in range(W_star.shape[1]):
            #     for k in range(W_star.shape[1]):
            #         wjt_Pi_wk[j,i,k] = np.sum(W_star[:,k] * Hi_eval * W_star[:,j])
        elif wjt_Pi_wk[i-1,i-1,i-1] != 0.0:
            #wjt_Pi_wk[:,i,:] = wjt_Pi_wk[:,i-1,:] - np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) / wjt_Pi_wk[i-1,i-1,i-1]
            wjt_Pi_wk[:,i,:] = wjt_Pi_wk[:,i-1,:] - np.outer(wjt_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]) / wjt_Pi_wk[i-1,i-1,i-1]
            #wjt_Pi_wk[:,i:,:] -= (np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_wk[:,i-1,i-1]))[:,np.newaxis,:]
            # for j in range(W_star.shape[1]):
            #     for k in range(W_star.shape[1]):
            #         wjt_Pi_wk[j,i,k] = wjt_Pi_wk[j,i-1,k] - (wjt_Pi_wk[j,i-1,i-1] * wjt_Pi_wk[i-1,i-1,k]) / wjt_Pi_wk[i-1,i-1,i-1]
        else:
            wjt_Pi_wk[:,i,:] = wjt_Pi_wk[:,i-1,:]

        #wjt_Pi_wk[i,i,i] = max(wjt_Pi_wk[i,i,i], MIN_VAL)
   
    #print(f"Pi: {round(time.time() - start, 4)} s")

    #print(np.min(wjt_Pi_wk[indices,indices,indices]))

    logdet_Wt_H_inv_W = float(np.sum(np.log(wjt_Pi_wk[indices,indices,indices])))
    #logdet_Wt_H_inv_W = float(np.linalg.slogdet(W.T @ (Hi_eval[:,np.newaxis] * W))[1])


    wjt_Pi_Pi_wk = np.zeros((W_star.shape[1], 
                            c+1, 
                            W_star.shape[1]), 
                            dtype=np.float64)

    tr_Pi = np.zeros((c+1,), dtype=np.float64)

    # Pi @ Pi Computations
    #start = time.time()
    for i in range(c+1): # Loop for Pi
        if i == 0:
            wjt_Pi_Pi_wk[:,:,:] = (W_star.T @ ((Hi_eval ** 2.0)[:,np.newaxis] * W_star))[:,np.newaxis,:]
        else:
            # wjt_Pi_Pi_wk[:,i:,:] +=  ((np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_wk[:,i-1,i-1])) \
            #                                 - (transpose_sum(np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]))))[:,np.newaxis,:]
            wjt_Pi_Pi_wk[:,i:,:] +=  ((np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_wk[:,i-1,i-1])) \
                                            - np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]) \
                                            - np.outer(wjt_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1]))[:,np.newaxis,:]
        wjt_Pi_Pi_wk[i,i,i] = max(wjt_Pi_Pi_wk[i,i,i], MIN_VAL)

    #print(f"Pi @ Pi: {round(time.time() - start, 4)} s")

    #start = time.time()
    tr_Pi[0] = np.sum(Hi_eval)

    # Use np.cumsum to compute trace
    if c > 0:
        tr_Pi[1:] = - wjt_Pi_Pi_wk[indices,indices,indices] / wjt_Pi_wk[indices,indices,indices]
        tr_Pi = np.cumsum(tr_Pi)
    #print(f"Trace Pi: {round(time.time() - start, 4)} s")


    # If not performing full computation
    # Skips computations if we only need first derivative (meaning brent)
    if not full:
        #start = time.time()
        precompute_dict = {
                        #'wjt_Pi_wk'                 : wjt_Pi_wk[:c, :, :c].astype(np.float32),
                        'wjt_Pi_wk'                 : wjt_Pi_wk[:, :, :].astype(np.float32),
                        'wjt_Pi_Pi_wk'              : wjt_Pi_Pi_wk[:c, :, :c].astype(np.float32),
                        'yt_Pi_y'                   : wjt_Pi_wk[c, :, c].reshape(-1).astype(np.float32),
                        'yt_Pi_Pi_y'                : wjt_Pi_Pi_wk[c, :, c].reshape(-1).astype(np.float32),
                        'tr_Pi'                     : tr_Pi.astype(np.float32),
                        'logdet_Wt_W'               : 0.0, #float(np.linalg.slogdet(W.T @ W)[1]),
                        'logdet_Wt_H_inv_W'         : float(logdet_Wt_H_inv_W),
                        'logdet_H'                  : float(np.sum(np.log(lam*eigenVals + 1.0))),
                        }
        #print(f"Precompute: {round(time.time() - start, 4)} s")

        return precompute_dict

    else:
        wjt_Pi_Pi_Pi_wk = np.zeros((W_star.shape[1], 
                                    c+1, 
                                    W_star.shape[1]), 
                                    dtype=np.float64)
        
        tr_Pi_Pi = np.zeros((c+1,), dtype=np.float64)
        
        # Pi @ Pi @ Pi Computations
        #start = time.time()
        for i in range(c+1): # Loop for Pi
            if i == 0:
                #wjt_Pi_Pi_Pi_wk[:,i,:] = W_star.T @ ((Hi_eval ** 3.0)[:,np.newaxis] * W_star)
                wjt_Pi_Pi_Pi_wk[:,i,:] = (W_star.T @ ((Hi_eval ** 3.0)[:,np.newaxis] * W_star))#[:,np.newaxis,:]
            else:
                wjt_Pi_Pi_Pi_wk[:,i,:] = wjt_Pi_Pi_Pi_wk[:,i-1,:] \
                                                    - np.outer(wjt_Pi_wk[:,i-1,i-1] * ((wjt_Pi_Pi_wk[i-1,i-1,i-1] ** 2.0) / (wjt_Pi_wk[i-1,i-1,i-1] ** 3.0)), wjt_Pi_wk[:,i-1,i-1]) \
                                                    - np.outer(wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_Pi_wk[:,i-1,i-1]) \
                                                    - np.outer(wjt_Pi_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1]) \
                                                    - np.outer(wjt_Pi_Pi_wk[:,i-1,i-1] / wjt_Pi_wk[i-1,i-1,i-1], wjt_Pi_Pi_wk[:,i-1,i-1]) \
                                                    + np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_Pi_wk[:,i-1,i-1]) \
                                                    + np.outer(wjt_Pi_Pi_wk[:,i-1,i-1], wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0))) \
                                                    + np.outer(wjt_Pi_wk[:,i-1,i-1] * (wjt_Pi_Pi_Pi_wk[i-1,i-1,i-1] / (wjt_Pi_wk[i-1,i-1,i-1] ** 2.0)), wjt_Pi_wk[:,i-1,i-1])
            wjt_Pi_Pi_Pi_wk[i,i,i] = max(wjt_Pi_Pi_Pi_wk[i,i,i], MIN_VAL)

        #print(f"Pi @ Pi @ Pi: {round(time.time() - start, 4)} s")


        # Use np.cumsum to compute trace
        #start = time.time()
        tr_Pi_Pi[0] = np.sum(Hi_eval ** 2.0)

        if c > 0:
            tr_Pi_Pi[1:] = ((wjt_Pi_Pi_wk[indices,indices,indices] / wjt_Pi_wk[indices,indices,indices]) ** 2.0) \
                            - 2 * (wjt_Pi_Pi_Pi_wk[indices,indices,indices] / wjt_Pi_wk[indices,indices,indices])
            tr_Pi_Pi = np.cumsum(tr_Pi_Pi)
        #print(f"Trace Pi @ Pi: {round(time.time() - start, 4)} s")

        #start = time.time()
        precompute_dict = {
                            #'wjt_Pi_wk'                 : wjt_Pi_wk[:c, :, :c].astype(np.float32),
                            'wjt_Pi_wk'                 : wjt_Pi_wk[:, :, :].astype(np.float32),
                            'wjt_Pi_Pi_wk'              : wjt_Pi_Pi_wk[:c, :, :c].astype(np.float32),
                            'wjt_Pi_Pi_Pi_wk'           : wjt_Pi_Pi_Pi_wk[:c, :, :c].astype(np.float32),
                            'tr_Pi'                     : tr_Pi.astype(np.float32),
                            'tr_Pi_Pi'                  : tr_Pi_Pi.astype(np.float32),
                            'yt_Pi_y'                   : wjt_Pi_wk[c, :, c].reshape(-1).astype(np.float32),
                            'yt_Pi_Pi_y'                : wjt_Pi_Pi_wk[c, :, c].reshape(-1).astype(np.float32),
                            'yt_Pi_Pi_Pi_y'             : wjt_Pi_Pi_Pi_wk[c, :, c].reshape(-1).astype(np.float32),
                            'logdet_Wt_W'               : 0.0, #float(np.linalg.slogdet(W.T @ W)[1]),
                            'logdet_Wt_H_inv_W'         : float(logdet_Wt_H_inv_W),
                            'logdet_H'                  : float(np.sum(np.log(lam*eigenVals + 1.0))),
                            }
        #print(f"Precompute: {round(time.time() - start, 4)} s")
        
        #print(f"Runtime: {round(time.time() - start, 4)} s")

        return precompute_dict

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef transpose_sum(np.ndarray[np.float32_t, ndim=2] mat):
    return mat + mat.T

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
                   np.ndarray[np.float32_t, ndim=2] W,
                   bint precompute=False,
                   float lambda_min=1e-5,
                   float lambda_max=1e5):

    cdef float lambda_root = lam
    cdef int iteration = 0
    cdef float r_eps = 0.0
    cdef float lambda_new, ratio, d1, d2

    cdef int n = W.shape[0]
    cdef int c = W.shape[1]

    cdef dict precompute_dict

    while True:
        if precompute:
            precompute_dict = precompute_mat(lambda_root, eigenVals, W, Y, full=True)

            d1 = likelihood_derivative1_restricted_lambda_overload(lam=lambda_root,
                                                    n=n,
                                                    c=c,
                                                    yt_Px_y=precompute_dict['yt_Pi_y'][c],
                                                    yt_Px_Px_y=precompute_dict['yt_Pi_Pi_y'][c],
                                                    tr_Px=precompute_dict['tr_Pi'][c])
            d2 = likelihood_derivative2_restricted_lambda_overload(lam=lambda_root,
                                                            n=n,
                                                            c=c,
                                                            yt_Px_y=precompute_dict['yt_Pi_y'][c],
                                                            yt_Px_Px_y=precompute_dict['yt_Pi_Pi_y'][c],
                                                            yt_Px_Px_Px_y=precompute_dict['yt_Pi_Pi_Pi_y'][c],
                                                            tr_Px=precompute_dict['tr_Pi'][c],
                                                            tr_Px_Px=precompute_dict['tr_Pi_Pi'][c])
        else:
            d1 = likelihood_derivative1_restricted_lambda(lambda_root, eigenVals, Y, W)
            d2 = likelihood_derivative2_restricted_lambda(lambda_root, eigenVals, Y, W)

        with cython.nogil:
            ratio = d1/d2

        if np.sign(ratio) * np.sign(d1) * np.sign(d2) <= 0.0:
            break

        lambda_new = lambda_root - ratio
        r_eps = fabs(lambda_new - lambda_root) / fabs(lambda_root)

        if lambda_new < lambda_min:
            lambda_new = lambda_min
            break

        if lambda_new > lambda_max:
            lambda_new = lambda_max
            break

        if np.isnan(lambda_new) or np.isinf(lambda_new):
            break

        lambda_root = lambda_new

        if r_eps < 1e-5 or iteration > 100:
            break

        iteration += 1

    return lambda_root

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
    
    return H_inv - H_inv_W @ np.linalg.inv(W.T @ H_inv_W) @ H_inv_W.T

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

# Might overload this too
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
    
    #cdef np.ndarray[np.float32_t, ndim=2] beta_vec = np.linalg.inv(W_x_t_H_inv @ W_x) @ (W_x_t_H_inv @ Y)
    
    #cdef np.float32_t beta = beta_vec[c,0] #compute_at_Pi_b(lam, c, mod_eig, W, x, Y) / max(compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL) # Double check this property holds, but I think it does bc positive symmetric
    cdef np.float32_t beta = compute_at_Pi_b(lam, c, mod_eig, W, x, Y) / max(compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL) # Double check this property holds, but I think it does bc positive symmetric

    cdef np.float32_t ytPxy = max(compute_at_Pi_b(lam, c+1, mod_eig, W_x, Y, Y), MIN_VAL)

    cdef np.float32_t se_beta = np.sqrt(ytPxy) / (np.sqrt(max(compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL)) * np.sqrt((n - c - 1)))

    cdef np.float32_t tau = (n-c-1)/ytPxy

    #return np.float32(beta), beta_vec, np.float32(se_beta), np.float32(tau)
    return np.float32(beta), 0.0, np.float32(se_beta), np.float32(tau)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef calc_beta_vg_ve_restricted_overload(np.ndarray[np.float32_t, ndim=1] eigenVals,
                                        np.ndarray[np.float32_t, ndim=2] W, 
                                        np.ndarray[np.float32_t, ndim=2] x, 
                                        np.float32_t lam, 
                                        np.ndarray[np.float32_t, ndim=2] Y):
    cdef np.ndarray[np.float32_t, ndim=2] W_x = np.c_[W,x]

    cdef dict precompute_dict = precompute_mat(lam, eigenVals, W_x, Y, full=False)

    cdef int n, c
    n = W.shape[0]
    c = W.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=1] mod_eig = lam*eigenVals + 1.0

    #cdef np.ndarray[np.float32_t, ndim=2] W_x_t_H_inv = ((1.0/mod_eig)[:,np.newaxis] * W_x).T
    
    #cdef np.ndarray[np.float32_t, ndim=2] beta_vec = np.linalg.inv(W_x_t_H_inv @ W_x) @ (W_x_t_H_inv @ Y)
    
    #cdef np.float32_t beta = beta_vec[c,0] #compute_at_Pi_b(lam, c, mod_eig, W, x, Y) / max(compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL) # Double check this property holds, but I think it does bc positive symmetric
    cdef np.float32_t beta = precompute_dict['wjt_Pi_wk'][c,c,c+1] / precompute_dict['wjt_Pi_wk'][c,c,c]

    cdef np.float32_t ytPxy = precompute_dict['yt_Pi_y'][c+1] #max(compute_at_Pi_b(lam, c+1, mod_eig, W_x, Y, Y), MIN_VAL)

    cdef np.float32_t se_beta = np.sqrt(ytPxy) / (np.sqrt(max(precompute_dict['wjt_Pi_wk'][c,c,c], MIN_VAL)) * np.sqrt((n - c - 1)))
    #cdef np.float32_t se_beta = np.sqrt(ytPxy) / (np.sqrt(max(compute_at_Pi_b(lam, c, mod_eig, W, x, x), MIN_VAL)) * np.sqrt((n - c - 1)))

    cdef np.float32_t tau = (n-c-1)/ytPxy

    return np.float32(beta), 0.0, np.float32(se_beta), np.float32(tau)
    #return np.float32(beta), beta_vec, np.float32(se_beta), np.float32(tau)

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

    cdef np.float32_t yT_Px_y = max(compute_at_Pi_b(lam, c, mod_eig, W, Y, Y), 0)
    cdef np.float32_t yT_Px_Px_y = max(compute_at_Pi_Pi_b(lam, c, mod_eig, W, Y, Y), 0)

    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam
    cdef np.float32_t result = -0.5*((n - c - trace_Pi(lam, c, mod_eig, W))/lam) # -0.5*tr(Px @ G)

    result = result + 0.5*(n - c)*yT_Px_G_Px_y/yT_Px_y # 0.5 * Y.T @ Px @ G @ Px @ Y / (Y.T @ Px @ Y)

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef wrapper_likelihood_derivative1_restricted_lambda(np.float32_t lam,
                                  np.ndarray[np.float32_t, ndim=1] eigenVals,
                                  np.ndarray[np.float32_t, ndim=2] Y, 
                                  np.ndarray[np.float32_t, ndim=2] W):
    cdef int n, c
    n = W.shape[0]
    c = W.shape[1]


    cdef dict precompute_dict = precompute_mat(lam, eigenVals, W, Y, full=False)

    return likelihood_derivative1_restricted_lambda_overload(lam=lam,
                                                    n=n,
                                                    c=c,
                                                    yt_Px_y=precompute_dict['yt_Pi_y'][c],
                                                    yt_Px_Px_y=precompute_dict['yt_Pi_Pi_y'][c],
                                                    tr_Px=precompute_dict['tr_Pi'][c])


# Lookup version
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative1_restricted_lambda_overload(float lam,
                                                int n,
                                                int c,
                                                float yt_Px_y,
                                                float yt_Px_Px_y,
                                                float tr_Px):

    cdef np.float32_t yT_Px_y = max(yt_Px_y, MIN_VAL)

    cdef np.float32_t result = -0.5*((n - c - tr_Px)/lam) # -0.5*tr(Px @ G)

    result = result + 0.5*(n - c)*((yT_Px_y - max(yt_Px_Px_y, 0))/lam)/yT_Px_y # 0.5 * Y.T @ Px @ G @ Px @ Y / (Y.T @ Px @ Y)

    return np.float32(result)

# Lookup version
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_derivative2_restricted_lambda_overload(float lam,
                                                int n,
                                                int c,
                                                float yt_Px_y,
                                                float yt_Px_Px_y,
                                                float yt_Px_Px_Px_y,
                                                float tr_Px,
                                                float tr_Px_Px):

    cdef np.float32_t yT_Px_y = max(yt_Px_y, MIN_VAL)

    cdef np.float32_t yT_Px_Px_y = max(yt_Px_Px_y, MIN_VAL)

    cdef np.float32_t yT_Px_Px_Px_y = max(yt_Px_Px_Px_y, MIN_VAL)

    cdef np.float32_t yT_Px_G_Px_G_Px_y = (yT_Px_y + yT_Px_Px_Px_y - 2*yT_Px_Px_y)/(lam**2.0)

    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam
    
    cdef np.float32_t result = 0.5*(n - c + tr_Px_Px - 2*tr_Px)/(lam**2.0)
    
    result = result - (n - c) * ((yT_Px_G_Px_G_Px_y * yT_Px_y) - 0.5 * yT_Px_G_Px_y*yT_Px_G_Px_y) / (yT_Px_y ** 2.0)
    
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

    cdef np.float32_t yT_Px_G_Px_G_Px_y = (yT_Px_y + yT_Px_Px_Px_y - 2*yT_Px_Px_y)/(lam**2.0)

    cdef np.float32_t yT_Px_G_Px_y = (yT_Px_y - yT_Px_Px_y)/lam

    cdef float[:] tr_arr = joint_trace(lam, c, mod_eig, W)
    
    #cdef np.float32_t result = 0.5*(n - c + tr_arr[0] - 2*tr_arr[1])/(lam**2.0)
    cdef np.float32_t result = 0.5*(n - c + trace_Pi_Pi(lam, c, mod_eig, W) - 2*trace_Pi(lam, c, mod_eig, W))/(lam*lam)
    
    result = result - (n - c) * ((yT_Px_G_Px_G_Px_y * yT_Px_y) - 0.5 * yT_Px_G_Px_y*yT_Px_G_Px_y) / (yT_Px_y ** 2.0)
    
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

    result = result - 0.5*np.linalg.slogdet(Wt_eig_W)[1]

    result = result - 0.5*(n - c)*log(max(compute_at_Pi_b(lam, c, mod_eig, W, Y, Y), MIN_VAL))

    return np.float32(result)

# Lookup version
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef likelihood_restricted_lambda_overload(float lam,
                                    int n,
                                    int c,
                                    float yt_Px_y,
                                    float logdet_H,
                                    float logdet_Wt_W,
                                    float logdet_Wt_H_inv_W):

    cdef np.float32_t result = 0.5*(n - c)*log(0.5*(n - c)/np.pi)
    result = result - 0.5*(n - c)
    result = result + 0.5*logdet_Wt_W
    result = result - 0.5 * logdet_H

    result = result - 0.5*logdet_Wt_H_inv_W

    result = result - 0.5*(n - c)*log(yt_Px_y)

    return np.float32(result)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef compute_H_inv(np.float32_t lam,
                   np.ndarray[np.float32_t, ndim=1] eigenVals,
                   np.ndarray[np.float32_t, ndim=2] U):            

    return U @ (1.0/(lam*eigenVals + 1.0)[:, np.newaxis] * U.T)

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
    cdef float[:,:] W_i, W_i_sub
    cdef float num, denom

    while i > 0:
        W_i = W[:, (i-1):i]
        W_i_sub = W[:, 0:(i-1)]
        result -= compute_at_Pi_Pi_b(lam, i-1, eigenVals, W_i_sub, W_i, W_i) / compute_at_Pi_b(lam, i-1, eigenVals, W_i_sub, W_i, W_i)

        i -= 1
    
    result += np.sum(np.power(eigenVals, -1.0))
    
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef trace_Pi_recursive(np.float32_t lam,
              int i,
              np.ndarray[np.float32_t, ndim=1] eigenVals,
              np.ndarray[np.float32_t, ndim=2] W):

    if i == 0:
        # Return tr(Pi_0) = tr(H^-1) = sum(1/(lam*eigenVals + 1))
        return np.sum(np.power(eigenVals, -1.0))

    else:
        return trace_Pi_recursive(lam, i-1, eigenVals, W) - compute_at_Pi_Pi_b(lam, i-1, eigenVals, W[:,0:(i-1)], W[:, (i-1):i], W[:, (i-1):i]) / compute_at_Pi_b(lam, i-1, eigenVals, W[:,0:(i-1)], W[:, (i-1):i], W[:, (i-1):i])

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
        W_i = W[:, (i-1):i]
        W_0_im1 = W[:,0:(i-1)]
        wi_Pim1_wi = compute_at_Pi_b(lam, i-1, eigenVals, W_0_im1, W_i, W_i)

        result += ((compute_at_Pi_Pi_b(lam, i-1, eigenVals, W_0_im1, W_i, W_i) / wi_Pim1_wi) ** 2.0)
        result -= 2.0 * compute_at_Pi_Pi_Pi_b(lam, i-1, eigenVals, W_0_im1, W_i, W_i) / wi_Pim1_wi
            
        i -= 1

    result += np.sum(np.power(eigenVals, -2.0))

    return result

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef joint_trace(float lam,
              int i,
              float[:] eigenVals,
              float[:,:] W):
    
    cdef float result_Pi, result_Pi_Pi = 0.0
    cdef float[:,:] W_i, W_0_im1
    cdef float wi_Pim1_wi, wi_Pim1_Pim1_wi, wi_Pim1_Pim1_Pim1_wi

    while i > 0:
        W_i = W[:, (i-1):i]
        W_i_sub = W[:, 0:(i-1)]
        wi_Pim1_wi = compute_at_Pi_b(lam, i-1, eigenVals, W_i_sub, W_i, W_i)
        wi_Pim1_Pim1_wi = compute_at_Pi_Pi_b(lam, i-1, eigenVals, W_i_sub, W_i, W_i)

        result_Pi_Pi += ((wi_Pim1_Pim1_wi / wi_Pim1_wi) ** 2.0)
        result_Pi_Pi -= 2.0 * compute_at_Pi_Pi_Pi_b(lam, i-1, eigenVals, W_i_sub, W_i, W_i) / wi_Pim1_wi
        result_Pi -=  wi_Pim1_Pim1_wi / wi_Pim1_wi
            
        i -= 1
        
    result_Pi += np.sum(np.power(eigenVals, -1.0))
    result_Pi_Pi += np.sum(np.power(eigenVals, -2.0))

    return np.array([result_Pi, result_Pi_Pi], dtype=np.float32)


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
    cdef float[:, :] inv, temp_vec, temp_vec2
    result = 0.0
    result2 = 0.0

    # a.T @ (H_inv - H_inv @ W @ (W.T @ H_inv @ W)^-1 @ W.T @ H_inv ) @ b

    # a.T @ H_inv @ b   
    result = np.sum(np.divide(np.multiply(a, b), eigenVals[:, None]))

    # H_inv @ W
    inv = np.divide(W, eigenVals[:, None])

    # W.T @ H_inv @ W
    inv = np.dot(W.T, inv)

    # (W.T @ H_inv @ W)^-1
    inv = np.linalg.inv(inv)

    # H_inv @ b
    temp_vec = b.copy()
    temp_vec = np.divide(temp_vec, eigenVals[:, None])

    # (H_inv @ a).T
    temp_vec2 = a.copy()
    temp_vec2 = np.divide(temp_vec2, eigenVals[:, None]).T

    # W.T @ H_inv @ b
    temp_vec = np.dot(W.T, temp_vec)

    # a.T @ H_inv @ W
    temp_vec2 = np.dot(temp_vec2, W)

    # (W.T @ H_inv @ W)^-1 @ W.T @ H_inv @ b
    temp_vec = np.dot(inv, temp_vec)

    # a.T @ H_inv @ (W @ (W.T @ H_inv @ W)^-1 @ W.T @ H_inv @ b)
    result2 = np.dot(temp_vec2, temp_vec)

    return result - result2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float compute_at_Pi_b_recursive(float lam,
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
        return compute_at_Pi_b_recursive(lam, i-1, eigenVals, W, a, b) - compute_at_Pi_b_recursive(lam, i-1, eigenVals, W, b, W_i) * compute_at_Pi_b_recursive(lam, i-1, eigenVals, W, a, W_i)/compute_at_Pi_b_recursive(lam, i-1, eigenVals, W, W_i, W_i) 

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

    # inv = W.copy()
    # for j in range(W.shape[0]):
    #     for k in range(W.shape[1]):
    #         inv[j, k] /= eigenVals[j]
    inv = np.divide(W, eigenVals[:, None])

    #inv = mat_mat_mult(W.T, inv)
    inv = np.dot(W.T, inv)

    #inv = invert_matrix(inv)
    inv = np.linalg.inv(inv)

    # temp_vec = b.copy()
    # temp_vec2 = a.copy()
    # for j in range(b.shape[0]):
    #     temp_vec[j, 0] /= eigenVals[j]
    #     temp_vec2[j, 0] /= eigenVals[j]

    temp_vec = np.divide(b, eigenVals[:, None])
    temp_vec2 = np.divide(a, eigenVals[:, None])

    #temp_vec = mat_vec_mult(W.T, temp_vec)
    temp_vec = np.dot(W.T, temp_vec)
    #temp_vec = mat_vec_mult(inv, temp_vec)
    temp_vec = np.dot(inv, temp_vec)
    #temp_vec = mat_vec_mult(W, temp_vec)
    temp_vec = np.dot(W, temp_vec)

    #temp_vec2 = mat_vec_mult(W.T, temp_vec2)
    temp_vec2 = np.dot(W.T, temp_vec2)
    #temp_vec2 = mat_vec_mult(inv, temp_vec2)
    temp_vec2 = np.dot(inv, temp_vec2)
    #temp_vec2 = mat_vec_mult(W, temp_vec2)
    temp_vec2 = np.dot(W, temp_vec2)

    # for j in range(temp_vec.shape[0]):
    #     temp_vec[j, 0] = (b[j, 0] - temp_vec[j, 0]) / eigenVals[j]
    #     temp_vec2[j, 0] = (a[j, 0] - temp_vec2[j, 0]) / eigenVals[j]
    #     result += temp_vec[j, 0] * temp_vec2[j, 0]

    temp_vec = np.divide(np.subtract(b,temp_vec), eigenVals[:, None])
    temp_vec2 = np.divide(np.subtract(a,temp_vec2), eigenVals[:, None])
    result += np.sum(np.multiply(temp_vec, temp_vec2))

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
    cdef float[:, :] inv, temp_vec, temp_vec2

    # inv = W.copy()
    # for j in range(W.shape[0]):
    #     for k in range(W.shape[1]):
    #         inv[j, k] /= eigenVals[j]
    inv = np.divide(W, eigenVals[:, None])

    #inv = mat_mat_mult(W.T, inv)
    inv = np.dot(W.T, inv)
    #inv = invert_matrix(inv)
    inv = np.linalg.inv(inv)

    # temp_vec = b.copy()
    # temp_vec2 = a.copy()
    # for j in range(b.shape[0]):
    #     temp_vec[j, 0] /= eigenVals[j]
    #     temp_vec2[j, 0] /= eigenVals[j]
    temp_vec = np.divide(b, eigenVals[:, None])
    temp_vec2 = np.divide(a, eigenVals[:, None])

    #temp_vec = mat_vec_mult(W.T, temp_vec)
    temp_vec = np.dot(W.T, temp_vec)
    #temp_vec = mat_vec_mult(inv, temp_vec)
    temp_vec = np.dot(inv, temp_vec)
    #temp_vec = mat_vec_mult(W, temp_vec)
    temp_vec = np.dot(W, temp_vec)

    #temp_vec2 = mat_vec_mult(W.T, temp_vec2)
    temp_vec2 = np.dot(W.T, temp_vec2)
    #temp_vec2 = mat_vec_mult(inv, temp_vec2)
    temp_vec2 = np.dot(inv, temp_vec2)
    #temp_vec2 = mat_vec_mult(W, temp_vec2)
    temp_vec2 = np.dot(W, temp_vec2)

    # for j in range(temp_vec.shape[0]):
    #     temp_vec[j, 0] = (b[j, 0] - temp_vec[j, 0]) / eigenVals[j]
    #     temp_vec2[j, 0] = (a[j, 0] - temp_vec2[j, 0]) / eigenVals[j]
    temp_vec = np.divide(np.subtract(b, temp_vec), eigenVals[:, None])
    temp_vec2 = np.divide(np.subtract(a, temp_vec2), eigenVals[:, None])

    # a.T @ H_inv @ b   
    result = np.sum(np.divide(np.multiply(temp_vec2, temp_vec), eigenVals[:, None]))

    # H_inv @ b
    temp_vec = np.divide(temp_vec, eigenVals[:, None])

    # (H_inv @ a).T
    temp_vec2 = np.divide(temp_vec2, eigenVals[:, None]).T

    # W.T @ H_inv @ b
    temp_vec = np.dot(W.T, temp_vec)

    # a.T @ H_inv @ W
    temp_vec2 = np.dot(temp_vec2, W)

    # (W.T @ H_inv @ W)^-1 @ W.T @ H_inv @ b
    temp_vec = np.dot(inv, temp_vec)

    # a.T @ H_inv @ (W @ (W.T @ H_inv @ W)^-1 @ W.T @ H_inv @ b)
    result2 = np.dot(temp_vec2, temp_vec)

    return result - result2

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