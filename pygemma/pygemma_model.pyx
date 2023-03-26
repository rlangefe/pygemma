import numpy as np
cimport numpy as np

def compute_Pc(
                np.ndarray[np.float32_t, ndim=1] eigenVals, 
                np.ndarray[np.float32_t, ndim=2] U, 
                np.ndarray[np.float32_t, ndim=1] W, 
                double lam
                ) -> np.ndarray[np.float32_t, ndim=2]:
    cdef np.ndarray[np.float32_t, ndim=2] H_inv = U.T @ np.diagflat(1/(lam*eigenVals + 1.0)) @ U

    return H_inv - H_inv @ W @ np.linalg.inv(W.T @ H_inv @ W) @ W.T @ H_inv


