import numpy as np
import pandas as pd

try:
    from rich.progress import track, Progress
    from rich.console import Console
except ModuleNotFoundError:
    print('Issues with rich for progress tracking')
    
import multiprocessing
from scipy import optimize, stats

import time

def calc_lambda(eigenVals, Y, W):
    # Loop over intervals and find where likelihood changes signs with respect to lambda
    step = 1.0

    lambda_pow_low = -5.0
    lambda_pow_high = 5.0

    lambda_possible = [(np.power(10.0, i), np.power(10.0, i+step)) for i in np.arange(lambda_pow_low,lambda_pow_high,step)]

    roots = [np.power(10.0, lambda_pow_low), np.power(10.0, lambda_pow_high)]
    
    for lambda0, lambda1 in lambda_possible:
    
        likelihood_lambda0 = likelihood_derivative1_lambda(lambda0, eigenVals, Y, W)
        likelihood_lambda1 = likelihood_derivative1_lambda(lambda1, eigenVals, Y, W)

        if np.sign(likelihood_lambda0) * np.sign(likelihood_lambda1) < 0:
            lambda_min = optimize.brentq(f=lambda l: likelihood_derivative1_lambda(l, eigenVals, Y, W), 
                                                a=lambda0, 
                                                b=lambda1,
                                                rtol=0.1,
                                                maxiter=5000,
                                                disp=False)
            
            lambda_min = optimize.newton(func=lambda l: likelihood_derivative1_lambda(l, eigenVals, Y, W), 
                                        x0=lambda_min,
                                        rtol=1e-5,
                                        fprime=lambda l: likelihood_derivative2_lambda(l, eigenVals, Y, W),
                                        maxiter=10,
                                        disp=False)
            
            roots.append(lambda_min)
            


    likelihood_list = [likelihood_lambda(lam, eigenVals, Y, W) for lam in roots]

    return roots[np.argmax(likelihood_list)]


def pygemma(Y, X, W, K, snps=None, verbose=0, nproc=1):
    if True: # Fix compatability issues with Python<=3.6.9 later (issue with rich package)
        console = Console()
    else:
        verbose = 0

    Y = Y.astype(np.float32).reshape(-1,1)
    W = W.astype(np.float32)
    X = X.astype(np.float32)

    Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    #K = ((K - np.mean(K, axis=0)) / np.std(K, axis=0)).astype(np.float32)
    #K = (K - K.mean(axis=1)[:, None]) / K.std(axis=1)[:, None]

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

            start = time.time()
            for col in range(W.shape[1]):
                if W[:,col].std() > 0:
                    W[:,col] = (W[:,col] - W[:,col].mean()) / W[:,col].std()
            console.log(f"[green]Covariate matrix centered - {round(time.time() - start,3)} s")

            console.log(f"[green]Running {X.shape[1]} SNPs with {Y.shape[0]} individuals...")

            # Calculate under null
            n, c = W.shape

            # start = time.time()
            # lambda_null = calc_lambda(eigenVals, U @ Y, U @ W)
            # console.log(f"[green]Null lambda computed: {round(lambda_null, 5)} - {round(time.time() - start,3)} s")

            # start = time.time()
            # Pc = compute_Pc(eigenVals, U, W, lambda_null)
    
            # Wt_Pc = W.T @ Pc
            # beta_vec_null = np.linalg.inv(Wt_Pc @ W) @ (Wt_Pc @ Y)
            # tau_null = float(n / (Y.T @ Pc @ Y))
            # console.log(f"[green]Null tau computed: {round(tau_null, 5)} - {round(time.time() - start,3)} s")

            # start = time.time()
            # l_null = likelihood(lambda_null, tau_null, beta_vec_null, eigenVals, U @ Y, U @ W)
            # console.log(f"[green]Null likelihood computed: {round(l_null, 5)} - {round(time.time() - start,3)} s")
    else:
        eigenVals, U = np.linalg.eig(K) # Perform eigendecomposition
        eigenVals = eigenVals.astype(np.float32)
        U = U.astype(np.float32)

        eigenVals = np.maximum(0, eigenVals)

        assert (eigenVals >= 0).all()

        X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

        for col in range(W.shape[1]):
            if W[:,col].std() > 0:
                W[:,col] = (W[:,col] - W[:,col].mean()) / W[:,col].std()

        # Calculate under null
        n, c = W.shape

        #lambda_null = calc_lambda(eigenVals, Y, W)

        # Pc = compute_Pc(eigenVals, U, W, lambda_null)
    
        # Wt_Pc = W.T @ Pc
        # beta_vec_null = np.linalg.inv(compute_at_Pi_b(lam, W.shape[1],
        #                                                 lam*eigenVals + 1.0,
        #                                                 U @ W,
        #                                                 U @ W,
        #                                                 U @ W)) @ (compute_at_Pi_b(lam, W.shape[1],
        #                                                                             lam*eigenVals + 1.0,
        #                                                                             U @ W,
        #                                                                             U @ W,
        #                                                                             U @ Y))
        # tau_null = float(n / compute_at_Pi_b(lam, W.shape[1],
        #               lam*eigenVals + 1.0,
        #               U @ W,
        #               U @ Y,
        #               U @ Y))

        # l_null = likelihood(lambda_null, tau_null, beta_vec_null, eigenVals, Y, W)

    X = U @ X
    Y = U @ Y
    W = U @ W

    if verbose > 0:
        #progress_bar = track(range(X.shape[1]), description='Testing SNPs...') #Uncomment later for good visualization and timing
        progress_bar = range(X.shape[1])
    else:
        progress_bar = range(X.shape[1])

    # for g in progress_bar:
    #     try:
    #         lambda_restricted = calc_lambda_restricted(eigenVals, Y, np.c_[W, X[:,g]])
    #         beta, beta_vec, se_beta, tau = calc_beta_vg_ve_restricted(eigenVals, W, X[:,g:(g+1)], lambda_restricted, Y)

    #         F_wald = np.power(beta/se_beta, 2.0)

    #         #lambda_alt = calc_lambda(eigenVals, Y, np.c_[W, X[:,g]])
    #         #_, beta_vec, _, tau_lrt = calc_beta_vg_ve(eigenVals, W, X[:,g], lambda_alt, Y)

    #         # #Fix these calculations later
    #         #l_alt = likelihood(lambda_alt, tau_lrt, beta_vec, eigenVals, Y, np.c_[W, X[:,g]])
    #         #D_lrt = 2 * (l_alt - l_null)
    #     except np.linalg.LinAlgError as e:
    #         beta = np.nan
    #         se_beta = np.nan
    #         tau = np.nan
    #         lambda_restricted = np.nan
    #         F_wald = np.nan
    #         D_lrt = np.nan

    #     # Store values
    #     results_dict['beta'].append(beta)
    #     results_dict['se_beta'].append(se_beta)
    #     results_dict['tau'].append(tau)
    #     results_dict['lambda'].append(lambda_restricted)
    #     results_dict['F_wald'].append(F_wald)
    #     results_dict['p_wald'].append(1-stats.f.cdf(x=F_wald, dfn=1, dfd=n-c-1))

    #     #results_dict['D_lrt'].append(D_lrt)
    #     #results_dict['p_lrt'].append(1-stats.chi2.cdf(x=D_lrt, df=1))

    #     #l_alt = likelihood_restricted(lambda_restricted, tau, eigenVals, U, Y, np.c_[W, X[:,g]])
    #     #results_dict['likelihood'].append(l_alt)


    # results_df = pd.DataFrame.from_dict(results_dict)

    # TODO: Apparently, we can use mpi4py to parallelize this
    # Let's actually do that at some point
    '''
    from mpi4py import MPI
    from multiprocessing import Pool

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    def my_function(x):
        # some code here

    if __name__ == '__main__':
        pool = None
        if rank == 0:
            pool = Pool(processes=size-1)
        data = comm.scatter(SampleIter(X, Y, W, eigenVals), root=0)
        result = my_function(data)
        results = comm.gather(result, root=0)
        if rank == 0:
            results_df = pd.DataFrame.from_dict(results)
    '''
    with multiprocessing.Pool(nproc) as pool:
        total = X.shape[1]

        #progress = Progress()
        #task = progress.add_task("Testing SNPs...", total=total)
        #progress.update(task, completed=0)

        #result_iterator = pool.starmap_async(calculate, SampleIter(X, Y, W, eigenVals))
        
        results = []
        #while not result_iterator.ready():

        done = 0
        for r in track(pool.imap(calculate, SampleIter(X, Y, W, eigenVals)), description='Testing SNPs...', total=total):
            # update the progress bar
            done += 1
            #progress.update(task, completed=done)
            
            results.append(r)
            

        results_df = pd.DataFrame.from_dict(results)
    

    if snps is not None:
        results_df['SNPs'] = snps

    return results_df

class SampleIter:
    def __init__ (self, X, Y, W, eigenVals):
        self.X = X
        self.Y = Y
        self.W = W
        self.eigenVals = eigenVals
        self.g = 0
        self.c = X.shape[1]
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.g < self.c:
            self.g += 1
            return (self.eigenVals, self.Y, self.W, self.X[:,self.g-1])
        else:
            raise StopIteration

        
def calculate(t):
    eigenVals, Y, W, X = t
    try:
        lambda_restricted = calc_lambda_restricted(eigenVals, Y, np.c_[W, X])
        beta, beta_vec, se_beta, tau = calc_beta_vg_ve_restricted(eigenVals, W, X.reshape(-1,1), lambda_restricted, Y)
        F_wald = np.power(beta/se_beta, 2.0)

        n = Y.shape[0]
        c = W.shape[1]

        return {
            'beta': beta,
            'se_beta': se_beta,
            'tau': tau,
            'lambda': lambda_restricted,
            'F_wald': F_wald,
            'p_wald': 1-stats.f.cdf(x=F_wald, dfn=1, dfd=n-c-1),
        }
    except np.linalg.LinAlgError as e:
        return {
            'beta': np.nan,
            'se_beta': np.nan,
            'tau': np.nan,
            'lambda': np.nan,
            'F_wald': np.nan,
            'p_wald': np.nan,
        }

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


