import numpy as np

import pygemma

# Seed tests
np.random.seed(42)

# Initializing parameters for tests
n = 100
covars = 10
K = np.random.uniform(size=(n, n))
K = np.abs(np.tril(K) + np.tril(K, -1).T)
K = np.dot(K, K.T)
eigenVals, U = np.linalg.eig(K)
W = np.random.rand(n, covars)
x = np.random.rand(n, 1)
Y = np.random.rand(n, 1)
lam = 5

# Testing compute_Px
Px = pygemma.compute_Px(eigenVals, U, W, x, lam)
print("Passed compute_Px")

likelihood = pygemma.likelihood_lambda(lam, eigenVals, U, Y, W, x)
print("Passed likelihood")

likelihood_deriv1 = pygemma.likelihood_derivative1_lambda(lam, eigenVals, U, Y, W, x)
print("Passed likelihood first derivative")

likelihood_deriv2 = pygemma.likelihood_derivative2_lambda(lam, eigenVals, U, Y, W, x)
print("Passed likelihood second derivative")

