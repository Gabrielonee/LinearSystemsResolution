import numpy as np
import time

def gradient(A, b, x0, tol=1e-6, nmax=1000):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A is not a square matrix")
    if x0.shape[0] != A.shape[0]:
        raise ValueError("Dimensions of matrix A does not match dimension of initial guess x0")
    
    #Matrix SPD
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A has to be symmestric")
    if np.any(np.linalg.eigvals(A) <= 0):
        raise ValueError("Matrix A has to be SPD")
    
    xk = x0.copy()
    residue = b - A @ xk
    nit = 0
    start_time = time.time()
    
    while np.linalg.norm(residue) > tol and nit < nmax:
        alpha = (residue.T @ residue) / (residue.T @ A @ residue)
        xk = xk + alpha * residue
        residue = b - A @ xk
        nit += 1
    
    end_time = time.time()
    err = np.linalg.norm(b - A @ xk) / np.linalg.norm(xk)
    return xk, nit, end_time - start_time, err