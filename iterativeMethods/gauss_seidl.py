import numpy as np
import time
from directMethods import triang_inf

def gauss_seidl(A, b, x0, tol,nmax):
    M, N = A.shape
    L = x0.shape[0]
    
    if M != N:
        print('Matrix A is not a square matrix')
        return None, None, None, None
    elif L != M:
        print('Dimensions of matrix A does not match dimension of initial guess x0')
        return None, None, None, None
    
    if np.any(np.diag(A) == 0):
        print('At least a diagonal entry is zero. The method automatically fails')
        return None, None, None, None
    
    L = np.tril(A) #lower triangular
    B = A-L
    xold = x0.copy()
    xnew = xold + 1 
    nit = 0

    start_time = time.time()
    while np.lianlg.norm(xnew-xold, np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        xnew = triang_inf(L, (b-B@ xold))
        nit += 1

    elapsed_time = start_time - time.time()
    err = np.linalg.norm(xnew - xold, np.inf)
    return xnew, nit, elapsed_time, err