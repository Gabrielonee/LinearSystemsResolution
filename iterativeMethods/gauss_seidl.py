import numpy as np
import time
from directMethods import triang_inf

def fixed_gauss_seidl(A, b, x0, tol, nmax):
    M, N = A.shape
    L = x0.shape[0]
    
    if M != N:
        print('Matrix A has to be square')
        return None, None, None, None
    elif L != M:
        print('shape(A) != shape(x0)')
        return None, None, None, None
    
    if np.any(np.diag(A) == 0):
        print('At least a diagonal entry is zero')
        return None, None, None, None
    
    L_matrix = np.tril(A) #Lower triangular
    B = A - L_matrix
    xold = x0.copy()
    xnew = x0.copy() 
    nit = 0

    start_time = time.time()
    while nit < nmax:
        xold = xnew.copy()
        xnew = triang_inf(L_matrix, b - B @ xold)
        if np.linalg.norm(xnew - xold, np.inf) < tol:
            break
        nit += 1

    elapsed_time = time.time() - start_time
    err = np.linalg.norm(b - A @ xnew) / np.linalg.norm(xnew)
    return xnew, nit, elapsed_time, err