import numpy as np
import time

def jacobi(A, b, x0, tol, nmax): 
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
    
    D = np.diag(np.diag(A)) #diagonal
    B = D - A
    
    xold = x0.copy()
    xnew = xold + 1 
    nit = 0 #iteration number
    
    start_time = time.time()
    
    while np.linalg.norm(xnew - xold, np.inf) > tol and nit < nmax: #if over thresold or nmax
        xold = xnew.copy()
        xnew = np.linalg.inv(D) @ (B @ xold + b) #iterate the computing of solution
        nit += 1
    
    elapsed_time = time.time() - start_time
    err = np.linalg.norm(xnew - xold, np.inf)
    
    return xnew, nit, elapsed_time, err