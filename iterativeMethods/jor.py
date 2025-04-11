import time
import numpy as np

def fixed_jor(A, b, x0, tol, nmax, omega):
    M, N = A.shape
    L = x0.shape[0]
    
    if M != N:
        print('Matrix A is not a square matrix')
        return None, None, None, None
    elif L != M:
        print('Dimensions of matrix A does not match dimension of initial guess x0')
        return None, None, None, None
    
    D = np.diag(np.diag(A))
    B = A - D  # (L+U)
    xold = x0.copy()
    xnew = x0.copy()
    nit = 0  # iteration number

    start_time = time.time()
    while nit < nmax:
        xold = xnew.copy()
        xnew = np.linalg.solve(D, b - B @ xold)  # D^-1 * (b - B * xold)
        xnew = omega * xnew + (1 - omega) * xold
        
        if np.linalg.norm(xnew - xold, np.inf) < tol:
            break
        nit += 1
    
    elapsed_time = time.time() - start_time
    err = np.linalg.norm(b - A @ xnew) / np.linalg.norm(xnew)
    return xnew, nit, elapsed_time, err