import time
from tracemalloc import start
import numpy as np

def jor(A,b, x0, tol, nmax, omega):
    #B = (A-D)
    #b = D^-1 * f
    #f = x - Bx
    M,N = A.shape
    L = x0.shape[0]
    if M != N:
        print('Matrix A is not a square matrix')
        return None, None, None, None
    elif L != M:
        print('Dimensions of matrix A does not match dimension of initial guess x0')
        return None, None, None, None
    D = np.diag(np.diag(A))
    B = A-D #(L+U)
    xold = x0.copy()
    xnew = xold + 1 
    nit = 0 #iteration number

    while np.linalg.norm(xnew -xold, np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        #D^-1 * (D^-1 * f - )
        xnew = np.linalg.inv(D) @ (b - B @ xold)
        xnew = omega * xnew + (1- omega) * xold
        nit += 1
    elapsed_time = time.time() - start.time()
    err = np.linalg.norm(xnew -xold, np.inf)

    return xnew,nit,elapsed_time,err