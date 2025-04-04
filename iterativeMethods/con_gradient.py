import numpy as np
import time

def con_gradient(A, b, x0, tol, nmax):
    M, N = A.shape
    L = len(x0)

    #Check Matrix's property
    if M != N:
        raise ValueError("Matrix not square")
    elif L != M:
        raise ValueError("Matrix's size != x0 size")
    elif np.any(np.linalg.eigvals(A) <= 0):
        raise ValueError("Matrix not SPD")

    nit = 0
    err = 1
    xold = x0.copy()
    rold = b - A @ xold
    pold = rold.copy()

    start_time = time.time()
    while nit < nmax and err > tol:
        step = (pold.T @ rold) / (pold.T @ A @ pold)
        xnew = xold + step * pold
        rnew = rold - step * A @ pold
        beta = (A @ pold).T @ rnew / ((A @ pold).T @ pold)
        pnew = rnew - beta * pold
        err = np.linalg.norm(b - A @ xnew) / np.linalg.norm(xnew)
        xold = xnew
        rold = rnew
        pold = pnew
        nit += 1
    total_time = time.time() - start_time
    xk = xnew
    return xk, nit, total_time, err
