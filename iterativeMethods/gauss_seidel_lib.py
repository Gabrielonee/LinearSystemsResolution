import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from utils.classes import IterativeResult

def gauss_seidel_solver_library(A_sparse, b, x0, tol: float, nmax: int):
    converged = False
    
    #Lower triangular di A
    L = sp.tril(A_sparse).tocsc()
    x_new = x0.copy()
    #Calcolo norma di b
    norm_b = np.linalg.norm(b)

    for nit in range(nmax):
        #Calcolo del residuo
        res = b - A_sparse @ x_new
        #Calcolo della norma del residuo divisa per la norma di b
        temp = np.linalg.norm(res) / norm_b
        #Controllo della convergenza
        if temp < tol:
            converged = True
            break
        # Risoluzione del sistema triangolare inferiore
        x0 = spla.spsolve_triangular(L, res, lower=True)
        x_new += x0
    
    if not converged:
        print("Gauss-Seidel method did not converge within the maximum number of iterations.")
    
    return IterativeResult(x_new, nit + 1, converged)