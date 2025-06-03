import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from utils.classes import IterativeResult

def gauss_seidel_solver_library(A_sparse, b, x0, tol: float, nmax: int):
    converged = False
    
    #Si suddivide A in L e N, dove L è la parte triangolare inferiore (inclusa la diagonale) e 
    #N è la parte triangolare superiore
    L = sp.tril(A_sparse)  
    N = A_sparse - L       
    #Converte L in formato CSC per rendere efficiente la risoluzione di sistemi triangolari
    L = L.tocsc()
    nit = 0
    x_new = x0.copy()
    for nit in range(nmax):
        #Aggiornamento del termine noto secondo la formula di Gauss-Seidel
        rhs = b - N @ x_new
        #Si risolve il sistema triangolare inferiore Lx = rhs
        x_new = spla.spsolve_triangular(L, rhs, lower = True)
        #Controlla la convergenza con norma due tra iterati successivi
        if np.linalg.norm(x_new - x0) < tol:
            converged = True
            break
        #Aggiornamento soluzione
        x0 = x_new.copy()

    #Ritorna la soluzione e il numero di iterazioni eseguite
    if not converged:
        print("Metodo Gauss Seidel non converge entro il numero massimo di iterazioni.")
    return IterativeResult(x0, nit + 1, converged)

