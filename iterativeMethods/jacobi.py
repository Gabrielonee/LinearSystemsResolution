import scipy.sparse as sp
import numpy as np
from utilities.classes import IterativeResult


def jacobi_solver(A_sparse, b, x0, tol: float, nmax: int):

    converged = False
    nit = 0

    # Estrai la diagonale principale della matrice A
    D = A_sparse.diagonal()
    
    # Verifica che la diagonale non contenga zeri o quasi zeri
    if np.any(np.abs(D) < 1e-10):
        raise ValueError("Matrice con zeri (o quasi) sulla diagonale: Jacobi non applicabile.")

    # Calcola l'inverso degli elementi diagonali
    D_inv = 1.0 / D

    # Inizializza il vettore soluzione con il vettore iniziale x0
    x = x0.copy()

    # Calcola la norma 2 del vettore b per il residuo relativo
    norm_b = np.linalg.norm(b)
    
    for nit in range(nmax):
        # Calcola il residuo: r = b - A*x
        res = b - A_sparse @ x
        
        # Calcola il residuo relativo
        res_rel = np.linalg.norm(res) / norm_b
        
        # Verifica il criterio di arresto
        if res_rel < tol:
            converged = True
            return IterativeResult(x, nit + 1, converged)
        
        # Aggiorna la soluzione secondo Jacobi: x^{k+1} = x^k + D^{-1} * (b - A*x^k)
        x = x + D_inv * res

    print("Metodo di Jacobi non converge entro il numero massimo di iterazioni.")

    # Restituisce la soluzione e il numero di iterazioni eseguite
    return IterativeResult(x, nit + 1, converged)