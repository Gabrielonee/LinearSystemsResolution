import numpy as np
import scipy.sparse as sp
from utils.classes import IterativeResult
from directMethods.trian_inf import triang_inf

def gauss_seidel_solver(A_sparse, b, x0, tol: float, nmax: int):
    converged = False
    nit = 0
    
    # Forza il formato CSR per efficienza
    if not sp.issparse(A_sparse):
        A_sparse = sp.csr_matrix(A_sparse)
    elif not sp.isspmatrix_csr(A_sparse):
        A_sparse = A_sparse.tocsr()
    # Fattorizzazione: L = parte inferiore + diagonale, N = parte superiore stretta
    L = sp.tril(A_sparse)         # Triangolare inferiore incluso diagonale
    N = A_sparse - L              # Parte superiore stretta
    # Conversione per risoluzione efficiente del sistema triangolare
    L = L.tocsc()
    # Inizializzazione soluzione
    x_new = x0.copy()
    for nit in range(nmax):
        #Costruisce RHS come b - N * x_k
        rhs = b - N @ x_new           
        x_new = triang_inf(L, rhs)  # Risolve L x = rhs
        #Controlla la convergenza usando norma 2
        if np.linalg.norm(x_new - x0) < tol:
            converged = True
            break

        x0 = x_new.copy()  # Prepara per prossima iterazione
    if not converged:
        print("Metodo di Gauss non converge entro il numero massimo di iterazioni.")
    return IterativeResult(x_new, nit + 1, converged)
