import numpy as np
import scipy.sparse as sp
from utilities.classes import IterativeResult
from directMethods.trian_inf import triang_inf

def gauss_seidel_solver(A_sparse, b, x0, tol: float, nmax: int):
    """
    Risolve il sistema lineare Ax = b usando il metodo iterativo di
    Gauss-Seidel per matrici sparse.

    Parameters
    ----------
    A_sparse : scipy.sparse matrix
        Matrice dei coefficienti A in formato sparso, preferibilmente quadrata
        e a diagonale dominante.
    b : ndarray
        Vettore dei termini noti.
    x0 : ndarray
        Vettore iniziale della soluzione.
    tol : float
        Tolleranza per il criterio di arresto basato sulla norma infinito
        della differenza tra iterazioni successive.
    nmax : int
        Numero massimo di iterazioni consentite.

    Returns
    -------
    IterativeResult
        Oggetto contenente:
            - x : soluzione approssimata del sistema.
            - nit : numero di iterazioni eseguite (1-based).

    Note
    ----
    La matrice A dovrebbe essere almeno debolmente diagonalmente dominante per
    garantire convergenza.
    """
    # Dimensione del sistema
    n = A_sparse.shape[0]
    # Estrai le parti della matrice A
    D = sp.spdiags(A_sparse.diagonal(), 0, n, n)  # Diagonale
    L = sp.tril(A_sparse, k=-1)                   # Triangolare inferiore stretta
    U = sp.triu(A_sparse, k=1)                    # Triangolare superiore stretta
    # La parte da invertire è D+L
    DL = D + L
    # Initialize solution vector
    x = x0.copy()
    # Iterative solution
    for it in range(1, nmax+1):
        x_old = x.copy()  # Salva la soluzione precedente per il test di convergenza
        # Calcola il termine noto per questa iterazione: b - U*x
        rhs = b - U @ x
        #(D+L)x = rhs
        x = triang_inf(DL.toarray(), rhs)
        # Check convergence using infinity norm
        error = np.linalg.norm(x - x_old, np.inf)
        print(f"Iteration {it}: x = {x}, error = {error}") #debug 
        if error < tol:
            return IterativeResult(x, it)
    return IterativeResult(x, nmax)