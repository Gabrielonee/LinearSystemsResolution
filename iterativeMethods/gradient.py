import numpy as np
from utilities.classes import IterativeResult


def gradient_solver(A_sparse, b, x0, tol, nmax):
    """
    Metodo della discesa più ripida (gradiente) per la risoluzione di sistemi lineari sparsi Ax = b

    Parametri:
    -----------
    A_sparse : matrice sparsa
    La matrice dei coefficienti (dovrebbe essere simmetrica definita positiva)
    b : ndarray
    Vettore del lato destro
    x0 : ndarray
    Stima iniziale per la soluzione
    tol : float
    Tolleranza per la convergenza
    nmax : int
    Numero massimo di iterazioni

    Restituisce:
    --------
    Oggetto IterativeResult contenente:
    - Vettore della soluzione
    - Numero di iterazioni eseguite
    """
    # Calculate initial residual
    r = b - A_sparse @ x0

    # Calculate initial residual norm and right-hand side norm
    norm_r = np.linalg.norm(r)
    norm_b = np.linalg.norm(b)

    # Set relative tolerance !!!
    rel_tol = tol * norm_b if norm_b > 0 else tol

    # Initialize iteration counter
    nit = 0

    # Main iteration loop
    while norm_r > rel_tol and nit < nmax:
        # Calculate A*r
        Ar = A_sparse @ r

        # Calculate step size
        r_dot_r = np.dot(r, r)
        r_dot_Ar = np.dot(r, Ar)

        # Avoid division by zero
        if abs(r_dot_Ar) < 1e-14:
            break

        alpha = r_dot_r / r_dot_Ar

        # Update solution
        x0 = x0 + alpha * r

        # Recompute residual
        r = b - A_sparse @ x0
        norm_r = np.linalg.norm(r)

        # Increment iteration counter
        nit += 1

    return IterativeResult(x0, nit)
