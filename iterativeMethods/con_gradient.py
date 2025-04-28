import numpy as np
from utilities.classes import IterativeResult


def conjugate_gradient_solver(A_sparse, b, x0, tol: float, nmax: int):
    """
    Metodo del gradiente coniugato per la risoluzione di sistemi lineari sparsi

    Parametri:
    -----------
    A_sparse : matrice sparsa
    La matrice dei coefficienti (dovrebbe essere simmetrica definita positiva)
    b : ndarray
    Vettore del lato destro
    x0 : ndarray
    Valore iniziale per la soluzione
    tol : float
    Tolleranza relativa per la convergenza
    nmax : int
    Numero massimo di iterazioni

    Restituisce:
    --------
    Oggetto IterativeResult contenente:
    - Vettore della soluzione
    - Numero di iterazioni eseguite
    """
    # Initialize residual and search direction
    r = b - A_sparse @ x0
    p = r.copy()

    # Calculate initial error for convergence check
    norm_b = np.linalg.norm(b)
    err = np.linalg.norm(r) / norm_b if norm_b > 0 else np.linalg.norm(r)

    # Iteration counter
    nit = 0

    # Store the first r dot product
    r_dot_r = np.dot(r, r)

    while nit < nmax and err > tol:
        Ap = A_sparse @ p

        # Calculate step size
        alpha = r_dot_r / np.dot(p, Ap)

        # Update solution
        x0 = x0 + alpha * p

        # Update residual
        r_new = r - alpha * Ap

        # Calculate error for convergence check
        err = np.linalg.norm(r_new) / \
            norm_b if norm_b > 0 else np.linalg.norm(r_new)

        # Calculate beta using Fletcher-Reeves formula
        r_dot_r_new = np.dot(r_new, r_new)
        beta = r_dot_r_new / r_dot_r

        # Update search direction
        p = r_new + beta * p

        # Update residual and its dot product for next iteration
        r = r_new
        r_dot_r = r_dot_r_new

        # Increment iteration counter
        nit += 1

    # Return result
    return IterativeResult(x0, nit)
