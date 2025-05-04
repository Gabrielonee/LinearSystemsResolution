import numpy as np
from utilities.classes import IterativeResult


def conjugate_gradient_solver(A_sparse, b, x0, tol: float, nmax: int):
    """
    Risolve un sistema lineare Ax = b usando il metodo del gradiente coniugato
    (CG) per matrici simmetriche definite positive (SPD).

    Parameters
    ----------
    A_sparse : scipy.sparse matrix or ndarray
        Matrice dei coefficienti A in formato sparso o denso, deve essere
        simmetrica definita positiva.
    b : ndarray
        Vettore dei termini noti.
    x0 : ndarray
        Vettore iniziale della soluzione.
    tol : float
        Tolleranza per il criterio di arresto (errore relativo ||r|| / ||b||).
    nmax : int
        Numero massimo di iterazioni.

    Returns
    -------
    IterativeResult
        Oggetto contenente:
            - x : soluzione approssimata del sistema.
            - nit : numero di iterazioni eseguite.

    Note
    ----
    Il metodo è efficiente solo per matrici simmetriche definite positive.
    L'errore è valutato come: `err = ||r_k|| / ||b||`
    """

    # Initialize residual and search direction
    r = b - A_sparse @ x0
    p = r.copy()  # first search direction is equal to residue

    # Calculate initial error for convergence check
    norm_b = np.linalg.norm(b)  # useful to normalize the error

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
