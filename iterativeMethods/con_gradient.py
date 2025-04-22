import numpy as np
from utilities.iterativeResult import IterativeResult


def con_gradient_solver(A_sparse, b, x0, tol: float, nmax: int):
    """
    Conjugate Gradient method for solving sparse linear systems Ax = b

    Parameters:
    -----------
    A_sparse : sparse matrix
        The coefficient matrix (should be symmetric positive definite)
    b : ndarray
        Right-hand side vector
    x0 : ndarray
        Initial guess for the solution
    tol : float
        Relative tolerance for convergence
    nmax : int
        Maximum number of iterations

    Returns:
    --------
    IterativeResult object containing:
        - Solution vector
        - Number of iterations performed
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
