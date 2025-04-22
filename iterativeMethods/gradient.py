import numpy as np
from utilities.iterativeResult import IterativeResult
from utilities.common import verify_accuracy


def gradient_sparse(A_sparse, b, x0, tol, nmax):
    """
    Steepest Descent (gradient) method for solving sparse linear systems Ax = b

    Parameters:
    -----------
    A_sparse : sparse matrix
        The coefficient matrix (should be symmetric positive definite)
    b : ndarray
        Right-hand side vector
    x0 : ndarray
        Initial guess for the solution
    tol : float
        Tolerance for convergence
    nmax : int
        Maximum number of iterations

    Returns:
    --------
    IterativeResult object containing solution vector,
    iteration count, and error
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

    # Calculate final error using verify_accuracy with solution of all ones
    x_true = np.ones_like(x0)
    err = verify_accuracy(x0, x_true)

    return IterativeResult(x0, nit, err)
