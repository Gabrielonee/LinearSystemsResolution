import numpy as np
from utils.classes import IterativeResult


def gradient_solver(A_sparse, b, x0, tol, nmax):
    """
    Solve the linear system Ax = b using the Gradient Method (Steepest Descent).

    Parameters
    ----------
    A_sparse : scipy.sparse matrix
        Sparse coefficient matrix (assumed symmetric positive definite).
    b : numpy.ndarray
        Right-hand side vector.
    x0 : numpy.ndarray
        Initial guess vector.
    tol : float
        Convergence tolerance for relative residual.
    nmax : int
        Maximum number of iterations allowed.

    Returns
    -------
    IterativeResult
        Object containing solution vector, number of iterations, and convergence status.
    """
    converged = False

    # Compute initial residual r = b - A*x0
    r = b - A_sparse @ x0
    
    # Norms for convergence check
    norm_r = np.linalg.norm(r)
    norm_b = np.linalg.norm(b)
    
    # Relative tolerance adjusted by norm of b to avoid scaling issues
    rel_tol = tol * norm_b if norm_b > 0 else tol
    
    nit = 0
    
    # Iterative loop until convergence or max iterations
    while norm_r > rel_tol and nit < nmax:
        # Compute A*r (needed for step size denominator)
        Ar = A_sparse @ r
        
        # Compute numerator and denominator for step size alpha_k
        r_dot_r = np.dot(r, r)
        r_dot_Ar = np.dot(r, Ar)
        
        # Avoid division by zero or very small denominator
        if abs(r_dot_Ar) < 1e-14:
            break
        
        # Compute step size alpha_k
        alpha = r_dot_r / r_dot_Ar
        
        # Update solution: x_{k+1} = x_k + alpha_k * r_k
        x0 = x0 + alpha * r
        
        # Update residual: r_{k+1} = b - A * x_{k+1}
        r = b - A_sparse @ x0
        
        # Update residual norm
        norm_r = np.linalg.norm(r)
        
        nit += 1

    # Check convergence status
    if nit >= nmax:
        print("Gradient method did not converge within maximum iterations.")
    else:
        converged = True

    return IterativeResult(x0, nit, converged)