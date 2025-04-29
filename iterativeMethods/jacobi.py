import scipy.sparse as sp
import numpy as np
from utilities.classes import IterativeResult

def jacobi_solver(A_sparse, b, x0, tol: float, nmax: int):
    #A = P - N 
    #P = D
    #N = L+U

    # Extract diagonal elements
    P = A_sparse.diagonal()

    # Check for zeros in the diagonal (which would cause division by zero)
    if np.any(np.abs(P) < 1e-10):
        raise ValueError(
            "Matrix has zeros on the diagonal -\
            Jacobi method cannot be applied")

    # Compute N = A - P --> N = L + U
    N = A_sparse - sp.diags(P)

    # Initialize iteration counter and solution vector
    nit = 0
    x_new = x0.copy()  # Create a copy to avoid modifying the input

    # Perform iterations
    for nit in range(nmax):
        x_new = (b - N @ x0) / P
        # Check convergence
        if np.linalg.norm(x_new - x0, np.inf) < tol:
            x0 = x_new.copy()
            break
        x0 = x_new.copy()
    return IterativeResult(x0, nit + 1)
