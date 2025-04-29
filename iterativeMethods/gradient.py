import numpy as np
from utilities.classes import IterativeResult

def gradient_solver(A_sparse, b, x0, tol, nmax):
    # Calculate initial residual
    r = b - A_sparse @ x0 #difference between the known term and the application of the matrix to the current estimate

    # Calculate initial residual norm and right-hand side norm
    norm_r = np.linalg.norm(r)
    norm_b = np.linalg.norm(b)
    # Set relative tolerance !!!
    rel_tol = tol * norm_b if norm_b > 0 else tol
    # Initialize iteration counter
    nit = 0

    # Main iteration loop until the residue norm is greater than the relative tolerance not exceeded the maximum number of iterations
    while norm_r > rel_tol and nit < nmax:
        # Calculate A*r
        Ar = A_sparse @ r
        # Calculate step size: scalar products 
        r_dot_r = np.dot(r, r) # alpha enumerator
        r_dot_Ar = np.dot(r, Ar) # alpha denominator
        
        # Avoid division by zero (if is it too small)
        if abs(r_dot_Ar) < 1e-14:
            break

        alpha = r_dot_r / r_dot_Ar # this one has the goal to define the optimal length of this step in the negative direction of gradient 
        # Update solution
        x0 = x0 + alpha * r
        
        # Recompute residual
        r = b - A_sparse @ x0
        norm_r = np.linalg.norm(r)
        
        # Increment iteration counter
        nit += 1
    return IterativeResult(x0, nit)
