import numpy as np
from utils.classes import IterativeResult

def conjugate_gradient_solver(A_sparse, b, x0, tol: float, nmax: int):
    converged = False
    
    r = b - A_sparse @ x0
    p = r.copy()
    norm_b = np.linalg.norm(b)
    err = np.linalg.norm(r) / norm_b if norm_b > 0 else np.linalg.norm(r)
    nit = 0
    r_dot_r = np.dot(r, r)
    
    while nit < nmax and err > tol:
        Ap = A_sparse @ p
        denom = np.dot(p, Ap)
        if abs(denom) < 1e-14:  # Avoid division by zero or near-zero
            print("Warning: denominator close to zero in alpha calculation.")
            break
        
        alpha = r_dot_r / denom
        x0 = x0 + alpha * p
        r_new = r - alpha * Ap
        
        err = np.linalg.norm(r_new) / norm_b if norm_b > 0 else np.linalg.norm(r_new)
        
        r_dot_r_new = np.dot(r_new, r_new)
        beta = r_dot_r_new / r_dot_r
        
        p = r_new + beta * p
        
        r = r_new
        r_dot_r = r_dot_r_new
        nit += 1
    
    if err <= tol:
        converged = True
    else:
        print(f"Conjugate Gradient did not converge within {nmax} iterations. Final error: {err:.2e}")
    
    return IterativeResult(x0, nit, converged)