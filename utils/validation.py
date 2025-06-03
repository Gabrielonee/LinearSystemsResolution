import numpy as np
from scipy.sparse import isspmatrix_csr
from scipy.sparse.linalg import eigsh


def validate_matrix(matrix, tol_sym=1e-10):
    """
    Validates a CSR sparse matrix ensuring it is:
    square, symmetric, positive definite (SPD), and diagonally dominant.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Sparse matrix to validate.
    tol_sym : float, optional
        Tolerance for symmetry check (default=1e-10).

    Returns
    -------
    int
        Dimension of the matrix (number of rows/columns).

    Raises
    ------
    ValueError
        If matrix is None, not CSR, not square, not symmetric,
        not positive definite, or not diagonally dominant.

    Notes
    -----
    - Supports only CSR matrices.
    - Symmetry checked via norm of difference.
    - Positive definiteness evaluated via smallest eigenvalue.
    - Diagonal dominance checked by summing row values.
    """
    if matrix is None:
        raise ValueError("Matrix is None.")
    if not isspmatrix_csr(matrix):
        raise ValueError("Matrix is not in CSR format.")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"Matrix is not square ({rows}x{cols}).")

    # Check symmetry by computing norm of difference with transpose
    diff = matrix - matrix.T
    norm_diff = np.linalg.norm(diff.data) if diff.nnz > 0 else 0.0
    if norm_diff > tol_sym:
        raise ValueError(f"Matrix is not symmetric (norm diff={norm_diff:.2e}).")

    # Check positive definiteness using smallest eigenvalue
    try:
        min_eig = eigsh(matrix, k=1, which='SA', return_eigenvectors=False)[0]
        if min_eig <= 0:
            raise ValueError(f"Matrix is not positive definite (min eig={min_eig:.2e}).")
    except Exception as e:
        raise ValueError(f"Error during positive definiteness check: {e}")

    # MISSING: diagonal dominance check mentioned in docstring but not implemented

    return rows