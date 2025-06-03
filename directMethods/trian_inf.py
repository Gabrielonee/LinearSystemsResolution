import numpy as np
import scipy.sparse as sp

def triang_inf(L_sparse, b):
    """
    Solves the lower triangular system L x = b via forward substitution,
    optimized for sparse CSR matrices.

    Parameters
    ----------
    L_sparse : scipy.sparse.csr_matrix
        Sparse lower triangular matrix with non-zero diagonal elements.
    b : numpy.ndarray
        Right-hand side vector (1D).

    Returns
    -------
    x : numpy.ndarray
        Solution vector.
    """
    if not sp.issparse(L_sparse):
        raise ValueError("Matrix L must be in sparse format")
    if not sp.isspmatrix_csr(L_sparse):
        L_sparse = L_sparse.tocsr()
    n = L_sparse.shape[0]

    diag = L_sparse.diagonal()
    if np.any(np.abs(diag) < 1e-14):
        raise ValueError("Matrix L has zero or near-zero diagonal elements")

    x = np.zeros_like(b, dtype=np.float64)

    indptr = L_sparse.indptr    # CSR index pointer array
    indices = L_sparse.indices  # CSR column indices array
    data = L_sparse.data        # CSR non-zero values array

    for i in range(n):
        sum_val = 0.0
        start = indptr[i]
        end = indptr[i+1]

        # Iterate over non-zero elements of the i-th row up to the diagonal (column < i)
        for idx in range(start, end):
            j = indices[idx]
            if j >= i:
                break
            sum_val += data[idx] * x[j]

        # Compute x[i] using forward substitution formula
        x[i] = (b[i] - sum_val) / diag[i]

    return x