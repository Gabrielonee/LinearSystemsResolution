import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import eigsh


def validate_matrix(matrix):
   #Validation: checking matrixs attrbiutes to check if fit into the methods
    if matrix is None:
        raise ValueError("Matrix is None.")

    if not hasattr(matrix, 'shape') or matrix.shape is None:
        raise ValueError("Matrix not valid or missing infos")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("La matrice non è quadrate. Impossibile procedere")

    #Simmetry 
    if isspmatrix(matrix):
        #For sparse matrix
        diff = matrix - matrix.T
        if diff.nnz != 0:
            raise ValueError("No simmetry")
    else:
        #For dense matrix
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError("No simmetry")

    #Positive defined
    try:
        # Minimum eigenvalue
        min_eig = eigsh(matrix, k=1, which='SA', return_eigenvectors=False)[0]
        if min_eig <= 0:
            raise ValueError("NOT SPD")
    except Exception as e:
        raise ValueError(
            f"Impossible to define the SPD: {e}")

    return rows
