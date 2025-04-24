def validate_matrix(matrix):
    if matrix is None:
        raise ValueError("Matrix is None.")
    if not hasattr(matrix, 'shape') or matrix.shape is None:
        raise ValueError("Matrix is invalid or missing shape info.")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Matrix is not square. Cannot proceed.")
    return rows
