import numpy as np

def triang_inf(L, b):
    """
    Solve lower triangular system L*x = b
    """
    M, N = L.shape
    x = np.zeros(M)
    
    if M != N:
        print('Matrix L is not a square matrix')
        return None
    elif np.sum(np.abs(L - np.tril(L))) > 1e-15:
        print('Matrix L is not a lower triangular matrix')
        return None
    else:
        x[0] = b[0] / L[0, 0]
        for i in range(1, N):
            x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    
    return x