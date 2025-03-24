import numpy as np

def factorization(A):
    M, N = A.shape
    if M != N:
        raise ValueError("Matrix must be square")
    
    A = A.copy()  
    L = np.eye(N)  #Identity matrix for L

    for n in range(N - 1):
        # If the pivot is too small execute partial pivoting
        if np.isclose(A[n, n], 0, atol=1e-12):
            max_row = np.argmax(np.abs(A[n:, n])) + n  #Find the row with the maximum value
            if A[max_row, n] == 0:
                raise ValueError("Singular matrix: LU decomposition not possible")
            #Swap rows
            A[[n, max_row], :] = A[[max_row, n], :]
            L[[n, max_row], :] = L[[max_row, n], :]

        for i in range(n + 1, N):
            L[i, n] = A[i, n] / A[n, n]
            A[i, n:] -= L[i, n] * A[n, n:]

    U = np.triu(A)
    return L, U

#Random matrix to test
np.random.seed(42)  
A = np.random.uniform(-1, 1, (5, 5))

#LU factorization
L, U = factorization(A)

print("Matrice A originale:\n", A)
print("Matrice L:\n", L)
print("Matrice U:\n", U)

#L*U must return A
reconstructed_A = np.dot(L, U)
error = np.linalg.norm(A - reconstructed_A)

print(f"Errore di ricostruzione ||A - LU||: {error:.2e}")
