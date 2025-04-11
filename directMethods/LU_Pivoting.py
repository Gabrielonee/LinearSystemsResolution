import numpy as np

def factorization_LU_pivoting(A):
    N = A.shape[0]
    A_old = A.copy() 
    M = np.eye(N)     #Matrix to cumulate trasnformations
    P = np.eye(N)     #Matrix of total permutation

    for n in range(N - 1):
        #Check if column is zero
        if np.sum(np.abs(A_old[n+1:, n])) < 1e-14:
            A_new = A_old
        else:
            #Find the index of the maximum value in abs
            pos = np.argmax(np.abs(A_old[n+1:, n])) + (n + 1)
            #Building P
            Pn = np.eye(N)
            Pn[[n+1, pos], :] = Pn[[pos, n+1], :]  #Swap rows
            P = Pn @ P
            A_old = Pn @ A_old
            #Computing elimination matrices
            Mn = np.eye(N)
            Mn_inv = np.eye(N)
            Mn[n+1:, n] = -A_old[n+1:, n] / A_old[n, n]
            Mn_inv[n+1:, n] = A_old[n+1:, n] / A_old[n, n]

            M = Mn @ Pn @ M
            A_old = Mn @ A_old

    U = A_old
    L = np.linalg.inv(M) @ P 

    return P, L, U