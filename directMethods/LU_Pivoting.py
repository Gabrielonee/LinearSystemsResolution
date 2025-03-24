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

def test_factorization_LU_pivoting():
    np.set_printoptions(precision=4, suppress=True) 
    
    test_cases = {
        "Random Matrix": np.random.uniform(-1, 1, (5, 5)),  
        "Diagonal Matrix": np.diag([1, 2, 3, 4, 5]),  
        "Hilbert Matrix": np.array([[1 / (i + j + 1) for j in range(5)] for i in range(5)]),
        "Singular Matrix": np.array([[1, 2, 3], [2, 4, 6], [1, 5, 7]]) 
    }

    for name, A in test_cases.items():
        print(f"\n🔹 Test: {name}")
        try:
            P, L, U = factorization_LU_pivoting(A)
            print("P:\n", P)
            print("L:\n", L)
            print("U:\n", U)

            reconstructed_A = P @ A
            LU = L @ U
            error = np.linalg.norm(reconstructed_A - LU)

            print(f"Reconstruction error ||P A - LU||: {error:.2e}")

            if name == "Singular Matrix":
                print("Singular matrix, decomposition should not work.")
        
        except np.linalg.LinAlgError:
            if name == "Singular Matrix":
                print("Right handling of the matrix")
            else:
                print("Error")

test_factorization_LU_pivoting()