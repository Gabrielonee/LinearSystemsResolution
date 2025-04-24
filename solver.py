import numpy as np
from utilities.common import verify_accuracy
from iterativeMethods import jacobi
import tracemalloc
import time


def solver_matrix(matrix, solution=None):
    # Verifica che la matrice sia stata caricata correttamente
    if matrix is None:
        raise ValueError("MatrixReader failed to load the matrix")

    # Verifica che la matrice abbia attributo `.shape` valido
    if not hasattr(matrix, 'shape') or matrix.shape is None:
        raise ValueError("Loaded matrix is invalid or missing shape info.")

    # Estrai dimensioni della matrice
    rows, cols = matrix.shape

    if rows != cols:
        raise ValueError("Matrix is not square. Cannot proceed.")

    # Genera vettore soluzione "vera" x e calcola b = A*x
    x_true = solution if solution is not None else np.ones(rows)
    b = matrix @ x_true  # Assicura che x_true sia la soluzione esatta

    # Misura tempo e memoria per il metodo Jacobi in versione sparsa
    tracemalloc.start()
    start_time = time.time()
    sol_sparse = jacobi.jacobi_solver(
        matrix, b, x0=np.zeros_like(b), tol=1e-10, nmax=10000)
    elapsed_time = time.time() - start_time
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Verifica l’accuratezza della soluzione ottenuta rispetto a x_true
    sparse_error = verify_accuracy(sol_sparse.solution, x_true)

    # Stampa dei risultati
    print(f"    Tempo: {elapsed_time:.6f} s")
    print(f"    Memoria: {peak_memory / 1e6:.2f} MB")
    print(f"    Errore relativo: {sparse_error:.2e}")
    print("—" * 60)
