import os
import readMatrix as rm
import numpy as np
import tracemalloc
import time

# Directory che contiene i file .mtx
directory = os.fsencode(
    "/Users/fraromeo/Documents/02_Areas/University/LM/LM_24-25/SEM2/MdCS/dati")

# Iterazione su tutti i file della directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    # Considera solo i file con estensione .mtx
    if filename.endswith(".mtx"):
        # Costruzione del path completo
        matrixPath = os.path.join(os.fsdecode(directory), filename)

        # Caricamento della matrice sparsa dal file .mtx
        A_sparse = rm.MatrixReader(matrixPath)

        # Verifica che la matrice sia stata caricata correttamente
        if A_sparse is None:
            raise ValueError(
                f"MatrixReader failed to load matrix from: {matrixPath}")

        # Verifica che la matrice abbia attributo `.shape` valido
        if not hasattr(A_sparse, 'shape') or A_sparse.shape is None:
            raise ValueError("Loaded matrix is invalid or missing shape info.")

        # Estrai dimensioni della matrice
        rows, cols = A_sparse.shape
        print(f"    Matrix: {filename} | Shape: {rows} x {cols}")

        if rows != cols:
            raise ValueError("Matrix is not square. Cannot proceed.")

        # Genera vettore soluzione "vera" x e calcola b = A*x
        x_true = np.ones(rows)
        b = A_sparse @ x_true  # Assicura che x_true sia la soluzione esatta

        # Misura tempo e memoria per il metodo Jacobi in versione sparsa
        tracemalloc.start()
        start_time = time.time()
        sol_sparse = rm.jacobi_sparse(A_sparse, b, tol=1e-10)
        elapsed_time = time.time() - start_time
        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Verifica l’accuratezza della soluzione ottenuta rispetto a x_true
        sparse_error = rm.verify_accuracy(sol_sparse, x_true)

        # Stampa dei risultati
        print(f"    Tempo: {elapsed_time:.6f} s")
        print(f"    Memoria: {peak_memory / 1e6:.2f} MB")
        print(f"    Errore relativo: {sparse_error:.2e}")
        print("—" * 60)
