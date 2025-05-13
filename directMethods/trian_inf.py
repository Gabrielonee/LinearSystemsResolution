import numpy as np

def triang_inf(L, b):
    """
    Risolve il sistema triangolare inferiore L x = b via forward substitution.

    Parametri
    ---------
    L : (n, n) numpy.ndarray
        Matrice triangolare inferiore, con diagonale non nulla.
    b : (n,) numpy.ndarray
        Vettore termine noto.

    Ritorna
    -------
    x : (n,) numpy.ndarray
        Soluzione del sistema.
    """
    n, m = L.shape
    # Verifica diagonale non nulla
    diag = np.diag(L)
    if np.any(np.abs(diag) < 1e-14):
        raise ValueError("La matrice L ha elementi diagonali nulli o prossimi a zero")

    # Forward substitution
    x = np.zeros(n, dtype=float)
    x[0] = b[0] / L[0, 0]
    for i in range(1, n):
        # Somma L[i, :i] * x[:i]
        s = L[i, :i] @ x[:i]
        x[i] = (b[i] - s) / L[i, i]

    return x
