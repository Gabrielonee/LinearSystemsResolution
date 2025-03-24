import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert, solve, cond

def metodo_Gauss_pivoting_totale(A, b):
    """Risoluzione di un sistema lineare Ax=b usando eliminazione di Gauss con pivoting totale"""
    return np.linalg.solve(A, b)

# Inizializzazione dei vettori di errore e numero di condizionamento
err_vec = []
cond_vec = []

for n in range(5, 55, 5):  # Loop sulle dimensioni del sistema
    A = hilbert(n)  # Matrice di Hilbert
    b = np.ones(n)  # Vettore termine noto

    # Generazione della perturbazione casuale
    pert_factor = 1e-9
    pert_mat = (np.random.rand(n, n) - 0.5) * 2  # Matrice con elementi in (-1,1)
    A_pert = A + pert_factor * pert_mat  # Matrice perturbata
    pert_vec = (np.random.rand(n) - 0.5) * 2  # Vettore con elementi in (-1,1)
    b_pert = b + pert_factor * pert_vec  # Vettore perturbato

    # Soluzione del sistema originale e perturbato
    x = metodo_Gauss_pivoting_totale(A, b)
    x_pert = metodo_Gauss_pivoting_totale(A_pert, b_pert)

    # Calcolo dell'errore relativo
    err_vec.append(np.linalg.norm(x - x_pert) / np.linalg.norm(x))
    cond_vec.append(cond(A))  # Numero di condizionamento della matrice A

# Stampa dei risultati
print("Errori relativi:", err_vec)
print("Condizionamenti:", cond_vec)

# Plot dell'errore relativo rispetto alla dimensione del sistema
plt.plot(range(5, 55, 5), err_vec, marker='o', label="Errore relativo")
plt.xlabel("Dimensione del sistema")
plt.ylabel("Errore relativo")
plt.title("Effetto delle perturbazioni sui sistemi lineari")
plt.legend()
plt.grid()
plt.show()