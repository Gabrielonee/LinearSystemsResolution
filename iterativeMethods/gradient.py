import numpy as np
from utilities.classes import IterativeResult


def gradient_solver(A_sparse, b, x0, tol, nmax):
    converged = False
    #Calcolo del residuo iniziale r = b - Ax0
    r = b - A_sparse @ x0
    #Calcolo della norma 2 del residuo e del termine noto
    norm_r = np.linalg.norm(r)
    norm_b = np.linalg.norm(b)
    #Definizione della tolleranza relativa: ||r_k|| <= tol * ||b||
    rel_tol = tol * norm_b if norm_b > 0 else tol
    #Inizializzazione contatore di iterazioni
    nit = 0
    
    while norm_r > rel_tol and nit < nmax:
        # Calcola A * r_k (serve per denominatore di α_k)
        Ar = A_sparse @ r
        #Calcola il numeratore e denominatore per α_k = (rᵏ ⋅ rᵏ) / (rᵏ ⋅ A rᵏ)
        r_dot_r = np.dot(r, r)
        r_dot_Ar = np.dot(r, Ar)
        if abs(r_dot_Ar) < 1e-14:
            break
        #Calcolo del passo α_k
        alpha = r_dot_r / r_dot_Ar
        #Aggiornamento della soluzione: x_{k+1} = x_k + α_k * r_k
        x0 = x0 + alpha * r
        #Calcolo del nuovo residuo r_{k+1} = b - A x_{k+1}
        r = b - A_sparse @ x0
        #Calcolo della nuova norma del residuo
        norm_r = np.linalg.norm(r)
        #Incremento del numero di iterazioni
        nit += 1

    #Controllo di convergenza
    if nit >= nmax:
        print("Metodo del Gradiente non converge entro il numero massimo di iterazioni.")
    else:
        converged = True

    #Ritorna la soluzione e il numero di iterazioni
    return IterativeResult(x0, nit, converged)
