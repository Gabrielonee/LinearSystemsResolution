import numpy as np
from utils.classes import IterativeResult

def conjugate_gradient_solver(A_sparse, b, x0, tol: float, nmax: int):
    converged = False
    #Calcolo del residuo iniziale r₀ = b - A x₀
    r = b - A_sparse @ x0

    #Prima direzione di ricerca: inizialmente p₀ = r₀
    p = r.copy()

    #Calcolo della norma di b, utile per normalizzare l'errore relativo
    norm_b = np.linalg.norm(b)

    #Calcolo dell'errore relativo iniziale ||r₀|| / ||b||
    err = np.linalg.norm(r) / norm_b if norm_b > 0 else np.linalg.norm(r)

    #Contatore di iterazioni
    nit = 0

    #Calcolo del prodotto scalare r₀ ⋅ r₀ (aggiornato in ogni iterazione)
    r_dot_r = np.dot(r, r)

    while nit < nmax and err > tol:
        #Moltiplicazione A * p_k, serve per α_k e aggiornamento r
        Ap = A_sparse @ p

        #Calcolo del passo α_k = (r_k ⋅ r_k) / (p_k ⋅ A p_k)
        alpha = r_dot_r / np.dot(p, Ap)

        #Aggiornamento soluzione: x_{k+1} = x_k + α_k * p_k
        x0 = x0 + alpha * p

        #Aggiornamento residuo: r_{k+1} = r_k - α_k * A p_k
        r_new = r - alpha * Ap

        #Calcolo del nuovo errore relativo
        err = np.linalg.norm(r_new) / norm_b if norm_b > 0 else np.linalg.norm(r_new)

        #Calcolo del nuovo prodotto scalare r_{k+1} ⋅ r_{k+1}
        r_dot_r_new = np.dot(r_new, r_new)

        #Calcolo di β_k = (r_{k+1} ⋅ r_{k+1}) / (r_k ⋅ r_k)
        beta = r_dot_r_new / r_dot_r

        #Aggiornamento direzione di ricerca: p_{k+1} = r_{k+1} + β_k * p_k
        p = r_new + beta * p

        #Aggiorna residuo e valore r⋅r per l’iterazione successiva
        r = r_new
        r_dot_r = r_dot_r_new

        #Incremento del contatore di iterazioni
        nit += 1

    #Ritorna soluzione approssimata e numero di iterazioni
    if nit >= nmax:
        print("Metodo del Gradiente Coniugato non converge entro il numero massimo di iterazioni.")
    return IterativeResult(x0, nit, converged=(err <= tol))
