import numpy as np
from utilities.metrics import verify_accuracy
from iterativeMethods import gauss_seidel, jacobi, gradient, con_gradient, gauss_seidel_lib
from utilities.classes import SolverResult
from utilities.validation import validate_matrix
from utilities.profiling import profile_solver


def solver_matrix(matrix, right_side=None, solution=None,
                  tol=1e-10, nmax=20000):
    """
    Esegue tutti i metodi iterativi implementati per la risoluzione di
    un sistema lineare Ax = b,
    dove A è una matrice simmetrica e definita positiva.

    Parametri
    ----------
    matrix : scipy.sparse matrix o numpy.ndarray
        Matrice dei coefficienti del sistema

    right_side : numpy.ndarray, opzionale
        Vettore relativo al lato destro del sistema lineare.
        Se non fornito, verrà calcolato a bartire da A e x.

    solution : numpy.ndarray, opzionale
        Vettore soluzione "vera" x da usare per il calcolo di b = Ax
        e del relativo errore.
        Se non fornito, si assume un vettore di 1 (np.ones).

    tol : float, default=1e-10
        Tolleranza sul residuo relativa per l'arresto dei metodi iterativi.

    nmax : int, default=20000
        Numero massimo di iterazioni concesso a ciascun metodo.

    Ritorna
    -------
    results : dict
        Dizionario contenente, per ogni metodo:
            - un oggetto SolverResult con tutti i dati della computazione
              (soluzione, iterazioni, tempo, memoria, errore),
            - oppure una stringa d'errore se il metodo fallisce.
    """

    # Validazione della matrice in input
    rows = validate_matrix(matrix)
    results = {}

    # Costruzione della soluzione esatta e del membro destro b = Ax
    x_true = solution if solution is not None else np.ones(rows)
    b = right_side if right_side is not None else matrix @ x_true

    # Definizione dei metodi iterativi disponibili
    solvers = {
        "Jacobi": jacobi.jacobi_solver,
        #"GaussSeidel": gauss_seidel.gauss_seidel_solver,
        "GaussSeidelLib": gauss_seidel_lib.gauss_seidel_solver_library,
        "Gradient": gradient.gradient_solver,
        "ConjugateGradient": con_gradient.conjugate_gradient_solver
    }

    # Esecuzione di ciascun metodo con profilazione di tempo e memoria
    for name, method in solvers.items():
        try:
            # Misura tempo e picco di memoria usata
            sol, elapsed_time, peak_memory = profile_solver(
                method,
                matrix, b,
                x0=np.zeros_like(b),
                tol=tol,
                nmax=nmax
            )

            # Calcolo errore relativo rispetto alla soluzione esatta
            rel_error = verify_accuracy(sol.solution, x_true)

            # Costruzione del risultato per il metodo corrente
            results[name] = SolverResult(
                method_name=name,
                tol=tol,
                max_iterations=nmax,
                method_result=sol,
                rel_error=rel_error,
                time_seconds=elapsed_time,
                memory_kb=peak_memory
            )

        except Exception as e:
            # In caso di errore, salva messaggio d'errore invece del risultato
            results[name] = f"Error: {str(e)}"

    return results
