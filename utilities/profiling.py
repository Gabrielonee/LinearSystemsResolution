import time
import tracemalloc


def profile_solver(solver_func, *args, **kwargs):
    """
    Esegue una funzione di risoluzione del sistema lineare misurando
    il tempo di esecuzione e la memoria massima utilizzata.

    Parametri
    ----------
    solver_func : callable
        Funzione del solver iterativo da profilare (es. jacobi_solver).

    *args : tuple
        Argomenti posizionali da passare alla funzione `solver_func`.

    **kwargs : dict
        Argomenti keyword da passare alla funzione `solver_func`.

    Ritorna
    -------
    result : qualsiasi tipo
        Risultato restituito dalla funzione solver.

    elapsed_time : float
        Tempo di esecuzione in secondi.

    peak_memory : float
        Memoria massima usata durante l'esecuzione, in kilobyte (KB).
    """
    # Avvia tracciamento memoria e tempo
    tracemalloc.start()
    start_time = time.perf_counter()

    # Esecuzione del solver
    result = solver_func(*args, **kwargs)

    # Tempo totale e memoria usata
    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, elapsed_time, peak_memory / 1024  # Convertito in KB
