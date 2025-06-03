import time
import tracemalloc


def profile_solver(solver_func, *args, **kwargs):
    """
    Esegue un solver per sistemi lineari tracciando tempo di esecuzione e
    memoria massima utilizzata.

    Parameters
    ----------
    solver_func : callable
        Funzione risolutiva (es. metodo iterativo) da profilare.
        Deve restituire un oggetto risultato.
    *args : tuple
        Argomenti posizionali da passare a `solver_func`.
    **kwargs : dict
        Argomenti keyword da passare a `solver_func`.

    Returns
    -------
    result : object
        Risultato restituito da `solver_func`.
        elapsed_time : float
        Tempo totale di esecuzione in secondi.
    peak_memory_kb : float
        Memoria massima utilizzata durante l'esecuzione, in kilobyte (KB).

    Notes
    -----
    Utilizza `tracemalloc` per rilevare il picco di memoria.
    La memoria viene riportata in kilobyte.
    Il tempo viene misurato con `time.perf_counter()`
    per la massima precisione.
    """
    tracemalloc.start()
    start_time = time.perf_counter()

    result = solver_func(*args, **kwargs)

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, elapsed_time, peak_memory / 1024  # Convertito in KB
