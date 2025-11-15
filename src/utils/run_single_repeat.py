def run_single_repeat(args):
    """
    Uniwersalne uruchomienie pojedynczego powtórzenia algorytmu TSP.
    Używane w multiprocessing.Pool.map().

    Parametry:
        args : tuple
            (solve_func, distance_matrix, params_dict)

    Zwraca:
        (cost, route, runtime)
    """
    solve_func, D, params = args
    route, cost, runtime, meta = solve_func(D, params)
    return cost, route, runtime
