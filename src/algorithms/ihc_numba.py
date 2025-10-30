import numpy as np
import time
from numba import njit
from src.utils.distance import route_length_fast
from src.utils.neighborhoods_numba import neighbor_cost


# ============================================================
# ITERATIVE HILL CLIMBING (IHC) — wersja przyspieszona Numboą
# ============================================================
# Logika:
#   - wielokrotne wspinaczki lokalne (multistart)
#   - w każdej wspinaczce: ruchy lokalne typu swap/insert/two_opt
#   - zakończenie po określonej liczbie iteracji lub braku poprawy
#
# Różnice:
#   - hill_climb() korzysta z numbowego helpera `neighbor_cost()`
#   - wybór typu sąsiedztwa (0,1,2) tylko raz, nie w każdej iteracji
# ============================================================


# ------------------------------------------------------------
# 1️⃣ NUMBA — pojedyncza wspinaczka lokalna
# ------------------------------------------------------------
@njit
def hill_climb_numba(distance_matrix, route, max_iter, stop_no_improve, neighbor_fn_id):
    """
    Hill Climb (Numba)
    ------------------
    Działa analogicznie do klasycznej wersji, ale z użyciem
    numbowych operatorów sąsiedztwa (`neighbor_cost()`).

    Parametry:
        distance_matrix : np.ndarray[n,n]
        route : np.ndarray[n]
        max_iter : int
        stop_no_improve : int
        neighbor_fn_id : int
            0 → swap, 1 → two_opt, 2 → insert
    """
    best_route = route.copy()
    best_cost = route_length_fast(distance_matrix, best_route)
    no_improve = 0

    for _ in range(max_iter):
        # wygeneruj sąsiada i policz koszt (numba helper)
        candidate, candidate_cost = neighbor_cost(distance_matrix, best_route, neighbor_fn_id)

        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_route = candidate
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= stop_no_improve:
            break

    return best_route, best_cost


# ------------------------------------------------------------
# 2️⃣ GŁÓWNA FUNKCJA ROZWIĄZUJĄCA TSP
# ------------------------------------------------------------
def solve_tsp(distance_matrix, params):
    """
    Iterative Hill Climbing (IHC) z Numba
    -------------------------------------
    Uruchamia wiele wspinaczek lokalnych z losowych startów
    i zwraca najlepsze znalezione rozwiązanie.

    Parametry:
        distance_matrix : np.ndarray
        params : dict
            {
              'n_starts': liczba restartów,
              'max_iter': maksymalna liczba iteracji jednej wspinaczki,
              'stop_no_improve': maksymalna liczba iteracji bez poprawy,
              'neighborhood_type': "swap" | "insert" | "two_opt"
            }
    Zwraca:
        route, cost, runtime, meta
    """
    start_time = time.time()
    n = distance_matrix.shape[0]

    # --- pobranie parametrów
    n_starts = int(params.get("n_starts", 10))
    max_iter = int(params.get("max_iter", 500))
    stop_no_improve = int(params.get("stop_no_improve", 50))
    neighborhood_type = params.get("neighborhood_type", "swap")

    # --- zamiana typu sąsiedztwa na ID (dla Numba)
    neighborhood_map = {"swap": 0, "two_opt": 1, "insert": 2}
    neighbor_fn_id = neighborhood_map.get(neighborhood_type, 0)

    # --- najlepsze globalne rozwiązanie
    best_route = None
    best_cost = np.inf

    # --- pętla wielokrotnych startów
    for _ in range(n_starts):
        route = np.random.permutation(n)
        local_route, local_cost = hill_climb_numba(
            distance_matrix,
            route,
            max_iter,
            stop_no_improve,
            neighbor_fn_id
        )

        if local_cost < best_cost:
            best_cost = local_cost
            best_route = local_route.copy()

    runtime = time.time() - start_time

    meta = {
        "n_starts": n_starts,
        "max_iter": max_iter,
        "stop_no_improve": stop_no_improve,
        "neighborhood_type": neighborhood_type,
    }

    return best_route, best_cost, runtime, meta