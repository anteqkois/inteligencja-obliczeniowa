import numpy as np
import time
from numba import njit
from src.utils.distance import route_length_fast
from src.utils.neighborhoods_numba import neighbor_cost


# ============================================================
# SIMULATED ANNEALING (SA) — Symulowane wyżarzanie
# ============================================================
# Heurystyka stochastyczna inspirowana procesem fizycznym
# powolnego schładzania metalu, aby uniknąć lokalnych minimów.
#
# Idea:
#   1. Start z losowej trasy.
#   2. W każdej iteracji:
#       - wygeneruj sąsiada
#       - zaakceptuj go, jeśli jest lepszy
#       - jeśli gorszy → zaakceptuj z prawdopodobieństwem p = exp(-Δ/T)
#   3. Obniż temperaturę zgodnie z wybraną metodą (T *= alpha)
#   4. Zakończ, gdy temperatura spadnie poniżej T_min lub osiągnięto max_iter
#
# Parametry:
#   - T0: temperatura początkowa
#   - T_min: temperatura minimalna (warunek stopu)
#   - alpha: współczynnik chłodzenia (np. 0.95)
#   - max_iter: maksymalna liczba iteracji
#   - neighborhood_type: rodzaj sąsiedztwa ("swap", "insert", "two_opt")
# ============================================================


# ------------------------------------------------------------
# 1️⃣ Wersja rdzenia z Numba — pętla wyżarzania
# ------------------------------------------------------------
@njit
def simulated_annealing_numba(distance_matrix, route, T0, T_min, alpha, max_iter, neighbor_fn_id):
    """
    Jedno przebieg symulowanego wyżarzania.
    Wykonuje iteracje obniżając temperaturę zgodnie z T *= alpha.
    """
    current_route = route.copy()
    current_cost = route_length_fast(distance_matrix, current_route)
    best_route = current_route.copy()
    best_cost = current_cost

    T = T0
    iter_count = 0

    while T > T_min and iter_count < max_iter:
        new_route, new_cost = neighbor_cost(distance_matrix, current_route, neighbor_fn_id)
        delta = new_cost - current_cost

        # --- reguła akceptacji
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_route = new_route
            current_cost = new_cost

            if new_cost < best_cost:
                best_cost = new_cost
                best_route = new_route

        # --- obniżenie temperatury
        T *= alpha
        iter_count += 1

    return best_route, best_cost


# ------------------------------------------------------------
# 2️⃣ Główna funkcja solve_tsp()
# ------------------------------------------------------------
def solve_tsp(distance_matrix, params):
    """
    Symulowane wyżarzanie (Simulated Annealing)
    -------------------------------------------
    Wykonuje stochastyczną optymalizację TSP.
    """
    start_time = time.time()

    # --- odczyt parametrów
    n = distance_matrix.shape[0]
    T0 = float(params.get("T0", 1000.0))
    T_min = float(params.get("T_min", 1.0))
    alpha = float(params.get("alpha", 0.99))
    max_iter = int(params.get("max_iter", 5000))
    neighborhood_type = params.get("neighborhood_type", "swap")

    # --- wybór rodzaju sąsiedztwa
    neighborhood_map = {"swap": 0, "two_opt": 1, "insert": 2}
    neighbor_fn_id = neighborhood_map.get(neighborhood_type, 0)

    # --- losowa trasa początkowa
    route = np.random.permutation(n)

    # --- uruchomienie pętli SA (Numba)
    best_route, best_cost = simulated_annealing_numba(
        distance_matrix, route, T0, T_min, alpha, max_iter, neighbor_fn_id
    )

    runtime = time.time() - start_time

    meta = {
        "T0": T0,
        "T_min": T_min,
        "alpha": alpha,
        "max_iter": max_iter,
        "neighborhood_type": neighborhood_type,
    }

    return best_route, best_cost, runtime, meta
