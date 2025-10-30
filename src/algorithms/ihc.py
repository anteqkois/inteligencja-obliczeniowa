import numpy as np
import time
from src.utils.distance import route_length_fast
from src.utils.neighborhoods import swap, insert, two_opt


# ============================================================
# ALGORYTM ITERATIVE HILL CLIMBING (IHC)
# ============================================================
# Heurystyka lokalnego przeszukiwania z wielokrotnym startem.
#
# Idea:
#  1. Wykonujemy wiele wspinaczek lokalnych (hill climbing),
#     każda zaczyna z losowej permutacji miast.
#  2. W każdej wspinaczce:
#     - generujemy sąsiadów jednym z operatorów (swap, insert, two_opt)
#     - jeśli sąsiad jest lepszy — przechodzimy do niego
#     - kończymy, gdy długo brak poprawy lub osiągnięto max_iter
#  3. Zwracamy najlepsze rozwiązanie spośród wszystkich startów.
#
# Zalety:
#  - prostota implementacji
#  - dobra skuteczność na małych i średnich TSP
# ============================================================


def hill_climb(distance_matrix, route, max_iter, stop_no_improve, neighbor_fn):
    """
    Jedna wspinaczka lokalna (Hill Climbing)
    ---------------------------------------
    - Przechodzi po przestrzeni rozwiązań lokalnych
    - Wybiera tylko lepszych sąsiadów
    - Kończy po osiągnięciu lokalnego optimum

    Parametry:
        distance_matrix : np.ndarray
            Macierz odległości NxN
        route : np.ndarray
            Permutacja miast (rozwiązanie początkowe)
        max_iter : int
            Maksymalna liczba iteracji
        stop_no_improve : int
            Liczba iteracji bez poprawy, po której zatrzymujemy proces
        neighbor_fn : callable
            Funkcja generująca sąsiada (np. swap, insert, two_opt)

    Zwraca:
        best_route : np.ndarray
            Najlepsza znaleziona trasa
        best_cost : float
            Długość tej trasy
    """
    # --- inicjalizacja
    best_route = route.copy()
    best_cost = route_length_fast(distance_matrix, best_route)
    no_improve = 0

    # --- główna pętla wspinaczki
    for _ in range(max_iter):
        # wygeneruj sąsiada przy użyciu wybranego operatora
        candidate = neighbor_fn(best_route)
        candidate_cost = route_length_fast(distance_matrix, candidate)

        # jeśli znaleziono lepszego — aktualizuj rozwiązanie
        if candidate_cost < best_cost:
            best_route = candidate
            best_cost = candidate_cost
            no_improve = 0  # reset licznika braku poprawy
        else:
            no_improve += 1

        # jeśli długo brak poprawy — zakończ lokalne poszukiwanie
        if no_improve >= stop_no_improve:
            break

    return best_route, best_cost


def solve_tsp(distance_matrix, params):
    """
    Iterative Hill Climbing (IHC)
    -----------------------------
    Wykonuje wielokrotną wspinaczkę lokalną (hill climbing)
    z różnych punktów startowych (multistart).
    Zwraca najlepsze znalezione rozwiązanie.

    Parametry:
        distance_matrix : np.ndarray
            Macierz odległości NxN
        params : dict
            Parametry algorytmu:
              - n_starts : liczba uruchomień wspinaczki
              - max_iter : maks. liczba iteracji
              - stop_no_improve : maks. liczba iteracji bez poprawy
              - neighborhood_type : rodzaj sąsiedztwa ("swap", "insert", "two_opt")

    Zwraca:
        best_route : np.ndarray
        best_cost : float
        runtime : float
        meta : dict
    """
    start_time = time.time()

    # --- odczyt parametrów wejściowych
    n = distance_matrix.shape[0]
    n_starts = int(params.get("n_starts", 10))
    max_iter = int(params.get("max_iter", 500))
    stop_no_improve = int(params.get("stop_no_improve", 50))
    neighborhood_type = params.get("neighborhood_type", "swap")

    # --- wybór operatora sąsiedztwa (raz, nie w każdej iteracji)
    if neighborhood_type == "swap":
        neighbor_fn = swap
    elif neighborhood_type == "insert":
        neighbor_fn = insert
    elif neighborhood_type == "two_opt":
        neighbor_fn = two_opt
    else:
        raise ValueError(f"Nieznany typ sąsiedztwa: {neighborhood_type}")

    # --- inicjalizacja najlepszych wyników globalnych
    best_route = None
    best_cost = np.inf

    # --- pętla multistartu
    for start_idx in range(n_starts):
        # losowa trasa początkowa (losowy start)
        route = np.random.permutation(n)

        # uruchom pojedynczą wspinaczkę lokalną
        local_route, local_cost = hill_climb(
            distance_matrix,
            route,
            max_iter,
            stop_no_improve,
            neighbor_fn
        )

        # jeśli lokalne rozwiązanie jest lepsze — zapamiętaj globalnie
        if local_cost < best_cost:
            best_cost = local_cost
            best_route = local_route.copy()

    # --- pomiar czasu wykonania
    runtime = time.time() - start_time

    # --- dane meta (dla raportu / analizy)
    meta = {
        "n_starts": n_starts,
        "max_iter": max_iter,
        "stop_no_improve": stop_no_improve,
        "neighborhood_type": neighborhood_type,
    }

    # --- zwrócenie wyników
    return best_route, best_cost, runtime, meta