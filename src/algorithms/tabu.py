import numpy as np
import time
from collections import deque
from src.utils.distance import route_length_fast
from src.utils.neighborhoods_numba import (
    neighbor_swap,
    neighbor_two_opt,
    neighbor_insert,
)

# ============================================================
# ALGORYTM TABU SEARCH (TS)
# ============================================================
# Heurystyka przeszukiwania lokalnego z pamięcią zakazów (tabu list).
#
# W każdej iteracji:
#   1. Generujemy wielu kandydatów (n_neighbors) przy użyciu operatora sąsiedztwa.
#   2. Wybieramy najlepszego kandydata, który nie jest tabu
#      (lub spełnia warunek aspiracji – jest lepszy od globalnego optimum).
#   3. Aktualizujemy bieżące rozwiązanie i dodajemy ruch do listy tabu.
#   4. Kończymy, gdy brak poprawy przez określoną liczbę iteracji
#      lub osiągnięto maksymalną liczbę iteracji.
#
# Zalety:
#   - pozwala unikać powrotu do wcześniejszych (lokalnych) minimów
#   - potrafi eksplorować lepsze rejony przestrzeni rozwiązań niż Hill Climbing
# ============================================================


def tabu_search(distance_matrix, init_route, max_iter, stop_no_improve,
                tabu_tenure, neighbor_fn, n_neighbors=30):
    """
    Tabu Search z eksploracją wielu sąsiadów na iterację.

    Parametry:
        distance_matrix : np.ndarray
            Macierz odległości NxN.
        init_route : np.ndarray
            Początkowa permutacja miast.
        max_iter : int
            Maksymalna liczba iteracji.
        stop_no_improve : int
            Liczba iteracji bez poprawy, po której zatrzymujemy algorytm.
        tabu_tenure : int
            Długość listy tabu (liczba ruchów pamiętanych).
        neighbor_fn : callable
            Funkcja generująca sąsiada (Numba).
        n_neighbors : int
            Liczba losowych sąsiadów testowanych w każdej iteracji.

    Zwraca:
        best_route : np.ndarray
            Najlepsza znaleziona trasa.
        best_cost : float
            Koszt tej trasy.
    """
    n = len(init_route)
    current_route = init_route.copy()
    current_cost = route_length_fast(distance_matrix, current_route)

    best_route = current_route.copy()
    best_cost = current_cost

    tabu_list = deque(maxlen=tabu_tenure)
    no_improve = 0

    for _ in range(max_iter):
        best_candidate = None
        best_candidate_cost = np.inf

        # 🔁 eksploracja wielu sąsiadów
        for _ in range(n_neighbors):
            candidate = neighbor_fn(current_route)
            move_key = tuple(candidate)
            candidate_cost = route_length_fast(distance_matrix, candidate)

            # warunek tabu (chyba że poprawa globalna)
            if move_key in tabu_list and candidate_cost >= best_cost:
                continue

            if candidate_cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = candidate_cost

        # brak poprawy — licznik stagnacji
        if best_candidate is None:
            no_improve += 1
            if no_improve >= stop_no_improve:
                break
            continue

        # aktualizacja rozwiązania i listy tabu
        current_route = best_candidate
        current_cost = best_candidate_cost
        tabu_list.append(tuple(best_candidate))

        if current_cost < best_cost:
            best_cost = current_cost
            best_route = best_candidate.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= stop_no_improve:
            break

    return best_route, best_cost


def solve_tsp(distance_matrix, params):
    """
    Funkcja główna: Tabu Search dla problemu TSP.

    Parametry:
        distance_matrix : np.ndarray
            Macierz odległości NxN.
        params : dict
            Parametry algorytmu:
              - max_iter : int              — maks. liczba iteracji
              - stop_no_improve : int       — limit braku poprawy
              - tabu_tenure : int           — długość listy tabu
              - neighborhood_type : str     — rodzaj sąsiedztwa ("swap", "insert", "two_opt")
              - n_neighbors : int           — liczba sąsiadów przeszukiwanych w każdej iteracji

    Zwraca:
        best_route, best_cost, runtime, meta
    """
    start_time = time.perf_counter()

    n = distance_matrix.shape[0]
    max_iter = int(params.get("max_iter", 2000))
    stop_no_improve = int(params.get("stop_no_improve", 200))
    tabu_tenure = int(params.get("tabu_tenure", 10))
    neighborhood_type = params.get("neighborhood_type", "two_opt")
    n_neighbors = int(params.get("n_neighbors", 30))

    # --- wybór operatora sąsiedztwa
    if neighborhood_type == "swap":
        neighbor_fn = neighbor_swap
    elif neighborhood_type == "insert":
        neighbor_fn = neighbor_insert
    elif neighborhood_type == "two_opt":
        neighbor_fn = neighbor_two_opt
    else:
        raise ValueError(f"Nieznany typ sąsiedztwa: {neighborhood_type}")

    # --- losowa trasa startowa
    init_route = np.random.permutation(n)

    # --- uruchomienie głównego przeszukiwania tabu
    best_route, best_cost = tabu_search(
        distance_matrix,
        init_route,
        max_iter,
        stop_no_improve,
        tabu_tenure,
        neighbor_fn,
        n_neighbors
    )

    runtime = time.perf_counter() - start_time

    # --- metadane dla raportów
    meta = {
        "max_iter": max_iter,
        "stop_no_improve": stop_no_improve,
        "tabu_tenure": tabu_tenure,
        "neighborhood_type": neighborhood_type,
        "n_neighbors": n_neighbors,
    }

    return best_route, best_cost, runtime, meta
