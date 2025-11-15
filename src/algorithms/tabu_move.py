import numpy as np
import time
from collections import deque
from src.utils.distance import route_length_fast
from src.utils.neighborhoods_numba import get_neighbor_function


# WYKRYWANIE RUCHU ZMIAN W TRASIE
def detect_move(old_route, new_route):
    """
    Wykrywa ruch wykonany między starą a nową trasą.
    Zwraca tuple (i, j) – standardowa reprezentacja ruchu tabu.

    Działa dla:
      • swap
      • insert
      • two-opt

    Mechanizm:
        1. Znajdujemy indeksy różnic między old_route i new_route.
        2. Pierwszy i ostatni indeks różnicy → (i, j)
           - dla swap: to dokładnie dwa różne miejsca
           - dla insert: i = usunięty element, j = miejsce wstawienia
           - dla two-opt: i, j to końce segmentu odwróconego
    """
    diffs = [k for k in range(len(old_route)) if old_route[k] != new_route[k]]

    if len(diffs) == 0:
        return None  # brak ruchu
    if len(diffs) == 1:
        return (diffs[0], diffs[0])  # minimalna zmiana (raczej się nie zdarza)

    return (diffs[0], diffs[-1])
# =====================================================



# ALGORYTM TABU SEARCH (TS)
# ------------------------------------------------------------
def tabu_search(distance_matrix, init_route, max_iter, stop_no_improve,
                tabu_tenure, neighbor_fn, n_neighbors=30):

    n = len(init_route)
    current_route = init_route.copy()
    current_cost = route_length_fast(distance_matrix, current_route)

    best_route = current_route.copy()
    best_cost = current_cost

    # LISTA TABU przechowuje ruchy typu (i, j)
    tabu_list = deque(maxlen=tabu_tenure)

    no_improve = 0

    # GŁÓWNA PĘTLA TS
    for _ in range(max_iter):

        best_candidate = None
        best_candidate_cost = np.inf
        best_candidate_move = None

        # GENEROWANIE WIELU SĄSIADÓW
        for _ in range(n_neighbors):
            candidate = neighbor_fn(current_route)
            candidate_cost = route_length_fast(distance_matrix, candidate)

            # wykrycie faktycznego ruchu
            move = detect_move(current_route, candidate)

            # warunek tabu (blokujemy ruch)
            if move in tabu_list and candidate_cost >= best_cost:
                continue

            # wybieramy najlepszego kandydata
            if candidate_cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = candidate_cost
                best_candidate_move = move

        # brak poprawnych kandydatów
        if best_candidate is None:
            no_improve += 1
            if no_improve >= stop_no_improve:
                break
            continue

        # AKTUALIZACJA ROZWIĄZANIA
        current_route = best_candidate
        current_cost = best_candidate_cost

        # DODAJEMY RUCH DO TABU
        if best_candidate_move is not None:
            tabu_list.append(best_candidate_move)

        # aktualizacja najlepszego wyniku globalnego
        if current_cost < best_cost:
            best_cost = current_cost
            best_route = best_candidate.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= stop_no_improve:
            break

    return best_route, best_cost



# FUNKCJA GŁÓWNA
def solve_tsp(distance_matrix, params):

    start_time = time.perf_counter()

    n = distance_matrix.shape[0]
    max_iter = int(params.get("max_iter", 2000))
    stop_no_improve = int(params.get("stop_no_improve", 200))
    tabu_tenure = int(params.get("tabu_tenure", 10))
    neighborhood_type = params.get("neighborhood_type", "two_opt")
    n_neighbors = int(params.get("n_neighbors", 30))

    neighbor_fn = get_neighbor_function(neighborhood_type)

    # LOSOWA TRASA STARTOWA
    init_route = np.random.permutation(n)

    # GŁÓWNY ALGORYTM
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

    meta = {
        "max_iter": max_iter,
        "stop_no_improve": stop_no_improve,
        "tabu_tenure": tabu_tenure,
        "neighborhood_type": neighborhood_type,
        "n_neighbors": n_neighbors,
    }

    return best_route, best_cost, runtime, meta
