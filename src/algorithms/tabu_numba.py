import numpy as np
import time
from numba import njit
from src.utils.distance import route_length_fast
from src.utils.neighborhoods_numba import (
    neighbor_swap,
    neighbor_insert,
    neighbor_two_opt,
)

# TABU SEARCH – WERSJA NUMBA
# Najważniejsze różnice w stosunku do wersji Pythonowej:
#
#   • lista tabu NIE może być deque / listą / setem → musi zostać zastąpiona przez
#       NUMPY ARRAY jako bufor kołowy.
#
#   • tabu przechowuje TYLKO ruch (i, j), a nie całą trasę,
#       bo:
#            - tuple(route) nie jest wspierane w Numba,
#            - ruch jako para indeksów jest standardowym sposobem reprezentacji tabu.
#
#   • wszystkie funkcje pomocnicze muszą być również numba-compatible.
#
# Algorytm działa identycznie:
#   - generuje wielu kandydatów
#   - odrzuca ruchy tabu (chyba że spełniają aspirację)
#   - wybiera najlepszy ruch
#   - aktualizuje trasę i tabu
#
# Dzięki Numba jest wielokrotnie szybszy od wersji Pythonowej.


# Wybór operatora sąsiedztwa dla Numba
# (musimy jawnie użyć ID funkcji, bo Numba nie lubi dynamicznych wywołań)

@njit(cache=True)
def call_neighbor(route, neighborhood_id):
    if neighborhood_id == 0:
        return neighbor_swap(route)
    elif neighborhood_id == 1:
        return neighbor_two_opt(route)
    else:
        return neighbor_insert(route)


# ------------------------------------------------------------------------------
# Sprawdzenie czy ruch (i,j) znajduje się w tabu
# ------------------------------------------------------------------------------
@njit(cache=True)
def is_tabu(i, j, tabu_moves, tabu_tenure, tabu_size):
    """
    Sprawdza, czy ruch (i,j) znajduje się w tabu.

    tabu_moves : 2D array shape (tabu_tenure, 2)
        Bufor kołowy przechowujący ruchy tabu.

    tabu_size : int
        Ile wpisów tabu jest aktualnie aktywnych.
    """
    for k in range(tabu_size):
        if tabu_moves[k, 0] == i and tabu_moves[k, 1] == j:
            return True
    return False


# ------------------------------------------------------------------------------
# Główna pętla Tabu Search (Numba)
# ------------------------------------------------------------------------------
@njit(cache=True)
def tabu_search_numba(distance_matrix, init_route, max_iter, stop_no_improve,
                       tabu_tenure, neighborhood_id, n_neighbors):
    """
    NUMBA – pełna wersja Tabu Search.

    Różnice względem wersji Python:
      - brak deque → bufor kołowy tabu_moves[][]
      - tabu przechowuje tylko ruch (i,j), NIE całą trasę
      - operator sąsiedztwa wybierany przez neighborhood_id
      - obsługa aspiracji jest zachowana

    """

    n = len(init_route)

    current_route = init_route.copy()
    current_cost = route_length_fast(distance_matrix, current_route)

    best_route = current_route.copy()
    best_cost = current_cost

    # --------------------------------------------------------------
    # Lista tabu w postaci bufora kołowego:
    #
    # tabu_moves[k] = [i, j]
    #
    # i, j – indeksy miast, które brały udział w ruchu
    #
    # tabu_index – wskazuje gdzie wpisać kolejny ruch
    # tabu_size – ilu wpisów jest aktualnie aktywnych
    # --------------------------------------------------------------
    tabu_moves = np.full((tabu_tenure, 2), -1)
    tabu_index = 0
    tabu_size = 0

    no_improve = 0

    # --------------------------------------------------------------
    # GŁÓWNA PĘTLA TS
    # --------------------------------------------------------------
    for _ in range(max_iter):

        best_candidate = None
        best_candidate_cost = np.inf
        best_move_i = -1
        best_move_j = -1

        # ----------------------------------------------------------
        # Generacja n_neighbors kandydatów
        # ----------------------------------------------------------
        for _ in range(n_neighbors):

            candidate = call_neighbor(current_route, neighborhood_id)

            # dla ruchów swap/insert/two_opt – ruch opisuje para (i,j)
            # musimy wykryć jaki ruch zaszedł, więc szukamy różnic
            i = -1
            j = -1
            for idx in range(n):
                if candidate[idx] != current_route[idx]:
                    if i == -1:
                        i = idx
                    else:
                        j = idx
                        break

            # jeśli nie znaleziono ruchu – pomijamy
            if i == -1 or j == -1:
                continue

            candidate_cost = route_length_fast(distance_matrix, candidate)

            # ------------------------------------------------------
            # Warunek tabu + aspiracja
            # ------------------------------------------------------
            if is_tabu(i, j, tabu_moves, tabu_tenure, tabu_size) and candidate_cost >= best_cost:
                continue

            # aktualizacja najlepszego kandydata
            if candidate_cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = candidate_cost
                best_move_i = i
                best_move_j = j

        # ----------------------------------------------------------
        # Jeśli nie znaleziono dobrego ruchu → stagnacja
        # ----------------------------------------------------------
        if best_candidate is None:
            no_improve += 1
            if no_improve >= stop_no_improve:
                break
            continue

        # ----------------------------------------------------------
        # Aktualizacja rozwiązania
        # ----------------------------------------------------------
        current_route = best_candidate
        current_cost = best_candidate_cost

        # ----------------------------------------------------------
        # Wpisanie ruchu do tabu (bufor kołowy)
        # ----------------------------------------------------------
        tabu_moves[tabu_index, 0] = best_move_i
        tabu_moves[tabu_index, 1] = best_move_j

        tabu_index = (tabu_index + 1) % tabu_tenure
        if tabu_size < tabu_tenure:
            tabu_size += 1

        # ----------------------------------------------------------
        # Aktualizacja globalnego optimum
        # ----------------------------------------------------------
        if current_cost < best_cost:
            best_cost = current_cost
            best_route = current_route.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= stop_no_improve:
            break

    return best_route, best_cost

def solve_tsp(distance_matrix, params):
    """
    Tabu Search (NUMBA)
    -------------------
    Wersja Tabu Search przyspieszona przez kompilator Numba.
    """

    start_time = time.perf_counter()

    n = distance_matrix.shape[0]
    max_iter = int(params.get("max_iter", 2000))
    stop_no_improve = int(params.get("stop_no_improve", 200))
    tabu_tenure = int(params.get("tabu_tenure", 10))
    neighborhood_type = params.get("neighborhood_type", "two_opt")
    n_neighbors = int(params.get("n_neighbors", 30))

    neighborhood_map = {"swap": 0, "two_opt": 1, "insert": 2}
    neighborhood_id = neighborhood_map.get(neighborhood_type, 1)

    init_route = np.random.permutation(n)

    best_route, best_cost = tabu_search_numba(
        distance_matrix,
        init_route,
        max_iter,
        stop_no_improve,
        tabu_tenure,
        neighborhood_id,
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
