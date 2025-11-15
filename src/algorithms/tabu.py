import numpy as np
import time
from collections import deque
from src.utils.distance import route_length_fast
from src.utils.neighborhoods_numba import get_neighbor_function

# ALGORYTM TABU SEARCH (TS)
# ------------------------------------------------------------
# Heurystyka przeszukiwania lokalnego wykorzystująca pamięć tabu,
# czyli listę ruchów, które są tymczasowo zabronione. Celem tej
# pamięci jest uniemożliwienie cyklicznego powrotu do niedawno
# odwiedzonych rozwiązań, co pozwala eksplorować przestrzeń rozwiązań
# efektywniej niż klasyczna wspinaczka lokalna.
#
# Schemat działania:
#   1. Start z losowej trasy początkowej.
#   2. W każdej iteracji generacja wielu kandydatów (n_neighbors)
#      przy użyciu wybranego operatora sąsiedztwa.
#   3. Wybór najlepszego kandydata, który:
#        - nie znajduje się na liście tabu
#        - albo spełnia warunek aspiracji (jest lepszy niż globalne optimum).
#   4. Aktualizacja rozwiązania bieżącego i dodanie ruchu do tabu.
#   5. Zatrzymanie po osiągnięciu limitu iteracji lub liczby iteracji
#      bez poprawy końcowego wyniku.
#
# Algorytm dobrze radzi sobie z unikaniem lokalnych minimów poprzez
# kontrolowaną eksplorację obszarów, które klasyczne metody omijają.
#
# Złożoność obliczeniowa:
#   O(max_iter · n_neighbors · koszt_sąsiedztwa)
# ------------------------------------------------------------


def tabu_search(distance_matrix, init_route, max_iter, stop_no_improve,
                tabu_tenure, neighbor_fn, n_neighbors=30):
    """
    Właściwa pętla algorytmu Tabu Search wykonująca iteracyjne
    przeszukiwanie lokalne z wykorzystaniem listy tabu.

    Parametry:
        distance_matrix : np.ndarray (n x n)
            Macierz odległości między miastami.

        init_route : np.ndarray (n)
            Trasa startowa wykorzystywana jako punkt początkowy.

        max_iter : int
            Maksymalna liczba iteracji algorytmu.

        stop_no_improve : int
            Liczba kolejnych iteracji bez poprawy najlepszego wyniku,
            po której algorytm kończy pracę.

        tabu_tenure : int
            Maksymalna długość listy tabu, określająca ile ostatnich
            ruchów jest traktowanych jako zabronione.

        neighbor_fn : callable
            Funkcja generująca sąsiada na bazie aktualnej trasy.
            Powinna implementować jeden z operatorów: swap, insert lub two_opt.

        n_neighbors : int
            Liczba kandydatów (losowych sąsiadów) generowanych w każdej iteracji.

    Zwraca:
        best_route : np.ndarray
            Najlepsze rozwiązanie odnalezione podczas przeszukiwania.

        best_cost : float
            Koszt tej trasy.
    """

    n = len(init_route)
    current_route = init_route.copy()
    current_cost = route_length_fast(distance_matrix, current_route)

    best_route = current_route.copy()
    best_cost = current_cost

    # lista tabu, która przechowuje ostatnie ruchy, aby unikać cykli
    tabu_list = deque(maxlen=tabu_tenure)

    no_improve = 0

    # główna pętla TS
    for _ in range(max_iter):

        best_candidate = None
        best_candidate_cost = np.inf

        # eksploracja wielu sąsiadów
        for _ in range(n_neighbors):
            candidate = neighbor_fn(current_route)
            move_key = tuple(candidate)
            candidate_cost = route_length_fast(distance_matrix, candidate)

            # warunek tabu z aspiracją (jeśli poprawiamy globalne optimum to ignorujemy tabu)
            if move_key in tabu_list and candidate_cost >= best_cost:
                continue

            if candidate_cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = candidate_cost

        # brak dobrego kandydata, stagnacja
        if best_candidate is None:
            no_improve += 1
            if no_improve >= stop_no_improve:
                break
            continue

        # aktualizacja rozwiązania
        current_route = best_candidate
        current_cost = best_candidate_cost
        tabu_list.append(tuple(best_candidate))

        # aktualizacja najlepszego globalnego rozwiązania
        if current_cost < best_cost:
            best_cost = current_cost
            best_route = best_candidate.copy()
            no_improve = 0
        else:
            no_improve += 1

        # zatrzymanie przy długiej stagnacji
        if no_improve >= stop_no_improve:
            break

    return best_route, best_cost


def solve_tsp(distance_matrix, params):
    """
    Tabu Search (TS)
    ----------------
    Funkcja uruchamia algorytm Tabu Search dla problemu TSP.
    Inicjalizuje parametry, generuje losową trasę początkową
    i wykonuje właściwe przeszukiwanie.

    Parametry:
        distance_matrix : np.ndarray (n x n)
            Macierz odległości pomiędzy wszystkimi miastami.

        params : dict
            Parametry sterujące algorytmem:
              'max_iter' : maksymalna liczba iteracji (int)
              'stop_no_improve' : limit iteracji bez poprawy (int)
              'tabu_tenure' : długość listy tabu (int)
              'neighborhood_type' : typ operatora sąsiedztwa (str)
                    dopuszczalne wartości: "swap", "insert", "two_opt"
              'n_neighbors' : liczba sąsiadów generowanych w iteracji (int)

    Zwraca:
        best_route : np.ndarray
            Najlepsza znaleziona trasa.
        best_cost : float
            Koszt tej trasy.
        runtime : float
            Czas działania algorytmu.
        meta : dict
            Parametry uruchomienia, przydatne w analizie wyników.
    """

    start_time = time.perf_counter()

    n = distance_matrix.shape[0]
    max_iter = int(params.get("max_iter", 2000))
    stop_no_improve = int(params.get("stop_no_improve", 200))
    tabu_tenure = int(params.get("tabu_tenure", 10))
    neighborhood_type = params.get("neighborhood_type", "two_opt")
    n_neighbors = int(params.get("n_neighbors", 30))

    neighbor_fn = get_neighbor_function(neighborhood_type)

    # losowa trasa startowa
    init_route = np.random.permutation(n)

    # uruchomienie algorytmu TS
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
