import numpy as np
import time

from src.algorithms.ihc_numba import hill_climb_delta_numba, hill_climb_numba

# ALGORYTM GRASP — GREEDY RANDOMIZED ADAPTIVE SEARCH PROCEDURE
# ---------------------------------------------------------------
# Algorytm metaheurystyczny używany do rozwiązywania problemu komiwojażera (TSP). Składa się z dwóch głównych faz:
#
# (1) Konstrukcji rozwiązania metodą „zachłanną z losowością” (Greedy Randomized Construction).
# Trasa budowana jest krok po kroku. Zamiast wybierać zawsze najbliższe miasto (jak w klasycznym algorytmie NN), tworzy się
# tzw. RCL — Restricted Candidate List, czyli listę kilku najlepszych kandydatów o najmniejszych odległościach. Z tej listy wybiera się
# następne miasto losowo, ale w sposób kontrolowany przez parametr α. Pozwala to budować różne, zróżnicowane trasy startowe,
# unikając deterministycznych wyborów i eksplorując szerszą część przestrzeni rozwiązań.
#
# (2) Lokalnego ulepszania trasy (Local Search), które w tym projekcie realizowane jest za pomocą lekkiej wspinaczki lokalnej (IHC-light).
# Po skonstruowaniu wstępnej trasy wykonywane jest lokalne przeszukiwanie oparte na prostym algorytmie Hill Climbing,
# które iteracyjnie poprawia trasę poprzez wykonywanie ruchów typu „swap”, „insert” lub „two-opt” (zależnie od parametrów).
# Ulepszanie trwa, dopóki trasa ulega poprawie lub dopóki nie zostanie przekroczony limit iteracji / stagnacji.
#
# GRASP wykonuje powyższe dwie fazy wielokrotnie (liczbę powtórzeń określa parametr iterations). W każdej iteracji powstaje nowa,
# nieco inna trasa startowa, wynikająca z losowego wyboru w fazie konstrukcyjnej. Następnie każda taka trasa jest ulepszana przez
# lokalne przeszukiwanie. Spośród wszystkich przebiegów algorytmu wybierane jest rozwiązanie o najmniejszym koszcie.
#
# Charakterystyczna cecha GRASP polega na połączeniu:
# • zachłonności (szybkie podejmowanie dobrych decyzji),
# • losowości (uniknięcie zbyt wczesnej koncentracji na jednym optimum),
# • lokalnego ulepszania (osiąganie jakościowego optimum lokalnego).
# Dzięki temu GRASP jest uniwersalny, prosty w implementacji i osiąga
# wysoką jakość rozwiązań nawet dla dużych instancji problemu TSP.
# Złożoność obliczeniowa:
# O(iterations · (koszt_konstrukcji + koszt_local_search))
# ---------------------------------------------------------------

# FAZA KONSTRUKCJI GRASP
def grasp_construct(distance_matrix, alpha):
    """
    Konstrukcja greedy-randomized używana w GRASP.
    Nie jest w Numba ponieważ listy, sortowania itd. są niekompatybilne.

    Parametry:
        distance_matrix : np.ndarray (n x n)
        alpha : float (0.0 - greedy, 1.0 - mocna losowość)

    Zwraca:
        route : np.ndarray - trasa startowa
    """

    n = distance_matrix.shape[0]
    route = np.empty(n, dtype=np.int64)

    # miasta nieodwiedzone
    remaining = list(range(n))

    # start w losowym mieście
    current = np.random.randint(0, n)
    route[0] = current
    remaining.remove(current)

    # budowa trasy
    for idx in range(1, n):

        # dystanse do kandydatów
        dists = np.array([distance_matrix[current, c] for c in remaining])

        min_d = dists.min()
        max_d = dists.max()
        threshold = min_d + alpha * (max_d - min_d)

        # RCL – kandydaci <= próg
        rcl_indices = np.where(dists <= threshold)[0]

        # losowy wybór z RCL
        chosen_idx = np.random.choice(rcl_indices)
        chosen_city = remaining[chosen_idx]

        route[idx] = chosen_city
        remaining.pop(chosen_idx)

        current = chosen_city

    return route

def solve_tsp(distance_matrix, params):
    """
    GRASP - Greedy Randomized Adaptive Search Procedure
    ---------------------------------------------------
    Algorytm działa w cyklu:
        (1) konstrukcja losowo-zachłanna trasy (greedy + randomness)
        (2) lokalne ulepszanie trasy (IHC-light → Numba)
        (3) wybór najlepszego rozwiązania z wielu iteracji

    Parametry:
        distance_matrix : np.ndarray (n x n)

        params : dict
            'alpha' : float (0-1)
                Poziom losowości konstrukcji.

            'iterations' : int
                Liczba powtórzeń GRASP (konstrukcja → local search).

            'neighborhood_type' : str
                Jeden z: "swap", "two_opt", "insert"
                Typ ruchu używany w local search.

            'ihc_max_iter' : int
                Limit iteracji IHC-light.

            'ihc_stop_no_improve' : int
                Limit stagnacji w IHC-light.

            'use_delta' : bool
                Jeśli True → hill_climb_delta_numba
                Jeśli False → hill_climb_numba

    Zwraca:
        best_route : np.ndarray
        best_cost : float
        runtime : float
        meta : dict
    """

    start_time = time.time()

    # odczyt parametrów
    alpha = float(params.get("alpha", 0.3))
    iterations = int(params.get("iterations", 100))
    neighborhood_type = params.get("neighborhood_type", "swap")
    ihc_max_iter = int(params.get("ihc_max_iter", 300))
    ihc_stop_no_improve = int(params.get("ihc_stop_no_improve", 100))
    use_delta = params.get("use_delta", True)

    # mapowanie nazw ruchów na liczby (kompatybilne z Numba dla ihs)
    neighborhood_map = {"swap": 0, "two_opt": 1, "insert": 2}
    neighbor_fn_id = neighborhood_map.get(neighborhood_type, 0)

    # wybór funkcji LS (delta / full), obcnie na sztywno ihs
    hill_climb_fn = hill_climb_delta_numba if use_delta else hill_climb_numba

    best_route = None
    best_cost = np.inf

    #  GŁÓWNA PĘTLA GRASP
    for _ in range(iterations):

        # (1) KONSTRUKCJA GREEDY + RANDOM
        route0 = grasp_construct(distance_matrix, alpha)

        # (2) LOCAL SEARCH – IHC-light
        local_route, local_cost = hill_climb_fn(
            distance_matrix,
            route0,
            ihc_max_iter,
            ihc_stop_no_improve,
            neighbor_fn_id
        )

        # aktualizacja najlepszego wyniku
        if local_cost < best_cost:
            best_cost = local_cost
            best_route = local_route.copy()

    # czas działania
    runtime = time.time() - start_time

    # meta-informacje
    meta = {
        "alpha": alpha,
        "iterations": iterations,
        "neighborhood_type": neighborhood_type,
        "ihc_max_iter": ihc_max_iter,
        "ihc_stop_no_improve": ihc_stop_no_improve,
        "use_delta": use_delta,
    }

    return best_route, best_cost, runtime, meta