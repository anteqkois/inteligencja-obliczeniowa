import numpy as np
import time
from numba import njit
from src.utils.distance import route_length_fast
from src.utils.neighborhoods_numba import neighbor_cost
from src.utils.neighborhoods_numba_delta import neighbor_cost_delta_numba


# ALGORYTM SYMULOWANEGO WYŻARZANIA (SIMULATED ANNEALING - SA)
# ------------------------------------------------------------
# Metaheurystyka stochastyczna oparta na analogii do procesu
# powolnego schładzania metalu. Algorytm wykonuje kolejne zmiany
# bieżącego rozwiązania, dopuszczając przyjęcie rozwiązań gorszych
# z pewnym prawdopodobieństwem zależnym od różnicy kosztów oraz
# aktualnej temperatury. Dzięki temu możliwe jest uciekanie z
# lokalnych minimów, co stanowi przewagę nad klasyczną wspinaczką.
#
# Schemat działania:
#   1. Start od losowej trasy.
#   2. W każdej iteracji generuje sąsiada:
#        - jeśli jest lepszy to zaakceptuj
#        - jeśli gorszy to zaakceptuj z prawdopodobieństwem exp(-Δ/T)
#   3. Obniż temperaturę zgodnie z parametrem alpha (T = T * alpha).
#   4. Kontynuuj, dopóki nie osiągnięto temperatury minimalnej
#      lub liczby max_iter.
#
# Złożoność obliczeniowa:
#   O(max_iter · koszt_sąsiedztwa)
# ------------------------------------------------------------


# NUMBA — wewnętrzna pętla algorytmu SA
@njit(cache=True)
def simulated_annealing_numba(distance_matrix, route, T0, T_min, alpha, max_iter, neighbor_fn_id):
    """
    Pojedyncze wykonanie algorytmu symulowanego wyżarzania.
    W każdej iteracji obniża temperaturę i stosuje regułę akceptacji
    zależną od jakości nowego rozwiązania oraz aktualnej temperatury.

    Parametry:
        distance_matrix : np.ndarray (n x n)
            Macierz odległości pomiędzy miastami.

        route : np.ndarray (n)
            Trasa początkowa, będąca permutacją miast.

        T0 : float
            Temperatura początkowa.

        T_min : float
            Minimalna temperatura. Osiągnięcie tej wartości
            kończy działanie algorytmu.

        alpha : float
            Współczynnik chłodzenia kontrolujący tempo obniżania
            temperatury (T = T * alpha).

        max_iter : int
            Maksymalna liczba iteracji algorytmu.

        neighbor_fn_id : int
            Id operatora sąsiedztwa:
              0 - swap
              1 - two-opt
              2 - insert

    Zwraca:
        best_route : np.ndarray
            Najlepsze znalezione rozwiązanie.
        best_cost : float
            Koszt tej trasy.
    """

    # bieżąca trasa i jej koszt
    current_route = route.copy()
    current_cost = route_length_fast(distance_matrix, current_route)

    # najlepsze dotychczasowe rozwiązanie
    best_route = current_route.copy()
    best_cost = current_cost

    T = T0
    iter_count = 0

    # główna pętla SA
    while T > T_min and iter_count < max_iter:

        # generowanie sąsiada
        new_route, new_cost = neighbor_cost(distance_matrix, current_route, neighbor_fn_id)
        delta = new_cost - current_cost

        # reguła akceptacji, akceptujemy poprawę lub gorsze rozwiązanie z pewnym prawdopodobieństwem
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_route = new_route
            current_cost = new_cost

            # aktualizacja najlepszego rozwiązania
            if new_cost < best_cost:
                best_cost = new_cost
                best_route = new_route

        # obniżenie temperatury i przejście do kolejnego kroku
        T *= alpha
        iter_count += 1

    return best_route, best_cost

@njit(cache=True)
def simulated_annealing_delta_numba(distance_matrix, route, T0, T_min, alpha, max_iter, neighbor_fn_id):
    """
    Pojedyncze wykonanie algorytmu symulowanego wyżarzania.
    W każdej iteracji obniża temperaturę i stosuje regułę akceptacji
    zależną od jakości nowego rozwiązania oraz aktualnej temperatury.

    Parametry:
        distance_matrix : np.ndarray (n x n)
            Macierz odległości pomiędzy miastami.

        route : np.ndarray (n)
            Trasa początkowa, będąca permutacją miast.

        T0 : float
            Temperatura początkowa.

        T_min : float
            Minimalna temperatura. Osiągnięcie tej wartości
            kończy działanie algorytmu.

        alpha : float
            Współczynnik chłodzenia kontrolujący tempo obniżania
            temperatury (T = T * alpha).

        max_iter : int
            Maksymalna liczba iteracji algorytmu.

        neighbor_fn_id : int
            Id operatora sąsiedztwa:
              0 - swap
              1 - two-opt
              2 - insert

    Zwraca:
        best_route : np.ndarray
            Najlepsze znalezione rozwiązanie.
        best_cost : float
            Koszt tej trasy.
    """

    # bieżąca trasa i jej koszt
    current_route = route.copy()
    current_cost = route_length_fast(distance_matrix, current_route)

    # najlepsze dotychczasowe rozwiązanie
    best_route = current_route.copy()
    best_cost = current_cost

    T = T0
    iter_count = 0

    # główna pętla SA
    while T > T_min and iter_count < max_iter:

        # szybkie liczenie kosztu z użyciem DELTA
        new_route, new_cost = neighbor_cost_delta_numba(
            distance_matrix, current_route, current_cost, neighbor_fn_id
        )
        delta = new_cost - current_cost

        # reguła akceptacji, akceptujemy poprawę lub gorsze rozwiązanie z pewnym prawdopodobieństwem
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_route = new_route
            current_cost = new_cost

            # aktualizacja najlepszego rozwiązania
            if new_cost < best_cost:
                best_cost = new_cost
                best_route = new_route

        # obniżenie temperatury i przejście do kolejnego kroku
        T *= alpha
        iter_count += 1

    return best_route, best_cost


# GŁÓWNA FUNKCJA ROZWIĄZUJĄCA TSP METODĄ SA
def solve_tsp(distance_matrix, params):
    """
    Simulated Annealing (SA)
    ------------------------
    Pełna implementacja algorytmu symulowanego wyżarzania dla TSP.
    Funkcja inicjalizuje losową trasę startową, przygotowuje
    parametry i uruchamia numbową pętlę optymalizacyjną.

    Parametry:
        distance_matrix : np.ndarray (n x n)
            Macierz odległości pomiędzy miastami.

        params : dict
            Parametry sterujące przebiegiem SA:
              'T0' : temperatura początkowa (float)
              'T_min' : temperatura minimalna (float)
              'alpha' : współczynnik chłodzenia (float)
              'max_iter' : limit iteracji (int)
              'neighborhood_type' : rodzaj sąsiedztwa ("swap", "two_opt", "insert")

    Zwraca:
        best_route : np.ndarray
            Najlepsza trasa znaleziona przez algorytm.
        best_cost : float
            Koszt tej trasy.
        runtime : float
            Całkowity czas działania algorytmu.
        meta : dict
            Dane pomocnicze użyte przy konfiguracji uruchomienia.
    """

    start_time = time.time()
    n = distance_matrix.shape[0]

    # odczyt parametrów wejściowych
    T0 = float(params.get("T0", 1000.0))
    T_min = float(params.get("T_min", 1.0))
    alpha = float(params.get("alpha", 0.99))
    max_iter = int(params.get("max_iter", 5000))
    neighborhood_type = params.get("neighborhood_type", "swap")
    use_delta = params.get("use_delta", False)

    # przypisanie operatora sąsiedztwa
    neighborhood_map = {"swap": 0, "two_opt": 1, "insert": 2}
    neighbor_fn_id = neighborhood_map.get(neighborhood_type, 0)

    # losowa trasa startowa
    route = np.random.permutation(n)

    # uruchomienie algorytmu SA
    simulated_annealing_fn = simulated_annealing_delta_numba if use_delta is True else simulated_annealing_numba
    best_route, best_cost = simulated_annealing_fn(
        distance_matrix, route, T0, T_min, alpha, max_iter, neighbor_fn_id
    )

    # pomiar czasu wykonania
    runtime = time.time() - start_time

    # dane informacyjne
    meta = {
        "T0": T0,
        "T_min": T_min,
        "alpha": alpha,
        "max_iter": max_iter,
        "neighborhood_type": neighborhood_type,
    }

    return best_route, best_cost, runtime, meta
