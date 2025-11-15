import numpy as np
import time
from numba import njit
from src.utils.distance import route_length_fast
from src.utils.neighborhoods_numba import neighbor_cost

# ALGORYTM IHC — ITERATIVE HILL CLIMBING
# ---------------------------------------
# Algorytm lokalnego przeszukiwania wykorzystywany do rozwiązywania
# problemu komiwojażera (TSP). Polega na wielokrotnym uruchomieniu
# wspinaczki lokalnej (Hill Climbing), z których każda rozpoczyna się
# od nowej, losowej permutacji miast. W każdej wspinaczce trasa jest
# stopniowo ulepszana poprzez przechodzenie do rozwiązania znajdującego
# się w jego sąsiedztwie, o ile prowadzi ono do obniżenia kosztu.
#
# Każde lokalne przeszukiwanie kończy się po osiągnięciu lokalnego
# optimum lub po określonej liczbie iteracji bez poprawy. Spośród
# wszystkich uzyskanych rozwiązań wybierane jest to o najmniejszym
# koszcie.
#
# Złożoność obliczeniowa: O(n_starts · max_iter · koszt_sąsiedztwa)
# ---------------------------------------------------------------

# NUMBA — pojedyncza wspinaczka lokalna
@njit(cache=True, parallel=True, fastmath=True)
def hill_climb_numba(distance_matrix, route, max_iter, stop_no_improve, neighbor_fn_id):
    """
    Hill Climb (wersja przyspieszona przez Numba)
    --------------------------------------------
    Funkcja wykonuje jedną wspinaczkę lokalną rozpoczynając
    od podanej trasy startowej. W każdej iteracji generuje
    sąsiada bieżącego rozwiązania i przechodzi do niego tylko
    wtedy, gdy ma on mniejszy koszt. Przeszukiwanie kończy się,
    gdy przez określoną liczbę kroków nie pojawi się żadna
    poprawa lub gdy zostanie osiągnięty limit iteracji.

    Parametry:
        distance_matrix : np.ndarray (n x n)
            Macierz odległości między miastami. distance_matrix[i,j]
            określa koszt przejścia z miasta i do j.

        route : np.ndarray (n)
            Trasa startowa, będąca permutacją indeksów miast.
            Jest to punkt wyjścia dla lokalnego przeszukiwania.

        max_iter : int
            Maksymalna liczba iteracji algorytmu.

        stop_no_improve : int
            Maksymalna liczba kolejnych iteracji bez poprawy.
            Po osiągnięciu tej wartości wspinaczka zostaje przerwana.

        neighbor_fn_id : int
            Identyfikator operatora sąsiedztwa:
              0 - swap
              1 - two-opt
              2 - insert

    Zwraca:
        best_route : np.ndarray
            Najlepsza znaleziona trasa.
        best_cost : float
            Koszt tej trasy.
    """

    # ustawienie bieżącego rozwiązania jako startowego
    best_route = route.copy()
    best_cost = route_length_fast(distance_matrix, best_route)

    # licznik iteracji bez poprawy
    no_improve = 0

    # główna pętla wspinaczki lokalnej
    for _ in range(max_iter):

        # wygenerowanie sąsiada bieżącego rozwiązania
        # Jest to zależne od neighbor_fn_id, czyli metody jaka została wybrana do dobierania sąsiada
        candidate, candidate_cost = neighbor_cost(
            distance_matrix,
            best_route,
            neighbor_fn_id
        )

        # jeśli sąsiad jest lepszy to przechodzimy do niego
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_route = candidate
            no_improve = 0
        else:
            # jeśli brak poprawy to zwiększamy licznik stagnacji
            no_improve += 1

        # jeśli zbyt długo nie było postępu to kończymy
        if no_improve >= stop_no_improve:
            break

    return best_route, best_cost


# GŁÓWNA FUNKCJA ROZWIĄZUJĄCA TSP
def solve_tsp(distance_matrix, params):
    """
    Iterative Hill Climbing (IHC)
    -----------------------------
    Funkcja oblicza wielokrotną wspinaczkę lokalną (Hill Climbing)
    z losowymi punktami startowymi. Każde uruchomienie HC rozpoczyna
    się od nowej, losowej permutacji miast i wykonywane jest do momentu
    osiągnięcia lokalnego optimum. Z wszystkich restartów wybierane jest
    rozwiązanie o najmniejszym koszcie.

    Parametry:
        distance_matrix : np.ndarray (n x n)
            Macierz odległości między miastami.
            distance_matrix[i,j] określa koszt przejścia z miasta i
            do j. Na tej podstawie obliczane są długości tras.

        params : dict
            Słownik parametrów sterujących przebiegiem algorytmu:
              'n_starts' : liczba restartów algorytmu (int).
                  Każdy restart rozpoczyna nową wspinaczkę od
                  losowej trasy startowej.
              'max_iter' : maksymalna liczba iteracji jednej wspinaczki (int).
              'stop_no_improve' : limit kolejnych iteracji bez poprawy (int).
                  Po jego osiągnięciu aktualna wspinaczka jest kończona.
              'neighborhood_type' : typ operatora sąsiedztwa (str),
                  jeden z: "swap", "two_opt", "insert".

    Zwraca:
        best_route : np.ndarray
            Najlepsza znaleziona trasa.
        best_cost : float
            Koszt tej trasy.
        runtime : float
            Czas działania algorytmu w sekundach.
        meta : dict
            Słownik z przekazanymi parametrami wykonania.
    """

    start_time = time.time()
    n = distance_matrix.shape[0]

    # odczyt parametrów wejściowych oraz wartości domyślnych
    n_starts = int(params.get("n_starts", 10))
    max_iter = int(params.get("max_iter", 500))
    stop_no_improve = int(params.get("stop_no_improve", 50))
    neighborhood_type = params.get("neighborhood_type", "swap")

    # zamiana nazwy operatora sąsiedztwa na kod liczbowy dla Numba
    # Numba:
    # - nie przyjmuje funkcji jako argumentów (chyba że to funkcje numbowe),
    # - nie potrafi operować na słownikach / Pythonowych wywołaniach,
    # - nie radzi sobie z dynamicznym wyborem funkcji.
    neighborhood_map = {"swap": 0, "two_opt": 1, "insert": 2}
    neighbor_fn_id = neighborhood_map.get(neighborhood_type, 0)

    # najlepsze globalne rozwiązanie ze wszystkich restartów
    best_route = None
    best_cost = np.inf

    # wielokrotne losowe restarty i uruchomienia HC
    for _ in range(n_starts):

        # losowa trasa startowa dla bieżącego uruchomienia HC
        route = np.random.permutation(n)

        # uruchomienie pojedynczej wspinaczki
        local_route, local_cost = hill_climb_numba(
            distance_matrix,
            route,
            max_iter,
            stop_no_improve,
            neighbor_fn_id
        )

        # aktualizacja najlepszego globalnego rozwiązania
        if local_cost < best_cost:
            best_cost = local_cost
            best_route = local_route.copy()

    # obliczenie łącznego czasu działania
    runtime = time.time() - start_time

    # parametry wykonania
    meta = {
        "n_starts": n_starts,
        "max_iter": max_iter,
        "stop_no_improve": stop_no_improve,
        "neighborhood_type": neighborhood_type,
    }

    return best_route, best_cost, runtime, meta