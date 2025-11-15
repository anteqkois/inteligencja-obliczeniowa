import numpy as np
import time
from src.utils.distance import route_length_fast


# ALGORYTM NAJBLIŻSZEGO SĄSIADA (NEAREST NEIGHBOR - NN)
# ------------------------------------------------------
# Algorytm konstrukcyjny tworzący trasę TSP w sposób zachłanny.
# Rozpoczyna w zadanym mieście startowym, a następnie w każdej
# iteracji wybiera najbliższe nieodwiedzone miasto. Proces trwa
# do momentu odwiedzenia wszystkich miast, po czym następuje
# powrót do miasta początkowego. Metoda jest szybka i prosta,
# ale nie gwarantuje znalezienia rozwiązania optymalnego.
#
# Złożoność obliczeniowa: O(n²)
# ------------------------------------------------------


def solve_tsp(distance_matrix: np.ndarray, params: dict):
    """
    Nearest Neighbor (NN)
    ---------------------
    Implementacja klasycznego algorytmu Najbliższego Sąsiada dla TSP.
    Algorytm rozpoczyna budowę trasy w podanym mieście startowym i
    kolejno dołącza najbliższe nieodwiedzone miasto. Na końcu
    dopełnia cykl, wracając do miasta początkowego.

    Parametry:
        distance_matrix : np.ndarray (n x n)
            Macierz odległości pomiędzy miastami. Element
            distance_matrix[i,j] oznacza koszt przejścia z miasta i
            do j. Na podstawie tej macierzy obliczana jest długość trasy.

        params : dict
            Zestaw parametrów sterujących:
              'start_city' : indeks miasta początkowego (int).
                  Określa, w którym mieście algorytm zaczyna budowę trasy.

    Zwraca:
        route : list[int]
            Kolejność odwiedzanych miast w trakcie przeszukiwania.
        cost : float
            Całkowity koszt trasy, łącznie z powrotem do miasta startowego.
        runtime : float
            Czas wykonania algorytmu w sekundach.
        meta : dict
            Informacje pomocnicze związane z wykonaniem algorytmu.
    """

    # pomiar czasu rozpoczęcia
    start_time = time.time()

    # liczba miast
    n = distance_matrix.shape[0]

    # odczyt parametru miasta startowego
    start_city = int(params.get("start_city", 0))

    # tablica oznaczająca odwiedzone miasta
    visited = np.zeros(n, dtype=bool)

    # trasa rozpoczyna się w mieście startowym
    route = [start_city]
    visited[start_city] = True
    current = start_city

    # główna pętla budowania trasy
    for _ in range(n - 1):

        # odległości od bieżącego miasta
        dists = distance_matrix[current].copy()

        # miasta już odwiedzone ustawiamy na nieskończony koszt,
        # aby nie były ponownie wybierane
        dists[visited] = np.inf

        # wybór najbliższego nieodwiedzonego miasta
        next_city = int(np.argmin(dists))

        # aktualizacja trasy i oznaczenie odwiedzin
        route.append(next_city)
        visited[next_city] = True
        current = next_city

    # obliczenie całkowitego kosztu trasy łącznie z powrotem
    cost = route_length_fast(distance_matrix, np.array(route))

    # pomiar czasu wykonania
    runtime = time.time() - start_time

    # dane informacyjne
    meta = {
        "start_city": start_city,
        "n_cities": n,
        "method": "nearest_neighbor"
    }

    return route, cost, runtime, meta