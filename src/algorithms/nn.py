import numpy as np
import time
from src.utils.distance import route_length_fast


# ============================================================
# ALGORYTM NAJBLIŻSZEGO SĄSIADA (NEAREST NEIGHBOR - NN)
# ============================================================
# Klasyczny algorytm konstrukcyjny do rozwiązania problemu TSP.
# Zaczynamy w wybranym mieście i w każdej iteracji wybieramy
# najbliższe nieodwiedzone miasto. Proces trwa do odwiedzenia
# wszystkich miast, po czym wracamy do miasta startowego.
#
# Parametry wejściowe:
#   - distance_matrix : macierz odległości (numpy.ndarray, n x n)
#   - params : dict zawierający:
#       • start_city – indeks miasta początkowego (int)
#
# Wyniki:
#   - route : lista odwiedzonych miast w kolejności
#   - cost : długość trasy (łącznie z powrotem do startu)
#   - runtime : czas wykonania algorytmu [s]
#   - meta : słownik z informacjami o parametrze startowym
# ============================================================


def solve_tsp(distance_matrix: np.ndarray, params: dict):
    """
    Klasyczna implementacja algorytmu Najbliższego Sąsiada (NN)
    -----------------------------------------------------------
    Tworzy trasę komiwojażera rozpoczynając od wskazanego miasta
    i iteracyjnie wybiera najbliższe nieodwiedzone miasto.

    Złożoność obliczeniowa: O(n²)
    """
    # --- Pomiar czasu rozpoczęcia
    start_time = time.time()

    # --- Pobranie danych wejściowych
    n = distance_matrix.shape[0]          # liczba miast
    start_city = int(params.get("start_city", 0))  # indeks miasta startowego

    # --- Inicjalizacja pomocniczych struktur
    visited = np.zeros(n, dtype=bool)     # tablica odwiedzin (True = odwiedzone)
    route = [start_city]                  # kolejność odwiedzanych miast
    visited[start_city] = True            # oznacz miasto startowe jako odwiedzone
    current = start_city                  # aktualne miasto

    # --- Główna pętla budowania trasy
    for _ in range(n - 1):
        # Kopiujemy odległości z bieżącego miasta
        dists = distance_matrix[current].copy()

        # Miasta już odwiedzone ustawiamy na "nieskończony" dystans,
        # żeby nie zostały ponownie wybrane
        dists[visited] = np.inf

        # Wybieramy indeks najbliższego nieodwiedzonego miasta
        next_city = int(np.argmin(dists))

        # Aktualizujemy trasę i oznaczamy odwiedziny
        route.append(next_city)
        visited[next_city] = True
        current = next_city

    # --- Obliczenie całkowitej długości trasy (łącznie z powrotem)
    cost = route_length_fast(distance_matrix, np.array(route))

    # --- Pomiar czasu zakończenia
    runtime = time.time() - start_time

    # --- Dane meta pomocne przy raportowaniu
    meta = {
        "start_city": start_city,
        "n_cities": n,
        "method": "nearest_neighbor"
    }

    # --- Zwracamy komplet wyników
    return route, cost, runtime, meta
