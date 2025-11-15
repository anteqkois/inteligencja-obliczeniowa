import numpy as np
from numba import njit

# FUNKCJE DO OBLICZANIA DŁUGOŚCI TRASY TSP
# Zawiera dwie wersje:
#   • route_length_np  – wariant czysto NumPy (czytelny, wolniejszy)
#   • route_length_fast – wariant zoptymalizowany, kompilowany
#                         przez Numba (szybki, używany we wszystkich algorytmach)
#
# Obie funkcje obliczają pełną długość cyklu Hamiltona, tzn.
# sumują koszty kolejnych przejść route[i] -> route[i+1],
# a na końcu dodają powrót z ostatniego miasta do pierwszego.
# ------------------------------------------------------------


def route_length_np(distance_matrix, route):
    """
    Obliczanie długości trasy (wersja NumPy)
    ----------------------------------------
    Implementacja wykorzystująca operacje tablicowe NumPy.
    Jest prosta i przejrzysta, ale wolniejsza niż wersja Numba.
    Stosowana głównie do celów kontrolnych i testowych.

    Parametry:
        distance_matrix : np.ndarray (n x n)
            Macierz odległości pomiędzy miastami.
        route : np.ndarray lub lista
            Permutacja indeksów miast tworząca cykl TSP.

    Zwraca:
        float : całkowita długość trasy.
    """
    return (
        np.sum(distance_matrix[route[:-1], route[1:]])
        + distance_matrix[route[-1], route[0]]
    )



@njit(cache=True, parallel=True, fastmath=True)
def route_length_fast(distance_matrix, route):
    """
    Obliczanie długości trasy (wersja szybka -> Numba)
    ------------------------------------------------
    Wariant zoptymalizowany, kompilowany JIT przez Numba.
    Pozwala znacząco przyspieszyć działanie algorytmów opartych
    na wielokrotnym obliczaniu długości trasy (np. HC, SA, TS).

    Parametry:
        distance_matrix : 2D array (n x n)
            Macierz kosztów przejścia pomiędzy miastami.
        route : 1D array
            Permutacja odwiedzanych miast.

    Zwraca:
        float : długość cyklu Hamiltona z powrotem do punktu startowego.
    """
    total = 0.0
    n = len(route)

    # sumowanie kolejnych przejść
    for i in range(n - 1):
        total += distance_matrix[route[i], route[i + 1]]

    # powrót do miasta początkowego
    total += distance_matrix[route[-1], route[0]]

    return total
