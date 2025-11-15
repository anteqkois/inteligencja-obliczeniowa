import numpy as np
from numba import njit
from src.utils.distance import route_length_fast

# OPERATORY SĄSIEDZTWA DLA TSP (Numba)
# ------------------------------------
# Zawiera implementacje trzech podstawowych operatorów:
#   • swap     – zamiana dwóch miast miejscami
#   • two_opt  – odwrócenie fragmentu trasy
#   • insert   – wyjęcie miasta i wstawienie go w inne miejsce
#
# Wszystkie funkcje są zgodne z ograniczeniami Numba:
#   - brak list i konstrukcji Pythonowych
#   - operacje wyłącznie na tablicach NumPy
#
# Operator neighbor_cost pozwala szybko wyliczać koszt
# wygenerowanego sąsiada i wybiera operator na podstawie
# identyfikatora liczbowego (0, 1, 2).
# ------------------------------------


@njit(cache=True, parallel=True, fastmath=True)
def neighbor_swap(route):
    """
    Operator sąsiedztwa: SWAP
    -------------------------
    Generuje nową trasę poprzez zamianę miejscami dwóch
    losowo wybranych miast.

    Parametry:
        route : 1D array
            Bieżąca trasa TSP (permutacja).

    Zwraca:
        new_route : 1D array
            Nowa trasa po wykonaniu ruchu.
    """
    n = len(route)
    i, j = np.random.randint(0, n), np.random.randint(0, n)
    while i == j:
        j = np.random.randint(0, n)

    new_route = route.copy()
    tmp = new_route[i]
    new_route[i] = new_route[j]
    new_route[j] = tmp

    return new_route


@njit(cache=True, parallel=True, fastmath=True)
def neighbor_two_opt(route):
    """
    Operator sąsiedztwa: TWO-OPT
    ----------------------------
    Odwraca losowy fragment trasy. Klasyczny operator do TSP
    poprawiający jakość rozwiązania poprzez redukcję przecięć.

    Parametry:
        route : 1D array
            Bieżąca trasa TSP.

    Zwraca:
        new_route : 1D array
            Trasa po wykonaniu ruchu two-opt.
    """
    n = len(route)
    i, j = np.random.randint(0, n), np.random.randint(0, n)

    if i > j:
        i, j = j, i
    if i == j:
        return route.copy()

    new_route = route.copy()
    new_route[i:j] = new_route[i:j][::-1]

    return new_route


@njit(cache=True, parallel=True, fastmath=True)
def neighbor_insert(route):
    """
    Operator sąsiedztwa: INSERT
    ---------------------------
    Wyjmuje jedno miasto z pozycji 'i' i wstawia je w miejsce 'j'.
    Pozwala przemieszczać fragmenty trasy bez jej odwracania.

    Parametry:
        route : 1D array
            Bieżąca trasa TSP.

    Zwraca:
        new_route : 1D array
            Trasa po wykonaniu ruchu insert.
    """
    n = len(route)
    i, j = np.random.randint(0, n), np.random.randint(0, n)
    while i == j:
        j = np.random.randint(0, n)

    new_route = route.copy()
    city = new_route[i]

    if i < j:
        for k in range(i, j):
            new_route[k] = new_route[k + 1]
        new_route[j] = city
    else:
        for k in range(i, j, -1):
            new_route[k] = new_route[k - 1]
        new_route[j] = city

    return new_route


@njit(cache=True, parallel=True, fastmath=True)
def neighbor_cost(distance_matrix, route, neighbor_fn_id):
    """
    Generator sąsiada + obliczanie kosztu
    -------------------------------------
    Tworzy sąsiada bieżącej trasy na podstawie identyfikatora
    operatora sąsiedztwa, a następnie oblicza jego długość.

    Parametry:
        distance_matrix : 2D array (n x n)
            Macierz odległości.
        route : 1D array
            Bieżąca trasa.
        neighbor_fn_id : int
            0 - swap
            1 - two_opt
            2 - insert

    Zwraca:
        new_route : 1D array
            Wygenerowany sąsiad.
        cost : float
            Długość nowej trasy.
    """
    if neighbor_fn_id == 0:
        new_route = neighbor_swap(route)
    elif neighbor_fn_id == 1:
        new_route = neighbor_two_opt(route)
    elif neighbor_fn_id == 2:
        new_route = neighbor_insert(route)
    else:
        new_route = neighbor_swap(route)

    cost = route_length_fast(distance_matrix, new_route)
    return new_route, cost


def get_neighbor_function(neighborhood_type):
    """
    Wybór operatora sąsiedztwa po nazwie (wersja Pythonowa)
    --------------------------------------------------------
    Używane w algorytmach niedziałających w czystej Numba,
    np. Tabu Search w Pythonie.

    Parametry:
        neighborhood_type : str
            "swap", "two_opt" lub "insert"

    Zwraca:
        callable : funkcja generująca sąsiada
    """
    if neighborhood_type == "swap":
        return neighbor_swap
    elif neighborhood_type == "insert":
        return neighbor_insert
    elif neighborhood_type == "two_opt":
        return neighbor_two_opt

    raise ValueError(f"Nieznany typ sąsiedztwa: {neighborhood_type}")
