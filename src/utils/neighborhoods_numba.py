import numpy as np
from numba import njit
from src.utils.distance import route_length_fast


# ============================================================
# NUMBA: OPERATORY SĄSIEDZTWA (dla IHC, SA, TS)
# ============================================================
# Każdy operator generuje nową trasę na podstawie obecnej.
# Funkcje są kompatybilne z Numba (brak list, insertów itp.)
# ============================================================


@njit(cache=True)
def neighbor_swap(route):
    """Losowy ruch: zamienia miejscami dwa miasta."""
    n = len(route)
    i, j = np.random.randint(0, n), np.random.randint(0, n)
    while i == j:
        j = np.random.randint(0, n)
    new_route = route.copy()
    tmp = new_route[i]
    new_route[i] = new_route[j]
    new_route[j] = tmp
    return new_route


@njit(cache=True)
def neighbor_two_opt(route):
    """Losowy ruch: odwraca fragment trasy (2-opt)."""
    n = len(route)
    i, j = np.random.randint(0, n), np.random.randint(0, n)
    if i > j:
        i, j = j, i
    if i == j:
        return route.copy()
    new_route = route.copy()
    new_route[i:j] = new_route[i:j][::-1]
    return new_route

@njit(cache=True)
def neighbor_insert(route):
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

@njit(cache=True)
def neighbor_cost(distance_matrix, route, neighbor_fn_id):
    """
    Zwraca nową trasę i jej koszt.
    neighbor_fn_id:
      0 → swap
      1 → two_opt
      2 → insert
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