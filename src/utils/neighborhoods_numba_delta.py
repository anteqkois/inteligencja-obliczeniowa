import numpy as np
from numba import njit

#                    OPERATORY SĄSIEDZTWA DLA TSP
# Ten moduł zawiera trzy podstawowe operatory dla problemu komiwojażera:
#   1. SWAP     – zamiana dwóch miast miejscami
#   2. TWO-OPT  – odwrócenie fragmentu trasy (klasyczny operator)
#   3. INSERT   – przesunięcie miasta na inną pozycję
#
# WAŻNE: Wszystkie funkcje zakładają ZAMKNIĘTĄ TRASĘ (cykl Hamiltona),
# czyli ostatnie miasto łączy się z pierwszym: route[n-1] -> route[0]
#
# Każda funkcja delta_* oblicza ZMIANĘ kosztu (nie pełny koszt!),
# co jest o wiele szybsze niż przeliczanie całej trasy od nowa.


@njit(cache=True)
def neighbor_cost_delta_numba(distance_matrix, route, current_cost, fn_id):
    """
    Wersja zoptymalizowana - używa NumPy slicing zamiast pętli.
    """
    n = len(route)

    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    while i == j:
        j = np.random.randint(0, n)

    new_route = route.copy()

    if fn_id == 0:  # SWAP
        new_route[i], new_route[j] = new_route[j], new_route[i]
        delta = delta_swap(distance_matrix, route, i, j)

    elif fn_id == 1:  # TWO-OPT
        if i > j:
            i, j = j, i
        # NumPy slicing jest szybsze niż [::-1] dla Numby
        new_route[i:j] = route[i:j][::-1]
        delta = delta_two_opt(distance_matrix, route, i, j)

    else:  # INSERT
        # To jest najtrudniejsze do optymalizacji bez pętli
        # Ale możemy zminimalizować liczbę operacji
        a = route[i]
        
        if i < j:
            # Zamiast pętli, użyj slicing (Numba to dobrze optymalizuje)
            new_route[i:j] = route[i+1:j+1]
            new_route[j] = a
        else:
            # Analogicznie
            new_route[j+1:i+1] = route[j:i]
            new_route[j] = a
            
        delta = delta_insert(distance_matrix, route, i, j)

    return new_route, current_cost + delta


# Funkcje delta z poprzedniego kodu (bez zmian)
@njit(cache=True)
def delta_swap(distance_matrix, route, i, j):
    if i == j:
        return 0.0
    n = len(route)
    if i > j:
        i, j = j, i
    a = route[i]
    b = route[j]
    a_prev = route[i - 1] if i > 0 else route[n - 1]
    a_next = route[i + 1] if i < n - 1 else route[0]
    b_prev = route[j - 1] if j > 0 else route[n - 1]
    b_next = route[j + 1] if j < n - 1 else route[0]
    delta = 0.0
    if j == i + 1:
        delta -= distance_matrix[a_prev, a]
        delta -= distance_matrix[a, b]
        delta -= distance_matrix[b, b_next]
        delta += distance_matrix[a_prev, b]
        delta += distance_matrix[b, a]
        delta += distance_matrix[a, b_next]
        return delta
    if i == 0 and j == n - 1:
        delta -= distance_matrix[b_prev, b]
        delta -= distance_matrix[b, a]
        delta -= distance_matrix[a, a_next]
        delta += distance_matrix[b_prev, a]
        delta += distance_matrix[a, b]
        delta += distance_matrix[b, a_next]
        return delta
    delta -= distance_matrix[a_prev, a]
    delta -= distance_matrix[a, a_next]
    delta -= distance_matrix[b_prev, b]
    delta -= distance_matrix[b, b_next]
    delta += distance_matrix[a_prev, b]
    delta += distance_matrix[b, a_next]
    delta += distance_matrix[b_prev, a]
    delta += distance_matrix[a, b_next]
    return delta


@njit(cache=True)
def delta_two_opt(distance_matrix, route, i, j):
    if i == j:
        return 0.0
    n = len(route)
    if i > j:
        i, j = j, i
    im1 = route[i - 1] if i > 0 else route[n - 1]
    ip1 = route[i]
    jm1 = route[j - 1]
    jp1 = route[j] if j < n else route[0]
    delta = 0.0
    delta -= distance_matrix[im1, ip1]
    delta -= distance_matrix[jm1, jp1]
    delta += distance_matrix[im1, jm1]
    delta += distance_matrix[ip1, jp1]
    return delta


@njit(cache=True)
def delta_insert(distance_matrix, route, i, j):
    if i == j:
        return 0.0
    n = len(route)
    a = route[i]
    a_prev = route[(i - 1) % n]
    a_next = route[(i + 1) % n]
    delta = 0.0
    delta -= distance_matrix[a_prev, a]
    delta -= distance_matrix[a, a_next]
    delta += distance_matrix[a_prev, a_next]
    
    if i < j:
        left = route[j]
        right_idx = (j + 1) % n
        if right_idx == i:
            right = a_next
        else:
            right = route[right_idx]
        delta -= distance_matrix[left, right]
        delta += distance_matrix[left, a]
        delta += distance_matrix[a, right]
    else:
        left_idx = (j - 1) % n
        if left_idx == i:
            left = a_prev
        else:
            left = route[left_idx]
        right = route[j]
        delta -= distance_matrix[left, right]
        delta += distance_matrix[left, a]
        delta += distance_matrix[a, right]
    
    return delta