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


@njit(cache=True)
def neighbor_swap_delta(route):
    """
    Operator SWAP dla wersji delta-cost.
    ------------------------------------
    Losowo wybiera dwie pozycje (i, j) i zamienia je w trasie.
    Zwraca także indeksy i, j — potrzebne do szybkiego
    obliczenia różnicy kosztu (delta).

    Zwraca:
        new_route : zmodyfikowana trasa
        i, j      : indeksy zamienionych miast
    """
    n = len(route)

    # losujemy dwa różne indeksy
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    while i == j:
        j = np.random.randint(0, n)

    # wykonujemy swap
    new_route = route.copy()
    tmp = new_route[i]
    new_route[i] = new_route[j]
    new_route[j] = tmp

    return new_route, i, j

@njit(cache=True)
def neighbor_insert_delta(route):
    """
    Operator INSERT dla delta-cost.
    -------------------------------
    Wyjmuje miasto z pozycji i i wstawia je w pozycję j.
    Zwraca również (i, j), aby można było policzyć,
    które krawędzie zostały zmienione.

    Zmienia się tutaj 4–6 krawędzi zależnie od kierunku ruchu.
    """
    n = len(route)

    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    while i == j:
        j = np.random.randint(0, n)

    new_route = route.copy()
    city = new_route[i]

    # przesuwamy pozostałe miasta
    if i < j:
        for k in range(i, j):
            new_route[k] = new_route[k + 1]
        new_route[j] = city
    else:
        for k in range(i, j, -1):
            new_route[k] = new_route[k - 1]
        new_route[j] = city

    return new_route, i, j

@njit(cache=True)
def neighbor_two_opt_delta(route):
    """
    Operator TWO-OPT dla delta-cost.
    --------------------------------
    Odwraca fragment trasy od i do j.
    Zmienia dokładnie 2 główne krawędzie:
        (i-1 -> i)  i  (j-1 -> j)
    które po ruchu stają się:
        (i-1 -> j-1) i (i -> j)

    Dzięki zwróceniu (i, j) możemy policzyć delta w O(1).
    """
    n = len(route)

    i = np.random.randint(0, n)
    j = np.random.randint(0, n)

    # i < j
    if i > j:
        i, j = j, i

    # brak zmiany
    if i == j:
        return route.copy(), i, j

    new_route = route.copy()
    new_route[i:j] = new_route[i:j][::-1]

    return new_route, i, j

@njit(cache=True)
def neighbor_cost_delta_numba(distance_matrix, route, current_cost, neighbor_fn_id):
    """
    Wersja generatora sąsiada oparta o DELTA-COST.
    ------------------------------------
    Zamiast liczyć koszt całej trasy od zera (O(n)),
    porównujemy tylko fragmenty które uległy zmianie (O(1)).

    Zwraca:
        new_route : nowa trasa
        new_cost  : zaktualizowany koszt (current_cost + delta)
    """

    # 1. Generujemy nowego sąsiada oraz pobieramy indeksy (i, j)
    if neighbor_fn_id == 0:      # SWAP
        new_route, i, j = neighbor_swap_delta(route)

    elif neighbor_fn_id == 1:    # TWO-OPT
        new_route, i, j = neighbor_two_opt_delta(route)

    else:                         # INSERT
        new_route, i, j = neighbor_insert_delta(route)

    n = len(route)

    # 2. W zależności od rodzaju ruchu liczymy DELTA

    # SWAP
    if neighbor_fn_id == 0:
        """
        SWAP zmienia krawędzie wokół i oraz j.
        Usuwamy stare (4 lub 6) i dodajemy nowe.
        """

        # sąsiedzi i
        im1 = route[i - 1] if i > 0 else route[n - 1]
        ip1 = route[i + 1] if i < n - 1 else route[0]

        # sąsiedzi j
        jm1 = route[j - 1] if j > 0 else route[n - 1]
        jp1 = route[j + 1] if j < n - 1 else route[0]

        delta = 0.0

        # odejmujemy stare krawędzie
        delta -= distance_matrix[im1, route[i]]
        delta -= distance_matrix[route[i], ip1]
        delta -= distance_matrix[jm1, route[j]]
        delta -= distance_matrix[route[j], jp1]

        # dodajemy nowe krawędzie
        delta += distance_matrix[im1, new_route[i]]
        delta += distance_matrix[new_route[i], ip1]
        delta += distance_matrix[jm1, new_route[j]]
        delta += distance_matrix[new_route[j], jp1]

        new_cost = current_cost + delta
        return new_route, new_cost

    # TWO OPT
    if neighbor_fn_id == 1:
        """
        TWO-OPT odwraca fragment [i, j).
        Zmieniają się TYLKO 2 krawędzie:
            przed ruchem: (i-1,i) oraz (j-1,j)
            po ruchu:      (i-1,j-1) oraz (i,j)
        """

        if i == j:
            return route, current_cost

        im1 = route[i - 1] if i > 0 else route[n - 1]

        delta = 0.0
        delta -= distance_matrix[im1, route[i]]
        delta -= distance_matrix[route[j - 1], route[j]]
        delta += distance_matrix[im1, route[j - 1]]
        delta += distance_matrix[route[i], route[j]]

        new_cost = current_cost + delta
        return new_route, new_cost

    # INSERT
    """
    INSERT przenosi jedno miasto z pozycji i → j.
    Zmienia 4 do 6 krawędzi.
    """

    im1 = route[i - 1] if i > 0 else route[n - 1]
    ip1 = route[i + 1] if i < n - 1 else route[0]
    jm1 = route[j - 1] if j > 0 else route[n - 1]
    jp1 = route[j + 1] if j < n - 1 else route[0]

    delta = 0.0

    # usuwamy stare krawędzie wokół pozycji i
    delta -= distance_matrix[im1, route[i]]
    delta -= distance_matrix[route[i], ip1]
    delta += distance_matrix[im1, ip1]

    # usuwamy i dodajemy nowe wokół pozycji j
    delta -= distance_matrix[jm1, route[j]]
    delta += distance_matrix[jm1, route[i]]
    delta += distance_matrix[route[i], route[j]]

    new_cost = current_cost + delta
    return new_route, new_cost
