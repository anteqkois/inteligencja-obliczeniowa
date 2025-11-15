import numpy as np

# OPERATORY SĄSIEDZTWA DLA TSP (wersja Pythonowa)
# ------------------------------------------------
# Zawiera trzy klasyczne operatory używane w heurystykach:
#   • swap     – zamiana dwóch miast miejscami
#   • insert   – wyjęcie miasta i wstawienie go w inne miejsce
#   • two_opt  – odwrócenie fragmentu trasy
#
# Wszystkie funkcje:
#   - tworzą nową trasę (route.copy()), nie modyfikują oryginału
#   - działają w czystym Pythonie / NumPy
#   - są używane m.in. w Tabu Search (wersja bez Numba)
# ------------------------------------------------


def swap(route: np.ndarray) -> np.ndarray:
    """
    Operator sąsiedztwa: SWAP
    -------------------------------------
    Wybiera dwa różne miasta w trasie i zamienia je miejscami.

    Parametry:
        route : np.ndarray (1D)
            Permutacja miast — bieżąca trasa TSP.

    Zwraca:
        np.ndarray
            Nowa trasa po wykonaniu ruchu swap.
    """
    new_route = route.copy()
    i, j = np.random.choice(len(route), 2, replace=False)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def insert(route: np.ndarray) -> np.ndarray:
    """
    Operator sąsiedztwa: INSERT
    -------------------------------------
    Wyjmuje jedno miasto z pozycji i i wstawia je w miejsce j.
    Pozwala na przesuwanie elementów trasy bez jej odwracania.

    Parametry:
        route : np.ndarray (1D)
            Bieżące rozwiązanie TSP.

    Zwraca:
        np.ndarray
            Nowa trasa po wykonaniu ruchu insert.
    """
    new_route = route.copy()
    i, j = np.random.choice(len(route), 2, replace=False)
    city = new_route[i]

    new_route = np.delete(new_route, i)
    new_route = np.insert(new_route, j, city)

    return new_route


def two_opt(route: np.ndarray) -> np.ndarray:
    """
    Operator sąsiedztwa: TWO-OPT
    -------------------------------------
    Odwraca wybrany fragment trasy. Ruch klasyczny dla TSP —
    zmniejsza często liczbę przecięć i poprawia jakość trasy.

    Parametry:
        route : np.ndarray (1D)
            Bieżąca trasa.

    Zwraca:
        np.ndarray
            Nowa trasa po zastosowaniu ruchu two-opt.
    """
    new_route = route.copy()
    i, j = sorted(np.random.choice(len(route), 2, replace=False))
    new_route[i:j] = new_route[i:j][::-1]
    return new_route


def get_neighbor(route: np.ndarray, neighborhood_type: str) -> np.ndarray:
    """
    Wybór sąsiada na podstawie nazwy operatora
    ------------------------------------------
    Wrapper używany w algorytmach działających w Pythonie
    (np. Tabu Search), gdy nie korzystamy z Numba.

    Parametry:
        route : np.ndarray (1D)
            Bieżąca trasa.
        neighborhood_type : str
            "swap", "insert" lub "two_opt".

    Zwraca:
        np.ndarray
            Nowa trasa po wykonaniu wybranego ruchu.
    """
    if neighborhood_type == "swap":
        return swap(route)
    elif neighborhood_type == "insert":
        return insert(route)
    elif neighborhood_type == "two_opt":
        return two_opt(route)

    raise ValueError(f"Nieznany typ sąsiedztwa: {neighborhood_type}")
