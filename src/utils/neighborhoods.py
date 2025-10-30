import numpy as np

# ============================================================
# MODUŁ GENEROWANIA SĄSIADÓW (dla TSP)
# ============================================================
# Zawiera trzy rodzaje ruchów używanych w heurystykach lokalnych:
#   1. swap      – zamiana dwóch losowych miast
#   2. insert    – przeniesienie miasta w inne miejsce
#   3. two_opt   – odwrócenie fragmentu trasy
#
# Każdy z operatorów zwraca NOWĄ trasę (nie modyfikuje oryginału).
# ============================================================


def swap(route: np.ndarray) -> np.ndarray:
    """Zamienia miejscami dwa losowe miasta w trasie."""
    new_route = route.copy()
    i, j = np.random.choice(len(route), 2, replace=False)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def insert(route: np.ndarray) -> np.ndarray:
    """Wyjmuje jedno miasto i wstawia je w inne miejsce."""
    new_route = route.copy()
    i, j = np.random.choice(len(route), 2, replace=False)
    city = new_route[i]
    new_route = np.delete(new_route, i)
    new_route = np.insert(new_route, j, city)
    return new_route


def two_opt(route: np.ndarray) -> np.ndarray:
    """Odwraca losowy fragment trasy (klasyczny ruch 2-opt)."""
    new_route = route.copy()
    i, j = sorted(np.random.choice(len(route), 2, replace=False))
    new_route[i:j] = new_route[i:j][::-1]
    return new_route


def get_neighbor(route: np.ndarray, neighborhood_type: str) -> np.ndarray:
    """Zwraca sąsiada zgodnie z wybraną metodą."""
    if neighborhood_type == "swap":
        return swap(route)
    elif neighborhood_type == "insert":
        return insert(route)
    elif neighborhood_type == "two_opt":
        return two_opt(route)
    else:
        raise ValueError(f"Nieznany typ sąsiedztwa: {neighborhood_type}")
