import numpy as np
from numba import njit

def route_length_np(distance_matrix, route):
    return np.sum(distance_matrix[route[:-1], route[1:]]) + distance_matrix[route[-1], route[0]]

# use optymalized version
@njit(cache=True)
def route_length_fast(distance_matrix, route):
    total = 0.0
    n = len(route)
    for i in range(n - 1):
        total += distance_matrix[route[i], route[i + 1]]
    total += distance_matrix[route[-1], route[0]]
    return total
