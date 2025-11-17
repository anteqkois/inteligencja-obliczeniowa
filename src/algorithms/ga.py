import numpy as np
import time
import random
from src.utils.distance import route_length_fast
from src.utils.neighborhoods_numba import get_neighbor_function

# ALGORYTM GENETYCZNY (GA)
# ------------------------------------------------------
# Metaheurystyka inspirowana ewolucją biologiczną, działająca
# na populacji rozwiązań. GA w każdej iteracji tworzy nowe pokolenie
# poprzez:
#   - selekcję rodziców (3 metody: tournament / roulette / ranking),
#   - krzyżowanie (3 metody: OX / PMX / CX),
#   - mutację (swap / insert / two_opt),
#   - elitaryzm (najlepszy osobnik przechodzi dalej).
#
# GA umożliwia eksplorację przestrzeni TSP w sposób stochastyczny
# oraz różnorodny, działając globalnie w odróżnieniu od lokalnych
# metod takich jak IHC czy TS.
#
# Złożoność: O(population_size · generations · n)
# ------------------------------------------------------


# SELEKCJA
def selection_tournament(population, costs, k=3):
    """
    TURNIEJOWA (Tournament selection)
    ---------------------------------
    - losujemy k osobników
    - wybieramy najlepszego (najmniejszy koszt)
    """
    n = len(population)
    idx = np.random.choice(n, k, replace=False)
    best_idx = idx[np.argmin(costs[idx])]
    return population[best_idx]


def selection_roulette(population, costs):
    """
    RULETKA (Roulette wheel selection)
    ----------------------------------
    - każdy osobnik ma wagę 1/cost
    - im lepszy, tym większa szansa wyboru
    """
    fitness = 1.0 / (costs + 1e-9)
    probs = fitness / fitness.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


def selection_ranking(population, costs):
    """
    RANKINGOWA (Ranking selection)
    ------------------------------
    - sortujemy osobniki wg kosztu
    - im wyższa pozycja w rankingu, tym większa szansa na wybór
    """
    order = np.argsort(costs)
    ranks = np.arange(1, len(population) + 1)
    probs = ranks / ranks.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[order[idx]]


SELECTION_MAP = {
    "tournament": selection_tournament,
    "roulette": selection_roulette,
    "ranking": selection_ranking,
}


# KRZYŻOWANIA
def crossover_OX(p1, p2):
    """
    OX - ORDER CROSSOVER
    ====================
    Idea:
        - zachowuje kolejność z p1 na wycinku [i:j]
        - pozostałe miasta wstawia wg kolejności z p2

    Dlaczego działa:
        - nie psuje struktury permutacji
        - zachowuje relacje kolejnościowe z obu rodziców
    """
    n = len(p1)
    i, j = sorted(random.sample(range(n), 2))

    child = [None] * n
    child[i:j] = p1[i:j]

    pos = j
    for city in p2:
        if city not in child:
            if pos == n:
                pos = 0
            child[pos] = city
            pos += 1

    return child


def crossover_PMX(p1, p2):
    """
    PMX - PARTIALLY MATCHED CROSSOVER
    =================================
    Idea:
        - zachowuje częściowe mapowanie pozycji między rodzicami
        - elementy spoza segmentu są mapowane tak, aby nie tworzyć duplikatów

    Dlaczego działa:
        - respektuje relacje pozycyjne
        - zachowuje dopasowanie strukturalne między p1 i p2
    """
    n = len(p1)
    i, j = sorted(random.sample(range(n), 2))

    child = [None] * n
    child[i:j] = p1[i:j]

    for k in range(i, j):
        if p2[k] not in child:
            val = p2[k]
            pos = k
            while child[pos] is not None:
                pos = p2.index(p1[pos])
            child[pos] = val

    for k in range(n):
        if child[k] is None:
            child[k] = p2[k]

    return child


def crossover_CX(p1, p2):
    """
    CX - CYCLE CROSSOVER
    ====================
    Idea:
        - Buduje cykle zależności: p2[i] → pozycja w p1 → p2[pos] → ...
        - W cyklu bierzemy wartości z p1, poza cyklem z p2.

    Dlaczego działa:
        - cykle zachowują logiczną strukturę permutacji
        - zapewnia 100% poprawną permutację bez duplikatów
    """
    n = len(p1)
    child = [None] * n

    cycle = []
    idx = 0
    while True:
        cycle.append(idx)
        idx = p1.index(p2[idx])
        if idx in cycle:
            break

    for i in range(n):
        if i in cycle:
            child[i] = p1[i]
        else:
            child[i] = p2[i]

    return child


CROSSOVER_MAP = {
    "OX": crossover_OX,
    "PMX": crossover_PMX,
    "CX": crossover_CX,
}


# MUTACJE
# Mutacja = wykonanie JEDNEGO ruchu sąsiedztwa
# korzystamy z gotowych operatorów:
#
# get_neighbor_function("swap")    - neighbor_swap(route)
# get_neighbor_function("insert")  - neighbor_insert(route)
# get_neighbor_function("two_opt") - neighbor_two_opt(route)
#
# Ruch wykorzystywany jako mutacja to dokładnie to samo,
# co "generowanie sąsiada" w SA / TS / IHC.


def apply_mutation(route_list, mutation_type):
    """
    Odpowiada za wykonanie mutacji używając Twoich gotowych operatorów.
    """
    route_np = np.array(route_list)
    mut_fn = get_neighbor_function(mutation_type)
    new_route = mut_fn(route_np)  # operator Numbo-Python
    return new_route.tolist()


# GŁÓWNA FUNKCJA GA
def solve_tsp(distance_matrix, params):
    """
    Genetic Algorithm (GA)
    ----------------------
    Pełna implementacja GA dla TSP z:
        - 3 selekcjami
        - 3 krzyżowaniami
        - mutacjami swap / insert / two_opt
        - elitaryzmem

    Zwraca:
        best_route : najlepsza znaleziona permutacja
        best_cost  : koszt trasy
        runtime    : czas działania
        meta       : parametry wykonania
    """

    start_time = time.time()

    n = distance_matrix.shape[0]

    # parametry
    pop_size = int(params.get("population_size", 80))
    generations = int(params.get("generations", 300))
    selection_name = params.get("selection", "tournament")
    crossover_name = params.get("crossover", "OX")
    mutation_type = params.get("mutation_type", "swap")
    mutation_prob = float(params.get("mutation_prob", 0.1))

    select_fn = SELECTION_MAP[selection_name]
    cross_fn = CROSSOVER_MAP[crossover_name]

    # inicjalizacja populacji
    population = [list(np.random.permutation(n)) for _ in range(pop_size)]
    costs = np.array(
        [route_length_fast(distance_matrix, np.array(r)) for r in population]
    )

    best_idx = np.argmin(costs)
    best_route = population[best_idx].copy()
    best_cost = float(costs[best_idx])

    # PĘTLA GŁÓWNA GA
    for _ in range(generations):

        new_pop = []

        # elita
        new_pop.append(best_route.copy())

        # generowanie potomstwa
        while len(new_pop) < pop_size:
            # selekcja rodziców
            p1 = select_fn(population, costs)
            p2 = select_fn(population, costs)

            # krzyżowanie
            child = cross_fn(p1, p2)

            # mutacja
            if random.random() < mutation_prob:
                child = apply_mutation(child, mutation_type)

            new_pop.append(child)

        population = new_pop
        costs = np.array(
            [route_length_fast(distance_matrix, np.array(r)) for r in population]
        )

        # aktualizacja najlepszego
        idx = np.argmin(costs)
        if costs[idx] < best_cost:
            best_cost = float(costs[idx])
            best_route = population[idx].copy()

    runtime = time.time() - start_time

    meta = {
        "population_size": pop_size,
        "generations": generations,
        "selection": selection_name,
        "crossover": crossover_name,
        "mutation_type": mutation_type,
        "mutation_prob": mutation_prob,
    }

    return best_route, best_cost, runtime, meta
