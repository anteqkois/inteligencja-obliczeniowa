# EXPERIMENT: GENETIC ALGORITHM (GA)
# Wykonuje serię eksperymentów dla algorytmu genetycznego:
#  - różne rozmiary populacji
#  - różne liczby pokoleń
#  - trzy metody selekcji
#  - trzy metody krzyżowania
#  - trzy typy mutacji (swap / insert / two_opt)
#  - cztery poziomy mutacji (0.05, 0.1, 0.2, 0.3)
#
# Każda kombinacja parametrów jest wykonywana wielokrotnie,
# wyniki są zapisywane do CSV i podsumowywane.

import time
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count

from src.utils.tsp_loader import load_tsp_matrix
from src.utils.result_saver import save_experiment_results
from src.utils.run_single_repeat import run_single_repeat
from src.algorithms.ga import solve_tsp
from src.utils.distance import route_length_fast


# USTAWIENIA EKSPERYMENTU
TSP_FILES = ["Dane_TSP_48.xlsx", "Dane_TSP_76.xlsx", "Dane_TSP_127.xlsx"]

PARAM_GRID_GA = {
    "population_size": [20, 50, 100, 300],
    "generations": [100, 200, 500, 1_000],
    "selection": ["tournament", "roulette", "ranking"],
    "crossover": ["OX", "PMX", "CX"],
    "mutation_type": ["swap", "insert", "two_opt"],
    "mutation_prob": [0.05, 0.1, 0.2, 0.3],
}

REPEATS = 5



if __name__ == "__main__":
    results = []


    # ROZGRZEWANIE NUMBA (JIT)
    print("Rozgrzewanie (kompilacja JIT route_length_fast)...")
    D_warm = load_tsp_matrix(TSP_FILES[0])
    _ = route_length_fast(D_warm, np.random.permutation(D_warm.shape[0]))
    _ = solve_tsp(D_warm, {})
    print("Kompilacja zakończona.\n")


    # GENEROWANIE KOMBINACJI
    all_combos = list(
        itertools.product(
            PARAM_GRID_GA["population_size"],
            PARAM_GRID_GA["generations"],
            PARAM_GRID_GA["selection"],
            PARAM_GRID_GA["crossover"],
            PARAM_GRID_GA["mutation_type"],
            PARAM_GRID_GA["mutation_prob"],
        )
    )

    total = len(all_combos) * len(TSP_FILES)
    counter = 0


    # GŁÓWNA PĘTLA EKSPERYMENTU
    start_total = time.perf_counter()

    for tsp_file in TSP_FILES:
        print(f"\nInstancja: {tsp_file}")
        D = load_tsp_matrix(tsp_file)

        for pop_size, generations, selection, crossover, mut_type, mut_prob in all_combos:
            counter += 1

            print(
                f"[{counter}/{total}] "
                f"population={pop_size}, generations={generations}, "
                f"selection={selection}, crossover={crossover}, "
                f"mutation_type={mut_type}, mutation_prob={mut_prob}"
            )

            params = {
                "population_size": pop_size,
                "generations": generations,
                "selection": selection,
                "crossover": crossover,
                "mutation_type": mut_type,
                "mutation_prob": mut_prob,
            }

            # multiprocessing — równoległe powtórzenia
            with Pool(processes=cpu_count()) as pool:
                parallel_jobs = [(solve_tsp, D, params) for _ in range(REPEATS)]
                results_parallel = pool.map(run_single_repeat, parallel_jobs)

            costs = [c for c, _, _ in results_parallel]
            routes = [r for _, r, _ in results_parallel]
            runtimes = [t for _, _, t in results_parallel]

            # najlepsza trasa z powtórzeń
            min_cost = min(costs)
            best_route = routes[costs.index(min_cost)]
            route_str = "-".join(map(str, best_route))

            # zapis wyniku
            results.append(
                {
                    "instance": tsp_file,
                    "population_size": pop_size,
                    "generations": generations,
                    "selection": selection,
                    "crossover": crossover,
                    "mutation_type": mut_type,
                    "mutation_prob": mut_prob,
                    "mean_cost": round(np.mean(costs), 3),
                    "min_cost": round(min_cost, 3),
                    "mean_runtime": np.mean(runtimes),
                    "min_route": route_str,
                }
            )


    # ZAPIS i PODSUMOWANIE
    end_total = time.perf_counter()
    elapsed = end_total - start_total

    print(f"\nŁączny czas eksperymentów: {elapsed/60:.2f} min ({elapsed:.2f} sek)\n")

    df = pd.DataFrame(results)
    save_experiment_results(df, time_seconds=int(elapsed), subfolder="GA")

    print("\nNajlepsze parametry dla każdej instancji:")
    instances = df["instance"].unique()

    for inst in instances:
        sub = df[df["instance"] == inst]
        best_row = sub.loc[sub["mean_cost"].idxmin()]
        print(f"\n{inst}")
        print(f"odległość {best_row['min_cost']} = {best_row.to_dict()}")