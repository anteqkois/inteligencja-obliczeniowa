# %%
# GRASP – SKRYPT EKSPERYMENTALNY
# Uruchamia serię eksperymentów dla algorytmu GRASP z różnymi
# parametrami. Struktura analogiczna do IHC, TS, SA itd.
#
# Wymagane:
#   - solve_tsp() z grasp_numba
#   - run_single_repeat()
#   - save_experiment_results()
#   - load_tsp_matrix()

import time
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count

from src.utils.tsp_loader import load_tsp_matrix
from src.algorithms.grasp_numba import solve_tsp
from src.utils.run_single_repeat import run_single_repeat
from src.utils.result_saver import save_experiment_results


# USTAWIENIA EKSPERYMENTU

TSP_FILES = ["Dane_TSP_48.xlsx", "Dane_TSP_76.xlsx", "Dane_TSP_127.xlsx"]
# TSP_FILES = ["Dane_TSP_48.xlsx"]

PARAM_GRID_GRASP = {
    "alpha": [0.2, 0.4, 0.6, 0.8],
    "iterations": [3_000, 5_000, 8_000, 10_000],
    "neighborhood_type": ["swap", "insert", "two_opt"],
    # parametry local-search (IHC-light)
    "ihc_max_iter": [4_000, 6_000, 8_000],
    "ihc_stop_no_improve": [400, 800, 1_500],
    # "use_delta": [True], # zawsze delta
    "use_delta": [False], # bez delta
}

REPEATS = 5


if __name__ == "__main__":
    results = []

    # ROZGRZANIE

    print("Rozgrzewanie Numba (kompilacja GRASP + IHC-light)...")

    D_tmp = load_tsp_matrix(TSP_FILES[0])

    # poprawna rozgrzewka: solve_tsp wymaga neighborhood_type jako string
    _ = solve_tsp(D_tmp, {
        "alpha": 0.3,
        "iterations": 2,
        "neighborhood_type": "swap",
        "ihc_max_iter": 50,
        "ihc_stop_no_improve": 10,
        "use_delta": False,
    })

    print("Rozgrzewanie zakończone.\n")


    # LISTA KOMBINACJI PARAMETRÓW

    all_combos = list(itertools.product(
        PARAM_GRID_GRASP["alpha"],
        PARAM_GRID_GRASP["iterations"],
        PARAM_GRID_GRASP["neighborhood_type"],
        PARAM_GRID_GRASP["ihc_max_iter"],
        PARAM_GRID_GRASP["ihc_stop_no_improve"],
        PARAM_GRID_GRASP["use_delta"],
    ))

    total = len(all_combos) * len(TSP_FILES)
    counter = 0

    start_total = time.perf_counter()


    # ============================================
    # GŁÓWNA PĘTLA
    # ============================================

    for tsp_file in TSP_FILES:
        print(f"\nInstancja: {tsp_file}")
        D = load_tsp_matrix(tsp_file)

        for alpha, iterations, neigh_type, ihc_iter, ihc_noimp, use_delta in all_combos:

            counter += 1
            print(
                f"[{counter}/{total}] "
                f"alpha={alpha}, iter={iterations}, neigh={neigh_type}, "
                f"ihc_iter={ihc_iter}, ihc_noimp={ihc_noimp}, delta={use_delta}"
            )

            params = {
                "alpha": alpha,
                "iterations": iterations,
                "neighborhood_type": neigh_type,
                "ihc_max_iter": ihc_iter,
                "ihc_stop_no_improve": ihc_noimp,
                "use_delta": use_delta,
            }

            # multiprocessing – równolegle REPEATS razy
            with Pool(processes=cpu_count()) as pool:
                parallel_jobs = [
                    (solve_tsp, D, params) for _ in range(REPEATS)
                ]
                results_parallel = pool.map(run_single_repeat, parallel_jobs)

            # rozpakowanie wyników
            costs = [c for c, _, _ in results_parallel]
            routes = [r for _, r, _ in results_parallel]
            runtimes = [t for _, _, t in results_parallel]

            # najlepsza trasa
            min_cost = min(costs)
            best_route = routes[costs.index(min_cost)]
            route_str = "-".join(map(str, best_route))

            results.append({
                "instance": tsp_file,
                "alpha": alpha,
                "iterations": iterations,
                "neighborhood_type": neigh_type,
                "ihc_max_iter": ihc_iter,
                "ihc_stop_no_improve": ihc_noimp,
                "use_delta": use_delta,

                "mean_cost": round(np.mean(costs), 3),
                "mean_runtime": np.mean(runtimes),
                "min_cost": round(min_cost, 3),
                "min_route": route_str,
            })


    # PODSUMOWANIE

    end_total = time.perf_counter()
    elapsed = end_total - start_total

    print(f"\nŁączny czas eksperymentów: {elapsed/60:.2f} min ({elapsed:.2f} sek)\n")

    df = pd.DataFrame(results)
    save_experiment_results(df, time_seconds=int(elapsed), subfolder="GRASP")

    print("\nNajlepsze parametry dla każdej instancji:")

    instances = df["instance"].unique()

    for inst in instances:
        sub = df[df["instance"] == inst]
        best_row = sub.loc[sub["mean_cost"].idxmin()]
        print(f"\n{inst}")
        # print(best_row["min_cost"], best_row.to_dict())
        print(f"odległość {best_row['min_cost']} = {best_row.to_dict()}")
