# %%
# SKRYPT EKSPERYMENTALNY
# Uruchamia analizę algorytmu Tabu Search (TS) dla różnych
# konfiguracji parametrów. Dla każdej kombinacji wykonywane jest
# kilka powtórzeń, zapisywane są statystyki jakości oraz czas
# działania. Wyniki trafiają do pliku CSV.

import time
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count

from src.utils.tsp_loader import load_tsp_matrix
from src.algorithms.tabu_move import solve_tsp
# from src.algorithms.tabu_full_path import solve_tsp
from src.utils.result_saver import save_experiment_results
from src.utils.run_single_repeat import run_single_repeat


# USTAWIENIA
TSP_FILES = ["Dane_TSP_48.xlsx", "Dane_TSP_76.xlsx", "Dane_TSP_127.xlsx"]

PARAM_GRID = {
    "max_iter": [10_000, 25_000, 75_000, 200_000],
    "stop_no_improve": [1_000, 2_000, 5_000, 7_000],
    "tabu_tenure": [10, 15, 25, 50],
    "n_neighbors": [30, 60, 100, 150],
    "neighborhood_type": ["swap", "insert", "two_opt"],
}

REPEATS = 5

if __name__ == "__main__":
    results = []

    # ROZGRZANIE (KOMPILACJA JIT JEŚLI WYSTĘPUJE)
    print("Rozgrzewanie (kompilacja JIT jeśli dotyczy)...")
    D = load_tsp_matrix(TSP_FILES[0])
    _ = solve_tsp(D, {"max_iter": 5, "tabu_tenure": 3, "n_neighbors": 5})
    print("Kompilacja zakończona.\n")

    # GŁÓWNA PĘTLA
    # Obliczamy łączną liczbę kombinacji
    all_combos = list(itertools.product(
        PARAM_GRID["max_iter"],
        PARAM_GRID["stop_no_improve"],
        PARAM_GRID["tabu_tenure"],
        PARAM_GRID["n_neighbors"],
        PARAM_GRID["neighborhood_type"],
    ))
    total = len(all_combos) * len(TSP_FILES)
    counter = 0
    start_total = time.perf_counter()

    for tsp_file in TSP_FILES:
        print(f"\nInstancja: {tsp_file}")
        D = load_tsp_matrix(tsp_file)

        for max_iter in PARAM_GRID["max_iter"]:
            for stop_no_improve in PARAM_GRID["stop_no_improve"]:
                for tabu_tenure in PARAM_GRID["tabu_tenure"]:
                    for n_neighbors in PARAM_GRID["n_neighbors"]:
                        for neighborhood_type in PARAM_GRID["neighborhood_type"]:
                            counter += 1

                            print(
                                f"[{counter}/{total}] "
                                f"max_iter={max_iter}, "
                                f"stop_no_improve={stop_no_improve}, "
                                f"tabu_tenure={tabu_tenure}, "
                                f"n_neighbors={n_neighbors}, "
                                f"neighborhood_type={neighborhood_type}"
                            )

                            params = {
                                "max_iter": max_iter,
                                "stop_no_improve": stop_no_improve,
                                "tabu_tenure": tabu_tenure,
                                "n_neighbors": n_neighbors,
                                "neighborhood_type": neighborhood_type,
                            }

                            # multiprocessing — równoległe powtórzenia
                            with Pool(processes=cpu_count()) as pool:
                                parallel_jobs = [
                                    (solve_tsp, D, params) for _ in range(REPEATS)
                                ]
                                results_parallel = pool.map(run_single_repeat, parallel_jobs)

                            # --- ZBIERANIE DANYCH ---
                            costs = [c for c, _, _ in results_parallel]
                            routes = [r for _, r, _ in results_parallel]
                            runtimes = [t for _, _, t in results_parallel]

                            # najlepsza trasa
                            min_cost = min(costs)
                            best_route_overall = routes[costs.index(min_cost)]
                            route_str = "-".join(map(str, best_route_overall))

                            results.append({
                                "instance": tsp_file,
                                "max_iter": max_iter,
                                "stop_no_improve": stop_no_improve,
                                "tabu_tenure": tabu_tenure,
                                "n_neighbors": n_neighbors,
                                "neighborhood_type": neighborhood_type,
                                "mean_cost": round(np.mean(costs), 3),
                                "mean_runtime": np.mean(runtimes),
                                "min_cost": round(min_cost, 3),
                                "min_route": route_str,
                            })

    # ZAPIS WYNIKÓW
    end_total = time.perf_counter()
    elapsed = end_total - start_total

    df = pd.DataFrame(results)
    save_experiment_results(df, filename='tabu_move_results.csv', time_seconds=int(elapsed), subfolder="TS")


    # RAPORT
    print(f"\nŁączny czas eksperymentów: {elapsed/60:.2f} min ({elapsed:.2f} sek)")

    print("\nNajlepsze parametry dla każdej instancji:")

    instances = df["instance"].unique()

    for inst in instances:
        sub = df[df["instance"] == inst]
        best_row = sub.loc[sub["mean_cost"].idxmin()]
        print(f"\n{inst}")
        # print(best_row["min_cost"], best_row.to_dict())
        print(f"odległość {best_row["min_cost"]} = {best_row.to_dict()}")