# %%
# SKRYPT EKSPERYMENTALNY
# Uruchamia serię eksperymentów z różnymi parametrami algorytmu,
# zapisuje statystyki dla każdej kombinacji oraz mierzy pełny
# czas działania. Dane wynikowe trafiają do pliku CSV.
#
# Główne elementy:
#   - rozgrzanie kompilatora Numba (aby uniknąć narzutu JIT)
#   - wielokrotne uruchamianie algorytmu z tymi samymi parametrami
#   - multiprocessing (każde powtórzenie w osobnym procesie)
#   - wyznaczanie średnich kosztów i czasów
#   - zapisywanie najlepszej (minimalnej) trasy
#   - zapis wyników przy użyciu save_experiment_results()

import time
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count

from src.utils.tsp_loader import load_tsp_matrix
from src.algorithms.ihc_numba import solve_tsp
from src.utils.result_saver import save_experiment_results
from src.utils.run_single_repeat import run_single_repeat

# USTAWIENIA
TSP_FILES = ["Dane_TSP_48.xlsx", "Dane_TSP_76.xlsx", "Dane_TSP_127.xlsx"]

PARAM_GRID = {
    "n_starts": [500, 1_000, 5_000, 10_000], # liczba restartów wspinaczki
    "max_iter": [1_000, 5_000, 50_000, 200_000], # liczba iteracji jednej wspinaczki
    "stop_no_improve": [500, 1_000, 3_000, 10_000],  # limit stagnacji
    "neighborhood_type": ["swap", "insert", "two_opt"],
}

REPEATS = 5   # liczba powtórzeń dla każdej kombinacji parametrów

if __name__ == "__main__":
    results = []  # tablica na wyniki


    # ROZGRZANIE NUMBA (KOMPILACJA JIT)
    print("Rozgrzewanie Numba (kompilacja JIT)...")
    D = load_tsp_matrix(TSP_FILES[0])
    _ = solve_tsp(D, {"n_starts": 2})
    print("Kompilacja zakończona.\n")


    # GŁÓWNA PĘTLA
    # Obliczamy łączną liczbę kombinacji
    all_combos = list(itertools.product(
        PARAM_GRID["n_starts"],
        PARAM_GRID["max_iter"],
        PARAM_GRID["stop_no_improve"],
        PARAM_GRID["neighborhood_type"]
    ))
    total = len(all_combos) * len(TSP_FILES)
    counter = 0

    start_total = time.perf_counter()

    for tsp_file in TSP_FILES:
        print(f"\nInstancja: {tsp_file}")
        D = load_tsp_matrix(tsp_file)

        for n_starts in PARAM_GRID["n_starts"]:
            for max_iter in PARAM_GRID["max_iter"]:
                for stop_no_improve in PARAM_GRID["stop_no_improve"]:
                    for neighborhood_type in PARAM_GRID["neighborhood_type"]:
                        counter += 1

                        print(
                            f"[{counter}/{total}] "
                            f"n_starts={n_starts}, "
                            f"max_iter={max_iter}, "
                            f"stop_no_improve={stop_no_improve}, "
                            f"neigh={neighborhood_type}"
                        )

                        params = {
                            "n_starts": n_starts,
                            "max_iter": max_iter,
                            "stop_no_improve": stop_no_improve,
                            "neighborhood_type": neighborhood_type,
                            "use_delta": False,
                        }

                        # multiprocessing — równoległe powtórzenia
                        with Pool(processes=cpu_count()) as pool:
                            parallel_jobs = [
                                (solve_tsp, D, params) for _ in range(REPEATS)
                            ]

                            results_parallel = pool.map(run_single_repeat, parallel_jobs)

                        costs = [c for c, _, _ in results_parallel]
                        routes = [r for _, r, _ in results_parallel]
                        runtimes = [t for _, _, t in results_parallel]

                        # wybór najlepszej trasy
                        min_cost = min(costs)
                        best_route_overall = routes[costs.index(min_cost)]
                        route_str = "-".join(map(str, best_route_overall))

                        # zapis danych
                        results.append({
                            "instance": tsp_file,
                            "n_starts": n_starts,
                            "max_iter": max_iter,
                            "stop_no_improve": stop_no_improve,
                            "neighborhood_type": neighborhood_type,
                            "mean_cost": round(np.mean(costs), 3),
                            "mean_runtime": np.mean(runtimes),
                            "min_cost": round(min_cost, 3),
                            "min_route": route_str,
                        })


    # PODSUMOWANIE I ZAPIS WYNIKÓW
    end_total = time.perf_counter()
    elapsed = end_total - start_total

    print(f"\nŁączny czas eksperymentów: {elapsed/60:.2f} min ({elapsed:.2f} sek)")

    df = pd.DataFrame(results)
    save_experiment_results(df, filename="no_delta__results.csv", time_seconds=int(elapsed), subfolder="IHC")


    # RAPORT
    print("\nNajlepsze parametry dla każdej instancji:")

    instances = df["instance"].unique()

    for inst in instances:
        sub = df[df["instance"] == inst]
        best_row = sub.loc[sub["mean_cost"].idxmin()]
        print(f"\n{inst}")
        # print(best_row["min_cost"], best_row.to_dict())
        print(f"odległość {best_row["min_cost"]} = {best_row.to_dict()}")