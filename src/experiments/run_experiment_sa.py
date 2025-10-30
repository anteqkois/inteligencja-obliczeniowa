import time
import pandas as pd
import numpy as np
from src.utils.tsp_loader import load_tsp_matrix
from src.algorithms.sa_numba import solve_tsp
from src.utils.result_saver import save_experiment_results

# ==============================================================
# EKSPERYMENT DLA ALGORYTMU SYMULOWANEGO WYŻARZANIA (SA)
# ==============================================================
# Parametry badane:
#   - temperatura początkowa (T0)
#   - współczynnik chłodzenia (alpha)
#   - maksymalna liczba iteracji
#   - minimalna temperatura
#   - rodzaj sąsiedztwa
# ==============================================================

TSP_FILES = ["Dane_TSP_48.xlsx"]

PARAM_GRID = {
    "T0": [500, 1000, 2000, 5000],
    "alpha": [0.99, 0.97, 0.95, 0.90],
    "T_min": [0.1],
    "max_iter": [2000, 5000],
    "neighborhood_type": ["swap", "two_opt", "insert"],
}
REPEATS = 5
results = []

# --- Warm-up dla Numba ---
print("🧠 Warm-up Numba (kompilacja JIT)...")
D = load_tsp_matrix(TSP_FILES[0])
_ = solve_tsp(D, {"T0": 1000, "max_iter": 100, "neighborhood_type": "swap"})
print("✅ Kompilacja zakończona. Start eksperymentu...\n")

# --- Eksperyment właściwy ---
for tsp_file in TSP_FILES:
    print(f"\n📂 Instancja: {tsp_file}")
    D = load_tsp_matrix(tsp_file)

    for T0 in PARAM_GRID["T0"]:
        for alpha in PARAM_GRID["alpha"]:
            for T_min in PARAM_GRID["T_min"]:
                for max_iter in PARAM_GRID["max_iter"]:
                    for neighborhood_type in PARAM_GRID["neighborhood_type"]:
                        costs, runtimes = [], []

                        for _ in range(REPEATS):
                            params = {
                                "T0": T0,
                                "alpha": alpha,
                                "T_min": T_min,
                                "max_iter": max_iter,
                                "neighborhood_type": neighborhood_type,
                            }

                            start_t = time.perf_counter()
                            route, cost, runtime, meta = solve_tsp(D, params)
                            total_time = time.perf_counter() - start_t

                            costs.append(cost)
                            runtimes.append(total_time)

                        results.append({
                            "instance": tsp_file,
                            "T0": T0,
                            "alpha": alpha,
                            "T_min": T_min,
                            "max_iter": max_iter,
                            "neighborhood_type": neighborhood_type,
                            "mean_cost": round(np.mean(costs), 3),
                            "min_cost": round(np.min(costs), 3),
                            "mean_runtime": np.mean(runtimes),
                        })

# --- zapis wyników ---
df = pd.DataFrame(results)
save_experiment_results(df, filename="sa_experiment_results.csv", subfolder="SA")

# --- raport ---
print("\n📊 Podsumowanie wyników:")
print(df.groupby("neighborhood_type")[["mean_cost", "min_cost", "mean_runtime"]].mean().round(3))

best = df.loc[df["mean_cost"].idxmin()]
print(f"\n🏆 Najlepsze parametry: {best.to_dict()}")
print("\n✅ Eksperyment zakończony pomyślnie.")
