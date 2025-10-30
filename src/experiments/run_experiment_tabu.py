import time
import numpy as np
import pandas as pd
from src.utils.tsp_loader import load_tsp_matrix
from src.algorithms.tabu import solve_tsp
from src.utils.result_saver import save_experiment_results


# ==============================================================
# EKSPERYMENT DLA ALGORYTMU TABU SEARCH (TS)
# ==============================================================
# Testowane parametry:
#   - max_iter
#   - stop_no_improve
#   - tabu_tenure
#   - neighborhood_type
# ==============================================================

TSP_FILES = ["Dane_TSP_48.xlsx"]

PARAM_GRID = {
    "max_iter": [1000, 2000],
    "stop_no_improve": [100, 200],
    "tabu_tenure": [5, 10, 20],
    "neighborhood_type": ["swap", "two_opt"],
}

REPEATS = 5
results = []

print("🧠 Warm-up Numba (kompilacja JIT)...")
D = load_tsp_matrix(TSP_FILES[0])
_ = solve_tsp(D, {"max_iter": 10, "tabu_tenure": 5})
print("✅ Kompilacja zakończona. Start eksperymentu...\n")

for tsp_file in TSP_FILES:
    print(f"\n📂 Instancja: {tsp_file}")
    D = load_tsp_matrix(tsp_file)

    for max_iter in PARAM_GRID["max_iter"]:
        for stop_no_improve in PARAM_GRID["stop_no_improve"]:
            for tabu_tenure in PARAM_GRID["tabu_tenure"]:
                for neighborhood_type in PARAM_GRID["neighborhood_type"]:
                    costs, runtimes = [], []

                    for _ in range(REPEATS):
                        params = {
                            "max_iter": max_iter,
                            "stop_no_improve": stop_no_improve,
                            "tabu_tenure": tabu_tenure,
                            "neighborhood_type": neighborhood_type,
                        }
                        start_t = time.perf_counter()
                        route, cost, runtime, meta = solve_tsp(D, params)
                        total_time = time.perf_counter() - start_t

                        costs.append(cost)
                        runtimes.append(total_time)

                    results.append({
                        "instance": tsp_file,
                        "max_iter": max_iter,
                        "stop_no_improve": stop_no_improve,
                        "tabu_tenure": tabu_tenure,
                        "neighborhood_type": neighborhood_type,
                        "mean_cost": round(np.mean(costs), 3),
                        "min_cost": round(np.min(costs), 3),
                        "mean_runtime": np.mean(runtimes),
                    })

# --- zapis wyników ---
df = pd.DataFrame(results)
save_experiment_results(df, filename="tabu_experiment_results.csv", subfolder="TS")

# --- raport ---
print("\n📊 Podsumowanie wyników:")
print(df.groupby("neighborhood_type")[["mean_cost", "min_cost", "mean_runtime"]].mean().round(3))

best = df.loc[df["mean_cost"].idxmin()]
print(f"\n🏆 Najlepsze parametry: {best.to_dict()}")
print("\n✅ Eksperyment zakończony pomyślnie.")
