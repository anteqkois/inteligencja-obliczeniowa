import time
import pandas as pd
from src.utils.tsp_loader import load_tsp_matrix
from src.algorithms.nn import solve_tsp
from src.utils.result_saver import save_experiment_results

# ==============================================================
# EKSPERYMENT DLA ALGORYTMU NAJBLIŻSZEGO SĄSIADA (NN)
# ==============================================================
# Celem: analiza wpływu miasta startowego na długość trasy.
# ==============================================================

TSP_FILES = ["Dane_TSP_48.xlsx"]
results = []

# --- rozgrzewka Numba: jedno wywołanie żeby się skompilowało ---
print("🧠 Warm-up Numba (kompilacja funkcji JIT)...")
solve_tsp(load_tsp_matrix(TSP_FILES[0]), {"start_city": 0})
print("✅ Kompilacja zakończona. Start eksperymentu...\n")

for tsp_file in TSP_FILES:
    print(f"\n📂 Przetwarzanie instancji: {tsp_file}")
    D = load_tsp_matrix(tsp_file)

    for start_city in range(D.shape[0]):
        params = {"start_city": start_city}

        start_t = time.perf_counter()
        route, cost, runtime, meta = solve_tsp(D, params)
        total_time = time.perf_counter() - start_t

        results.append({
            "instance": tsp_file,
            "start_city": start_city,
            "cost": cost,
            "runtime": total_time,
        })

# --- zapis wyników ---
df = pd.DataFrame(results)
save_experiment_results(df, filename="nn_experiment_results.csv", subfolder="NN")

# --- raport ---
print("\n📈 Statystyki zbiorcze:")
print(df.describe().round(3))

best = df.loc[df["cost"].idxmin()]
worst = df.loc[df["cost"].idxmax()]

print(f"\n🏆 Najlepsze miasto startowe: {best.start_city} (koszt {best.cost:.2f})")
print(f"💀 Najgorsze miasto startowe: {worst.start_city} (koszt {worst.cost:.2f})")
print("✅ Eksperyment zakończony pomyślnie.")
