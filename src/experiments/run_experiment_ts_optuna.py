# %%
import optuna
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# Import algorytmu Tabu Search i narzędzi pomocniczych
from src.algorithms.tabu_full_path import solve_tsp
from src.utils.tsp_loader import load_tsp_matrix
from src.utils.result_saver import save_experiment_results
from src.utils.run_single_repeat import run_single_repeat

# %%
# Konfiguracja eksperymentu
INSTANCES = [
    'Dane_TSP_48.xlsx',
    'Dane_TSP_76.xlsx',
    'Dane_TSP_127.xlsx'
]

# Liczba prób optymalizacji dla każdej instancji (ile razy Optuna wybierze parametry)
N_TRIALS = 100

# Liczba powtórzeń dla każdego zestawu parametrów
REPEATS = 5

# %%
def objective(trial, distance_matrix):
    """
    Funkcja celu dla Optuny.
    Optuna dobiera parametry, my uruchamiamy algorytm wielokrotnie (REPEATS) 
    i zwracamy wynik (minimalny koszt z powtórzeń).
    """
    # Definiujemy zakresy, z których Optuna może losować wartości.
    max_iter = trial.suggest_int("max_iter", 5_000, 100_000, step=5_000)
    stop_no_improve = trial.suggest_int("stop_no_improve", 500, 10_000, step=500)
    
    tabu_tenure = trial.suggest_int("tabu_tenure", 5, 100, step=5)
    n_neighbors = trial.suggest_int("n_neighbors", 10, 150, step=10)
    
    neighborhood_type = trial.suggest_categorical("neighborhood_type", ["swap", "insert", "two_opt"])
    
    params = {
        "max_iter": max_iter,
        "stop_no_improve": stop_no_improve,
        "tabu_tenure": tabu_tenure,
        "n_neighbors": n_neighbors,
        "neighborhood_type": neighborhood_type
    }
    
    # Uruchomienie Algorytmu Tabu Search wielokrotnie (równolegle)
    with Pool(processes=cpu_count()) as pool:
        parallel_jobs = [
            (solve_tsp, distance_matrix, params) for _ in range(REPEATS)
        ]
        # run_single_repeat zwraca (cost, route, runtime)
        results_parallel = pool.map(run_single_repeat, parallel_jobs)
    
    # Agregacja wyników
    costs = [res[0] for res in results_parallel]
    routes = [res[1] for res in results_parallel]
    runtimes = [res[2] for res in results_parallel]
    
    min_cost = min(costs)
    mean_cost = np.mean(costs)
    mean_runtime = np.mean(runtimes)
    
    # Znalezienie trasy odpowiadającej minimalnemu kosztowi
    best_idx = costs.index(min_cost)
    best_route = routes[best_idx]
    
    # Zapisanie dodatkowych statystyk w atrybutach triala
    route_str = "-".join(map(str, best_route))
    trial.set_user_attr("min_route", route_str)
    trial.set_user_attr("mean_cost", mean_cost)
    trial.set_user_attr("mean_runtime", mean_runtime)
    
    # Optuna minimalizuje wartość zwracaną. 
    # Zwracamy min_cost (najlepszy wynik z serii), aby znaleźć parametry dające szansę na najlepszy wynik.
    # Można by też zwracać mean_cost, jeśli zależy nam na stabilności.
    return min_cost

# %%
if __name__ == "__main__":
    experiment_results = []

    for instance_file in INSTANCES:
        print(f"\nOptymalizacja dla instancji: {instance_file}")

        distance_matrix = load_tsp_matrix(instance_file)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Parametry startowe (warm start)
        study.enqueue_trial({
            "max_iter": 5_000,
            "stop_no_improve": 500,
            "tabu_tenure": 10,
            "n_neighbors": 30,
            "neighborhood_type": "two_opt"
        })

        study.optimize(lambda trial: objective(trial, distance_matrix), n_trials=N_TRIALS)

        print(f"Najlepsze parametry dla {instance_file}: {study.best_params}")
        print(f"Najlepszy koszt dla {instance_file}: {study.best_value}")

        df_trials = study.trials_dataframe()

        for _, row in df_trials.iterrows():
            record = {
                "instance": instance_file,
                "max_iter": row.get("params_max_iter"),
                "tabu_tenure": row.get("params_tabu_tenure"),
                "n_neighbors": row.get("params_n_neighbors"),
                "neighborhood_type": row.get("params_neighborhood_type"),
                "stop_no_improve": row.get("params_stop_no_improve"),
                # Statystyki z powtórzeń
                "min_cost": row.get("value"),
                "mean_cost": row.get("user_attrs_mean_cost"),
                "mean_runtime": row.get("user_attrs_mean_runtime"),
                "min_route": row.get("user_attrs_min_route"),
                "state": row.get("state"),
                "trial_number": row.get("number")
            }
            experiment_results.append(record)

    # %%
    if experiment_results:
        df_final = pd.DataFrame(experiment_results)

        # Sortowanie wyników: najpierw po instancji, potem po numerze triala
        df_final = df_final.sort_values(by=["instance", "trial_number"])

        save_experiment_results(df_final, time_seconds=0, subfolder="TS_Optuna", sort_by_cost=False)

        print("\nWyniki zostały zapisane pomyślnie w folderze project/results/TS_Optuna/")
        print(df_final[["instance", "trial_number", "min_cost", "mean_cost", "mean_runtime"]].head())