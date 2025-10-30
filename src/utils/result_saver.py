import os
import pandas as pd

def save_experiment_results(df, filename="results.csv", subfolder=None, show_summary=True):
    """
    Zapisuje wyniki eksperymentu do folderu /results w katalogu głównym projektu.

    Parametry:
    ----------
    df : pd.DataFrame
        Dane wynikowe do zapisania (np. z kolumnami mean_cost, min_cost, mean_runtime).
    filename : str
        Nazwa pliku CSV (np. "nn_experiment_results.csv").
    subfolder : str | None
        Jeśli podane, wyniki trafią do podfolderu w /results (np. "NN", "SA").
    show_summary : bool
        Czy wyświetlić krótkie podsumowanie w konsoli po zapisaniu.

    Zwraca:
    -------
    path_csv : str
        Pełna ścieżka do zapisanego pliku CSV.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    results_dir = os.path.join(project_root, "results")
    if subfolder:
        results_dir = os.path.join(results_dir, subfolder)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, filename)

    df.to_csv(csv_path, index=False)

    if show_summary:
        print("\n📊 Podsumowanie (pierwsze 10 wierszy):")
        print(df.head(10).to_string(index=False))
        if "strategy" in df.columns:
            print("\n📈 Średnie wg strategii:")
            print(df.groupby("strategy")[["mean_cost", "min_cost", "mean_runtime"]].mean().round(3))
        # print("\n✅ Wyniki zapisano w:")
        # print(f"  - {csv_path}")

    return csv_path
