import os
from datetime import datetime

# FUNKCJA ZAPISUJĄCA WYNIKI EKSPERYMENTÓW DO CSV
# ---------------------------------------------
# Zapisuje wyniki algorytmów TSP do katalogu /results/, z opcjonalną
# obsługą podfolderów (np. IHC, SA, TS). Automatycznie dodaje znacznik
# czasowy do nazwy pliku, a także, jeśli podano to czas trwania całego
# eksperymentu. Może również wyświetlać krótkie podsumowanie danych.
# ---------------------------------------------


def save_experiment_results(
    df,
    filename: str = "results.csv",
    time_seconds: int | None = None,
    subfolder: str | None = None,
    show_summary: bool = True,
    summary_count: int = 20,
    sort_by_cost: bool = True,
):
    """
    Zapis wyników eksperymentu do pliku CSV.

    Parametry:
        df : pd.DataFrame
            Dane wynikowe do zapisania (np. koszty, czasy, parametry).
        filename : str
            Nazwa pliku CSV, np. "ihc_results.csv".
        time_seconds : int | None
            Łączny czas działania eksperymentu w sekundach.
            Jeśli podany, zostanie dodany na początek nazwy pliku.
        subfolder : str | None
            Jeśli ustawione, wyniki trafiają do results/<subfolder>.
            Pozwala rozdzielać wyniki algorytmów (IHC, SA, TS, NN).
        show_summary : bool
            Czy wypisać krótkie podsumowanie po zapisaniu wyników.
        summary_count : int
            Liczba wierszy pokazywanych w podsumowaniu.
        sort_by_cost : bool
            Czy sortować wyniki wg kolumny min_cost (domyślnie True).

    Zwraca:
        str : pełna ścieżka zapisanego pliku CSV.
    """

    # lokalizacja katalogu results/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    results_dir = os.path.join(project_root, "results_new")

    if subfolder:
        results_dir = os.path.join(results_dir, subfolder)

    os.makedirs(results_dir, exist_ok=True)

    # sortowanie wyników według najlepszego min_cost (jeśli istnieje)
    if sort_by_cost and "min_cost" in df.columns:
        df = df.sort_values(by="min_cost", ascending=True).reset_index(drop=True)

    # znacznik czasowy w nazwie pliku
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M")

    # dodanie czasu trwania eksperymentu do nazwy pliku
    if time_seconds is not None:
        filename = f"{time_seconds}_sec__{filename}"

    csv_path = os.path.join(results_dir, f"{timestamp}__{filename}")

    # zapis CSV
    df.to_csv(csv_path, index=False)

    # wyświetlenie podsumowania
    if show_summary:
        print(f"\nPodsumowanie (pierwsze {summary_count} wierszy):")
        print(df.head(summary_count).to_string(index=False))

        if "strategy" in df.columns:
            print("\nŚrednie wg strategii:")
            print(
                df.groupby("strategy")[["mean_cost", "min_cost", "mean_runtime"]]
                .mean()
                .round(3)
            )

    return csv_path
