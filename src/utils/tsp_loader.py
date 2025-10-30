import pandas as pd
import numpy as np
import os

def load_tsp_matrix(filename: str, data_dir: str = None) -> np.ndarray:
    """
    Wczytuje gotową macierz odległości między miastami (TSP).
    Obsługuje pliki Excel (.xlsx) lub CSV.
    Automatycznie usuwa kolumny/wiersze z numerami miast.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")

    ext = os.path.splitext(filename)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath, header=None)
    elif ext == ".csv":
        df = pd.read_csv(filepath, header=None)
    else:
        raise ValueError(f"Nieobsługiwany format pliku: {ext}")

    matrix = df.values.astype(float)

    # Jeśli pierwsza kolumna lub wiersz wygląda na indeksy (1,2,3,...)
    if np.allclose(matrix[0, 1:], np.arange(1, matrix.shape[1])):
        matrix = matrix[1:, 1:]  # usuń pierwszy wiersz i kolumnę

    n, m = matrix.shape
    if n != m:
        raise ValueError(f"Macierz nie jest kwadratowa ({n}x{m}).")

    if not np.allclose(matrix, matrix.T, atol=1e-8):
        print("⚠️ Uwaga: macierz nie jest idealnie symetryczna — może zawierać błędy danych.")

    return matrix
