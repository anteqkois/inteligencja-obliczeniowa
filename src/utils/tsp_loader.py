import os
import pandas as pd
import numpy as np

def load_tsp_matrix(filename):
    """
    Wczytuje macierz odległości TSP z pliku .xlsx.
    Obsługuje ścieżki względne względem głównego katalogu projektu.
    """
    # Ścieżka do głównego katalogu projektu (tam, gdzie jest folder src)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(project_root, "src/data")
    filepath = os.path.join(data_dir, filename)

    # Sprawdzenie czy plik istnieje
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")

    # Wczytanie arkusza i konwersja do macierzy numpy
    df = pd.read_excel(filepath, header=None)
    matrix = df.to_numpy(dtype=float)

    # Usunięcie ewentualnych nanów / błędnych wartości
    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix, nan=0.0)

    # Ostrzeżenie jeśli macierz nie jest idealnie symetryczna
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        print("⚠️ Uwaga: macierz nie jest idealnie symetryczna — może zawierać błędy danych.")

    return matrix
