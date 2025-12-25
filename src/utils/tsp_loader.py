import os
import re
import pandas as pd
import numpy as np


def _expected_size_from_filename(filename: str):
    """Extract expected matrix size from digits in the filename, e.g. 48 in 'Dane_TSP_48.xlsx'."""
    basename = os.path.basename(filename)
    match = re.search(r"(\d+)", basename)
    return int(match.group(1)) if match else None


def _validate_matrix_shape(matrix: np.ndarray, filename: str):
    expected_size = _expected_size_from_filename(filename)
    if expected_size is None:
        return

    rows, cols = matrix.shape
    if (rows, cols) != (expected_size, expected_size):
        raise ValueError(
            f"Niepoprawny kształt macierzy dla pliku {filename}: "
            f"oczekiwano ({expected_size}, {expected_size}), otrzymano ({rows}, {cols}). "
            "Sprawdź czy plik zawiera nagłówki lub dodatkowe wiersze/kolumny."
        )


def load_tsp_matrix_broken(filename):
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

    # Walidacja kształtu na podstawie liczby w nazwie pliku (np. 48 -> (48, 48))
    _validate_matrix_shape(matrix, filename)

    # Ostrzeżenie jeśli macierz nie jest idealnie symetryczna
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        print("⚠️ Uwaga: macierz nie jest idealnie symetryczna — może zawierać błędy danych.")

    return matrix


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

    # Wczytanie arkusza
    # Zakładamy, że plik ma nagłówki (miasta/indeksy) w pierwszym wierszu i pierwszej kolumnie
    try:
        df = pd.read_excel(filepath, index_col=0)
    except Exception:
        # Fallback jeśli coś pójdzie nie tak (np. brak nagłówków) - czytamy raw
        df = pd.read_excel(filepath, header=None)

    # Konwersja do macierzy numpy
    matrix = df.to_numpy(dtype=float)

    # Usunięcie ewentualnych nanów / błędnych wartości
    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix, nan=0.0)

    # Walidacja kształtu na podstawie liczby w nazwie pliku (np. 48 -> (48, 48))
    _validate_matrix_shape(matrix, filename)

    # Ostrzeżenie jeśli macierz nie jest idealnie symetryczna
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        print(
            "⚠️ Uwaga: macierz nie jest idealnie symetryczna — może zawierać błędy danych."
        )

    return matrix
