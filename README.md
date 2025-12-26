# 🌍 Rozwiązywanie Problemu Komiwojażera (TSP)

Projekt skupia się na implementacji i analizie działania algorytmów metaheurystycznych rozwiązujących Problem Komiwojażera (Traveling Salesperson Problem). Celem było nie tylko napisanie działającego kodu, ale również jego optymalizacja oraz przetestowanie różnych wariantów i zestawień hiperparametrów.

Projekt został zrealizowany w języku **Python**, z silnym naciskiem na wydajność obliczeniową uzyskaną dzięki bibliotece **Numba** oraz zrównolegleniu obliczeń (**Multiprocessing**).

---

## 🚀 Zaimplementowane Algorytmy

W katalogu `project/src/algorithms` znajdują się implementacje następujących metod:

### 1. **NN (Nearest Neighbor)**
Najprostsza heurystyka konstrukcyjna. Startujemy z losowego miasta i zawsze wybieramy najbliższe nieodwiedzone miasto.
*   *Zastosowanie:* Szybkie wygenerowanie rozwiązania początkowego dla bardziej zaawansowanych algorytmów.

### 2. **IHC (Iterative Hill Climbing)**
Algorytm wspinaczkowy z wielokrotnym startem (Multistart).
*   Eksploruje przestrzeń rozwiązań poprzez wykonywanie ruchów w sąsiedztwie (swap, insert, two-opt).
*   Działa iteracyjnie: jeśli ruch poprawia wynik, jest akceptowany.
*   Zastosowano mechanizm "restartów", aby uciekać z minimów lokalnych.

### 3. **SA (Simulated Annealing - Symulowane Wyżarzanie)**
Inspirowany procesem wyżarzania w metalurgii.
*   Pozwala na akceptację gorszych rozwiązań z pewnym prawdopodobieństwem (zależnym od temperatury), co umożliwia ucieczkę z minimów lokalnych.
*   Wraz z czasem "temperatura" spada, a algorytm staje się bardziej zachłanny (zbiega do optimum).

### 4. **TS (Tabu Search - Przeszukiwanie z Tabu)**
Zaawansowana metoda przeszukiwania lokalnego wykorzystująca pamięć.
*   Wykorzystuje **Listę Tabu** do blokowania niedawno wykonanych ruchów, co zapobiega cyklom i zmusza algorytm do eksploracji nowych obszarów.
*   Zaimplementowano autorski wariant **"Move Tabu"**, który blokuje konkretne atrybuty ruchu (np. parę zamienionych miast) zamiast całej trasy.

### 5. **GA (Genetic Algorithm - Algorytm Genetyczny)**
Algorytm ewolucyjny operujący na populacji rozwiązań.
*   **Selekcja:** Turniejowa, Ruletka, Rankingowa.
*   **Krzyżowanie:** OX (Order), PMX (Partially Mapped), CX (Cycle).
*   **Mutacja:** Swap, Insert, Two-Opt.
*   Ewolucja przebiega przez wiele pokoleń, promując najlepsze ("najlepiej przystosowane") trasy.

### 6. **GRASP (Greedy Randomized Adaptive Search Procedure)**
Wybrany przez nas dodatkowy, ciekawy algorytm.
*   Łączy **zachłanną, losową konstrukcję** (budowanie trasy z listy najlepszych kandydatów - RCL) z **lokalnym przeszukiwaniem** (IHC).
*   Jest kompromisem między szybkością NN a dokładnością metod ewolucyjnych.

---

## ⚡ Optymalizacje i Aspekty Techniczne

Aby algorytmy działały szybko i efektywnie dla dużych instancji TSP, wprowadziliśmy szereg usprawnień:

### 🔹 1. Delta Evaluation (Liczenie przyrostowe)
Zamiast obliczać długość całej trasy po każdej zmianie (co jest kosztowne - O(N)), obliczamy tylko **różnicę (delta)** wynikającą z zamiany konkretnych krawędzi.
*   Umożliwia to błyskawiczną ocenę sąsiadów w czasie O(1) lub O(k).
*   Zaimplementowano dla ruchów: `swap`, `insert`, `two-opt`.

### 🔹 2. Kompilacja JIT (Numba)
Kluczowe funkcje obliczeniowe (liczenie dystansu, generowanie sąsiadów, pętle algorytmów IHC/SA/GRASP) zostały ozdobione dekoratorem `@jit` z biblioteki **Numba**.
*   Kod Pythonowy jest kompilowany do kodu maszynowego, co daje szybkość porównywalną z C++.

### 🔹 3. Autorski Wariant Tabu Search ("Move Tabu")
W standardowym Tabu Search często blokuje się całe rozwiązanie (hash trasy). My zaimplementowaliśmy blokowanie **RUCHU** (np. jeśli zamieniliśmy miasto A z B, to przez X iteracji nie możemy zamienić ich z powrotem).
*   Jest to bardziej efektywne dla permutacji w TSP i wymusza lepszą dywersyfikację.

### 🔹 4. Zrównoleglenie (Multiprocessing)
Testy i strojenie hiperparametrów są czasochłonne. Wykorzystaliśmy moduł `multiprocessing` oraz bibliotekę **Optuna** (w trybie równoległym), aby uruchamiać wiele instancji algorytmów jednocześnie na wszystkich rdzeniach procesora.

---

## 📂 Struktura Projektu

*   `src/algorithms/` – Implementacje algorytmów (pliki `.py`). Większość posiada wersje zoptymalizowane Numbą.
*   `src/notebooks/` – Notebooki Jupyter (`.ipynb`) służące do uruchamiania eksperymentów, wizualizacji wyników i strojenia parametrów (Optuna).
*   `src/utils/` – Funkcje pomocnicze (ładowanie danych, funkcje dystansu, operatory sąsiedztwa).
*   `data/` – Instancje problemu TSP, czyli odległości między miastami.
*   `results/` – Wyniki eksperymentów zapisywane do plików CSV.

---

## 🛠️ Jak uruchomić?

Zainstalować potrzebne pakiety
`python -m venv .venv`        # lub: python3 -m venv venv
`source .venv/bin/activate`   # Linux/Mac
`pip install -r requirements.txt`
`pip install -e .`

Najlepiej korzystać z przygotowanych notebooków w `src/notebooks`. 
Przykładowo, aby przetestować Tabu Search z optymalizacją Optuna:
1.  Otwórz `src/notebooks/tabu_optuna.ipynb`.
2.  Upewnij się, że masz zainstalowane wymagane biblioteki (`numpy`, `numba`, `pandas`, `optuna`, `matplotlib`, `tqdm`).
3.  Uruchom komórki notebooka.
