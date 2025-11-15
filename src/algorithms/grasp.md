# GRASP – Greedy Randomized Adaptive Search Procedure

GRASP to metaheurystyka dwufazowa, składająca się z:

1. Konstrukcji rozwiązania (losowo–zachłannej)
2. Lokalnej optymalizacji (np. IHC albo 2-opt)

Cały proces powtarzamy wiele razy i wybieramy najlepszy wynik.

W praktyce jest to połączenie:

NN + Randomization + IHC
(robione wielokrotnie)

---

# Jaki jest cel GRASP?

NN daje dobre, ale całkowicie deterministyczne rozwiązania – za każdym razem takie samo.

IHC daje lepsze wyniki, ale startuje z przypadkowej permutacji, więc bywa „chaotyczny”.

GRASP daje „mądrze losowe” starty – czyli nie przypadkowe, tylko kontrolowane przez zachłanność.

W efekcie dostajemy:

mądra losowość + lokalna optymalizacja
co daje bardzo dobrą jakość.

---

# Jak działa GRASP dla TSP?

## Faza 1 – Konstrukcja trasy (greedy randomized)

To działa podobnie do NN, tylko wprowadzamy kontrolowaną losowość.

Standardowy NN robi:

wybieramy najbliższe dostępne miasto

W GRASP robimy:

wybieramy losowo jedno z K najlepszych miast

Czyli krok po kroku:

1. Startujemy w losowym mieście.
2. Liczymy odległości do wszystkich nieodwiedzonych.
3. Tworzymy RCL (Restricted Candidate List), czyli listę najlepszych kandydatów.
   Na przykład:
   • 3 najbliższe miasta
   • albo wszystkie miasta w odległości ≤ min_dist + α*(max_dist-min_dist)
4. Wybieramy losowo jedno z miast w RCL.
5. Przechodzimy do niego.
6. Powtarzamy aż zbudujemy pełną trasę.

Efekt:

* nie jest to czysty greedy,
* nie jest to czysty random,
* dostajemy dobry punkt startowy.

---

## Faza 2 – Local Search (czyli używamy to co mamy)

Po zbudowaniu trasy uruchamiamy lokalne ulepszanie.
Możemy użyć wszystkiego, co już mamy:

* IHC (hill climbing)
* 2-opt
* insert
* swap
* delta-cost

Najczęściej stosowane:

2-opt aż do braku poprawy
lub
pełna IHC (swap + insert + 2-opt)

---

# Iteracje GRASP

Cały algorytm wykonujemy wiele razy:

W praktyce:

* dla TSP ~100–200 miast robimy 100–500 iteracji,
* każda iteracja to mniej więcej O(n²),
* działa bardzo szybko.

---

# Jak GRASP wygląda na tle pozostałych algorytmów?

NN – zachłanny, prosty, średnia jakość
IHC – lokalne ulepszanie, lepsza jakość
SA – probabilistyczne ulepszanie
TS – tabu, pamięć, bardzo dobra jakość
GA – populacje, dużo losowości
GRASP – losowo–zachłanna konstrukcja + local search

GRASP wyposażony jest w bardzo dobrą strukturę, wygląda profesjonalnie, jest prosty i w praktyce daje świetne wyniki.

---

# GRASP vs IHC

IHC:

* startuje z losowej permutacji - jakość startowa różna
* ulepsza do najbliższego optimum lokalnego

GRASP:

* startuje z mądrej (pół-greedy) konstrukcji - lepsze starty
* ulepsza do lokalnego optimum
* zwykle daje znacznie lepsze końcowe wyniki

W praktyce GRASP prawie zawsze wygrywa z IHC dla TSP.

---

# GRASP vs SA / TS / GA

SA: wymaga wielu parametrów, GRASP jest stabilniejszy
TS: bardzo dobry, ale cięższy w implementacji
GA: silny, ale bardzo rozbudowany i wolniejszy
GRASP: prosty, szybki, powtarzalny, świetny do raportu