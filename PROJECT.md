## 🧩 Opis projektu

**Temat:** Problem komiwojażera (TSP)

**Forma realizacji:**
Projekty wykonywane są **w grupach liczących od 3 do 5 osób**.

**Obrona projektu:**
Odbędzie się na ostatnich zajęciach i będzie polegała na:

* wyjaśnieniu kodu,
* odpowiedziach na pytania dotyczące fragmentu kodu lub sprawozdania,
* omówieniu metody działania każdego z algorytmów,
* przedstawieniu uzyskanych wyników i wniosków.

Obrona odbywa się **w grupach**, ale **każda osoba** musi posiadać wiedzę o całym projekcie i umieć wyjaśnić działanie dowolnego algorytmu/heurystyki.

---

## 📘 Zakres projektu

Projekt powinien obejmować:

### 1. Implementację algorytmów:

* **NN** – algorytm najbliższego sąsiada,
* **IHC** – wspinaczka z multistartem (iteracyjna wspinaczka),
* **SA** – symulowane wyżarzanie,
* **TS** – przeszukiwanie Tabu,
* **GA** – algorytmy genetyczne,
* **+1 dowolny, ciekawy algorytm** wybrany przez grupę.

Każdy z powyższych algorytmów powinien mieć zaimplementowane **co najmniej trzy rodzaje przeszukiwania ruchów (generowania sąsiadów)**.

Algorytmy genetyczne powinny zawierać:

* **minimum trzy rodzaje metod krzyżowania,**
* **minimum trzy metody doboru rodziców.**

---

### 2. Usprawnienia algorytmów

Projekt musi zawierać **co najmniej dwa usprawnienia** dla wybranych algorytmów:

* jedno z usprawnień **musi być autorskim pomysłem grupy** (nie może pochodzić z Chata),
* pomysł powinien być **racjonalny**, nawet jeśli nie poprawi wyników.

---

### 3. Zestawienie wyników

Dla każdego algorytmu należy przedstawić:

* wpływ różnych wartości parametrów na wyniki (np. liczba iteracji, temperatura, długość listy tabu, metoda selekcji itp.),
* dla każdego parametru przetestować **co najmniej 4 różne wartości**,
* obliczenia dla każdej kombinacji parametrów oraz **dla trzech instancji TSP** (pliki dostępne na Teams),
* dla algorytmów losowych wykonać **co najmniej 5 powtórzeń** (również dla Solvera Excela),
* zestawić **wartości minimalne i średnie** uzyskane dla różnych wartości parametrów,
* uwzględnić **czas wykonywania algorytmu**.

---

### 4. Analiza i porównanie wyników

* Odnieść wyniki do **rozwiązań uzyskanych za pomocą Solver Excela**.
* W sprawozdaniu zawrzeć **analizę i wnioski**.

---

### 5. Zestawienie końcowe

* Najlepsze wyniki każdego algorytmu należy umieścić w **dodatkowym pliku Excela** (szablon w materiałach z zajęć),
* w pliku podać także **uszeregowaną trasę komiwojażera**, która dała najlepszy wynik.

---

## 🏆 Dodatkowe punkty

* **+5% do oceny końcowej** za najlepszy wynik dla danej instancji TSP (jeśli kilka grup uzyska ten sam wynik – punkty dzielone).

---

## ⏰ Terminy i punkty

* Termin oddania projektu zostanie podany na kanale **Teams**.
* Za każdy rozpoczęty dzień opóźnienia: **–25% maksymalnej punktacji**.
* Projekt wysyła **jedna osoba z grupy**.

---

## 💻 Wymagania techniczne

* Algorytmy mogą być napisane **w dowolnym języku programowania**.
* Jeśli pojawi się konieczność doprecyzowania wymagań – informacje zostaną opublikowane na Teams.

---

## 🚀 Dla chętnych

Można użyć narzędzia do **optymalizacji hiperparametrów** (np. **Optuna**) i przeprowadzić dodatkowe analizy — umożliwi to **podniesienie oceny z projektu**.

---

## 📊 Przykładowe parametry algorytmów

| Algorytm                         | Parametry do testowania                                                                                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NN**                           | miasto startowe                                                                                                                                                  |
| **IHC / SA / TS / GA (mutacja)** | rodzaj sąsiedztwa                                                                                                                                                |
| **IHC / SA / TS / GA**           | kryterium stopu (liczba iteracji, liczba iteracji bez poprawy)                                                                                                   |
| **IHC / SA / TS / GA**           | liczba iteracji dla wybranego kryterium                                                                                                                          |
| **SA**                           | temperatura początkowa, metoda redukcji temperatury, liczba sprawdzanych rozwiązań                                                                               |
| **TS**                           | długość listy tabu                                                                                                                                               |
| **GA**                           | metoda doboru rodziców, prawdopodobieństwo krzyżowania, rodzaj krzyżowania, wielkość populacji, metoda tworzenia populacji potomstwa, prawdopodobieństwo mutacji |

Dodatkowo można analizować **wpływ jakości rozwiązania początkowego** (np. długości trasy startowej) na uzyskiwane wyniki.

---

## 👥 Minimalne wymagania

* **Liczba członków grupy:** co najmniej **3 osoby**.
* **Minimalna liczba analizowanych parametrów:** równa liczbie osób w grupie.
* **Rodzaj sąsiedztwa** może być liczony jako jeden z parametrów.