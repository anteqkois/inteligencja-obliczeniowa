import numpy as np
from numba import njit

#                    OPERATORY SĄSIEDZTWA DLA TSP
# Ten moduł zawiera trzy podstawowe operatory dla problemu komiwojażera:
#   1. SWAP     – zamiana dwóch miast miejscami
#   2. TWO-OPT  – odwrócenie fragmentu trasy (klasyczny operator)
#   3. INSERT   – przesunięcie miasta na inną pozycję
#
# WAŻNE: Wszystkie funkcje zakładają ZAMKNIĘTĄ TRASĘ (cykl Hamiltona),
# czyli ostatnie miasto łączy się z pierwszym: route[n-1] -> route[0]
#
# Każda funkcja delta_* oblicza ZMIANĘ kosztu (nie pełny koszt!),
# co jest o wiele szybsze niż przeliczanie całej trasy od nowa.


@njit(cache=True)
def delta_swap(distance_matrix, route, i, j):
    """
    Oblicza zmianę kosztu po zamianie dwóch miast miejscami.
    
    Operacja: route[i] <--> route[j]
    
    Przykład: [A, B, C, D, E], swap(1, 3)
             [A, D, C, B, E]  (B i D zamieniły się miejscami)
    
    Zwraca:
        float: delta (zmiana kosztu), dodatnia = pogorszenie, ujemna = poprawa
    """
    if i == j:
        return 0.0

    n = len(route)
    
    # Upewniamy się, że i < j (ułatwia logikę)
    if i > j:
        i, j = j, i

    # Miasta, które zamieniamy
    a = route[i]
    b = route[j]

    # Sąsiedzi miasta 'a' (przed i po nim w trasie)
    a_prev = route[i - 1] if i > 0 else route[n - 1]
    a_next = route[i + 1] if i < n - 1 else route[0]

    # Sąsiedzi miasta 'b'
    b_prev = route[j - 1] if j > 0 else route[n - 1]
    b_next = route[j + 1] if j < n - 1 else route[0]

    delta = 0.0

    # PRZYPADEK 1: Miasta są bezpośrednimi sąsiadami (np. i=5, j=6)
    # Przed: ... -> a_prev -> a -> b -> b_next -> ...
    # Po:    ... -> a_prev -> b -> a -> b_next -> ...
    #
    # Zmieniają się tylko 3 krawędzie:
    #   - usuń: (a_prev, a), (a, b), (b, b_next)
    #   + dodaj: (a_prev, b), (b, a), (a, b_next)
    
    if j == i + 1:
        delta -= distance_matrix[a_prev, a]
        delta -= distance_matrix[a, b]
        delta -= distance_matrix[b, b_next]

        delta += distance_matrix[a_prev, b]
        delta += distance_matrix[b, a]
        delta += distance_matrix[a, b_next]
        return delta

    # PRZYPADEK 2: Wrap-around (i=0, j=n-1)
    # Specjalny przypadek, gdy zamieniamy pierwsze i ostatnie miasto:
    # Przed: b -> a -> a_next -> ... -> b_prev -> b -> a (wraparound!)
    # Po:    a -> b -> a_next -> ... -> b_prev -> a -> b (wraparound!)
    #
    # Zmieniają się krawędzie:
    #   - usuń: (b_prev, b), (b, a), (a, a_next)
    #   + dodaj: (b_prev, a), (a, b), (b, a_next)
    
    if i == 0 and j == n - 1:
        delta -= distance_matrix[b_prev, b]
        delta -= distance_matrix[b, a]
        delta -= distance_matrix[a, a_next]

        delta += distance_matrix[b_prev, a]
        delta += distance_matrix[a, b]
        delta += distance_matrix[b, a_next]
        return delta

    # PRZYPADEK 3: Normalny swap (miasta nie sąsiadują)
    # Przed: ... -> a_prev -> a -> a_next -> ... -> b_prev -> b -> b_next -> ...
    # Po:    ... -> a_prev -> b -> a_next -> ... -> b_prev -> a -> b_next -> ...
    #
    # Zmieniają się 4 krawędzie wokół obu miast:
    #   - usuń: (a_prev, a), (a, a_next), (b_prev, b), (b, b_next)
    #   + dodaj: (a_prev, b), (b, a_next), (b_prev, a), (a, b_next)
    
    delta -= distance_matrix[a_prev, a]
    delta -= distance_matrix[a, a_next]
    delta -= distance_matrix[b_prev, b]
    delta -= distance_matrix[b, b_next]

    delta += distance_matrix[a_prev, b]
    delta += distance_matrix[b, a_next]
    delta += distance_matrix[b_prev, a]
    delta += distance_matrix[a, b_next]

    return delta


@njit(cache=True)
def delta_two_opt(distance_matrix, route, i, j):
    """
    Oblicza zmianę kosztu po odwróceniu fragmentu trasy (2-opt).
    
    Operacja: odwróć fragment route[i:j]
    
    Przykład: [A, B, C, D, E, F], two_opt(1, 4)
             [A, D, C, B, E, F]  (fragment B,C,D został odwrócony)
    
    To jest KLASYCZNY operator 2-opt – bardzo skuteczny dla TSP!
    
    Zwraca:
        float: delta (zmiana kosztu)
    """
    if i == j:
        return 0.0

    n = len(route)
    
    # Upewniamy się, że i < j
    if i > j:
        i, j = j, i

    # LOGIKA 2-OPT
    # Odwracamy fragment [i, i+1, ..., j-1]
    #
    # Przed: ... -> im1 -> ip1 -> ... -> jm1 -> jp1 -> ...
    #              (im1 = route[i-1], ip1 = route[i], itd.)
    #
    # Po:    ... -> im1 -> jm1 -> ... -> ip1 -> jp1 -> ...
    #              (połączenie się zmienia)
    #
    # Zmieniają się TYLKO 2 krawędzie:
    #   - usuń: (im1, ip1) i (jm1, jp1)
    #   + dodaj: (im1, jm1) i (ip1, jp1)
    
    im1 = route[i - 1] if i > 0 else route[n - 1]
    ip1 = route[i]

    jm1 = route[j - 1]
    jp1 = route[j] if j < n else route[0]

    delta = 0.0

    # Usuwamy stare krawędzie
    delta -= distance_matrix[im1, ip1]
    delta -= distance_matrix[jm1, jp1]

    # Dodajemy nowe krawędzie
    delta += distance_matrix[im1, jm1]
    delta += distance_matrix[ip1, jp1]

    return delta


@njit(cache=True)
def delta_insert(distance_matrix, route, i, j):
    """
    Oblicza zmianę kosztu po przesunięciu miasta z pozycji i na pozycję j.
    
    Operacja: wyjmij route[i], przesuń pozostałe, wstaw na pozycję j
    
    Przykład 1 (i < j): [A, B, C, D, E], insert(1, 3)
                        [A, C, D, B, E]  (B przesunęło się za D)
    
    Przykład 2 (i > j): [A, B, C, D, E], insert(3, 1)
                        [A, D, B, C, E]  (D przesunęło się przed B)
    
    UWAGA NA PRZYPADKI BRZEGOWE!
    - Gdy i=0, j=n-1: przesuwamy pierwsze miasto na koniec
    - Wtedy route[(j+1)%n] = route[0] = miasto, które właśnie usuwamy!
    
    Zwraca:
        float: delta (zmiana kosztu)
    """
    if i == j:
        return 0.0

    n = len(route)
    a = route[i]  # Miasto, które przesuwamy

    # KROK 1: Oblicz sąsiadów miasta 'a' w STAREJ pozycji
    # Używamy modulo (%), żeby obsłużyć wraparound (cykl)
    a_prev = route[(i - 1) % n]
    a_next = route[(i + 1) % n]

    delta = 0.0

    # KROK 2: USUNIĘCIE miasta 'a' z pozycji i
    # Przed: ... -> a_prev -> a -> a_next -> ...
    # Po:    ... -> a_prev -> a_next -> ...
    #
    # Zmiany:
    #   - usuń: (a_prev, a), (a, a_next)
    #   + dodaj: (a_prev, a_next)  <-- "zszywamy" lukę
    
    delta -= distance_matrix[a_prev, a]
    delta -= distance_matrix[a, a_next]
    delta += distance_matrix[a_prev, a_next]

    # KROK 3: WSTAWIENIE miasta 'a' na nową pozycję j
    
    if i < j:
        # Przesuwamy w PRAWO (i < j)
        # Po usunięciu 'a', miasta [i+1, i+2, ..., j] przesuwają się
        # o 1 pozycję w lewo. Wstawiamy 'a' na pozycję j, czyli:
        #
        # MIĘDZY:
        #   - left  = route[j]       (w STAREJ trasie)
        #   - right = route[(j+1)%n] (w STAREJ trasie)
        #
        # PROBLEM: Co jeśli (j+1)%n == i?
        # Wtedy right wskazuje na miasto, które właśnie usunęliśmy!
        # 
        # Przykład: i=0, j=n-1 (przesuwamy pierwsze na koniec)
        #   route = [0, 32, ..., 16]
        #   j = n-1 = 76
        #   (j+1)%n = 0  <-- to jest pozycja, z której usuwamy!
        #
        # W takim przypadku right to faktycznie a_next (bo po usunięciu
        # miasta 'a', jego następnik staje się "prawym sąsiadem")
        
        left = route[j]
        right_idx = (j + 1) % n
        
        if right_idx == i:
            # PRZYPADEK BRZEGOWY: wstawiamy tuż przed naszą starą pozycją
            # (może się zdarzyć przy wraparound, np. i=0, j=n-1)
            right = a_next
        else:
            right = route[right_idx]
        
        # Przed: ... -> left -> right -> ...
        # Po:    ... -> left -> a -> right -> ...
        #
        # Zmiany:
        #   - usuń: (left, right)
        #   + dodaj: (left, a), (a, right)
        
        delta -= distance_matrix[left, right]
        delta += distance_matrix[left, a]
        delta += distance_matrix[a, right]

    else:
        # Przesuwamy w LEWO (i > j)
        # Po usunięciu 'a', miasta [j, j+1, ..., i-1] przesuwają się
        # o 1 pozycję w prawo. Wstawiamy 'a' na pozycję j, czyli:
        #
        # MIĘDZY:
        #   - left  = route[(j-1)%n] (w STAREJ trasie)
        #   - right = route[j]       (w STAREJ trasie)
        #
        # PROBLEM: Co jeśli (j-1)%n == i?
        # Analogicznie jak wyżej – left wskazuje na usuwane miasto.
        
        left_idx = (j - 1) % n
        
        if left_idx == i:
            # PRZYPADEK BRZEGOWY: wstawiamy tuż za naszą starą pozycją
            left = a_prev
        else:
            left = route[left_idx]
            
        right = route[j]
        
        # Przed: ... -> left -> right -> ...
        # Po:    ... -> left -> a -> right -> ...
        
        delta -= distance_matrix[left, right]
        delta += distance_matrix[left, a]
        delta += distance_matrix[a, right]

    return delta


@njit(cache=True)
def neighbor_cost_delta_numba(distance_matrix, route, current_cost, fn_id):
    """
    Generuje losowego sąsiada i oblicza jego koszt używając delty.
    
    Parametry:
        distance_matrix: macierz odległości (n x n)
        route: aktualna trasa (permutacja miast)
        current_cost: aktualny koszt trasy
        fn_id: wybór operatora (0=swap, 1=two_opt, 2=insert)
    
    Zwraca:
        (new_route, new_cost): nowa trasa i jej koszt
    """
    n = len(route)

    # Losujemy dwa różne indeksy
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    while i == j:
        j = np.random.randint(0, n)

    # Tworzymy kopię trasy (będziemy ją modyfikować)
    new_route = route.copy()

    if fn_id == 0:  # SWAP
        # Zamieniamy miasta miejscami
        tmp = new_route[i]
        new_route[i] = new_route[j]
        new_route[j] = tmp
        delta = delta_swap(distance_matrix, route, i, j)

    elif fn_id == 1:  # TWO-OPT
        # Odwracamy fragment trasy [i:j]
        if i > j:
            i, j = j, i
        new_route[i:j] = new_route[i:j][::-1]
        delta = delta_two_opt(distance_matrix, route, i, j)

    else:  # INSERT (fn_id == 2)
        # Przesuwamy miasto z pozycji i na pozycję j
        a = new_route[i]
        if i < j:
            # Przesuwamy w prawo
            for k in range(i, j):
                new_route[k] = new_route[k + 1]
            new_route[j] = a
        else:
            # Przesuwamy w lewo
            for k in range(i, j, -1):
                new_route[k] = new_route[k - 1]
            new_route[j] = a
        delta = delta_insert(distance_matrix, route, i, j)

    # Zwracamy nową trasę i jej koszt (stary koszt + delta)
    return new_route, current_cost + delta


@njit(cache=True)
def neighbor_cost_delta_numba_debug(distance_matrix, route, current_cost, fn_id):
    """
    Wersja do debugowania – zwraca również użyte indeksy i, j.
    
    Przydatne do testowania poprawności funkcji delta_*.
    """
    n = len(route)
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    while i == j:
        j = np.random.randint(0, n)

    new_route = route.copy()

    if fn_id == 0:  # SWAP
        tmp = new_route[i]
        new_route[i] = new_route[j]
        new_route[j] = tmp
        delta = delta_swap(distance_matrix, route, i, j)

    elif fn_id == 1:  # TWO-OPT
        if i > j:
            i, j = j, i
        new_route[i:j] = new_route[i:j][::-1]
        delta = delta_two_opt(distance_matrix, route, i, j)

    else:  # INSERT
        a = new_route[i]
        if i < j:
            for k in range(i, j):
                new_route[k] = new_route[k + 1]
            new_route[j] = a
        else:
            for k in range(i, j, -1):
                new_route[k] = new_route[k - 1]
            new_route[j] = a
        delta = delta_insert(distance_matrix, route, i, j)

    # Zwracamy również i, j do celów debugowania
    return new_route, current_cost + delta, i, j