1) Pierwsza optymalizacja to pojawiający się w kodzie paramer use_delta.
Otóż zamiast zawsze na nowo obliczać dystans między dwoma miastami, korzystamy z tego, że możemy zapisać informację z przeszłych ruchów i wykorzystać ją w dalszym ciągu do obliczenia tylko różnic w dystansach między miastami, które zmieniły swoją kolejność, a następnie tą zmianę zsumować do starejtrasy (i wtedy uzyskujemy nową, bez liczenia całości)

Wyniki do porówania są w katalogi results/GRASP
z użyciem parametru use_delta: project/results/GRASP/2025-11-17__06-27__35633_sec__results.csv
bez użycia parametru use_delta: project/results/GRASP/


2) Druga optymalizacja to dodanie wariantu co uznajemy za wykonany ruch w tabu_search.
Zamiast tylko używać podstawowego wariantu czyli blokowania całej ścieżki na x ruchów, możemy zastosować wariant, który blokuje danych ruch między miastami.

Podstawowy wariant to "full_tabu".
**Działanie**: Na listę Tabu trafia cała trasa (jako krotka lub hash).
**Sens**: Sprawdzamy, czy już cała trasa była wykonana? Działa to dobrze w małych przestrzeniach stanów. W TSP, gdzie liczba permutacji jest ogromna, szansa na powrót do dokładnie tej samej trasy losowo jest znikoma tym bardziej gdy sprawdzamy to dla 127 miast.
**Wniosek**: Jest to wariant bazowy. W praktyce przy naszym problemie rzadziej coś blokuje.

Wariant dodany przez nas to "move_tabu"
**Działanie**: Na listę Tabu trafia ruch (atrybut ruchu), np. para indeksów (i, j), które zostały zamienione.
**Sens**: Blokujemy operację odwrotną. Jeśli zamieniliśmy miasta A i B, to przez tabu_tenure kadencji nie możemy zamienić ich z powrotem.
**Wniosek**: To jest według naszego uznania lepsze rozwiązanie Tabu Search. Zmusza algorytm do eksploracji nowych obszarów.

Wyniki do porówania są w katalogi results/TS (podspiane odpowednio)