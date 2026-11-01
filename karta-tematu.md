# Karta zgłoszenia tematu projektu

**Grupa:** L3 (niestacjonarna) **Data zgłoszenia:** 3.10.2026

---

## 1. Skład zespołu

|Imię i nazwisko|Nr albumu|Rola w zespole|
|---|---|---|
|Adrianna Golak|31877|współodpowiedzialna za całość: dane, implementacja, ewaluacja, dokumentacja|
|Gilbert Helwich|24546|współodpowiedzialny za całość: dane, implementacja, ewaluacja, dokumentacja|
|Dawid Ślusarczyk|31077|współodpowiedzialny za całość: dane, implementacja, ewaluacja, dokumentacja|

Każda osoba ma równy wkład w każdy etap projektu. Konkretny podział zadań w poszczególnych etapach jest w pkt 13, a koordynacja rotuje razem z dyżurem raportu tygodniowego.

## 2. Tytuł projektu

**Po polsku:** Klasyfikator wydźwięku recenzji filmowych

**Po angielsku:** Movie Review Sentiment Classifier

## 3. Problem

Redakcja serwisu z recenzjami filmowymi codziennie dostaje setki recenzji pisanych przez użytkowników. Żeby pokazać przy filmie, jak przyjęli go widzowie, i szybko wyłapać skrajnie negatywne opinie do moderacji, ktoś musi każdą recenzję przeczytać i oznaczyć jako pozytywną albo negatywną. Przy takiej liczbie tekstów to godziny pracy dziennie, a oceny różnych osób bywają niespójne. System ma na podstawie samego tekstu recenzji wskazać jej wydźwięk (pozytywny / negatywny), żeby redakcja czytała tylko przypadki wątpliwe.

## 4. Dlaczego zwykły algorytm nie wystarczy

- [x] nie umiemy zapisać reguły krok po kroku
- [ ] przepis istnieje, ale jest obliczeniowo za drogi
- [ ] reguł jest za dużo i zmieniają się w czasie

_Uzasadnienie:_ Człowiek od razu rozpoznaje, czy recenzja jest pochwałą, czy krytyką, ale nie potrafi tego zapisać jako reguły. Te same słowa zmieniają znaczenie w kontekście („nie był wcale zły”, „tak zły, że aż dobry”), recenzenci używają ironii, a lista „dobrych” i „złych” słów myli się przy zaprzeczeniach i porównaniach. Wiedza o tym, co oznacza pozytywną recenzję, siedzi w oznaczonych przykładach, a nie w regułach.

## 5. Technologia SI

- [ ] sieć neuronowa
- [ ] system ekspertowy / wnioskowanie regułowe
- [ ] algorytm genetyczny
- [x] klasyczne uczenie maszynowe
- [ ] inne:

_Dlaczego akurat ta technologia:_ Problem to klasyfikacja: mamy tysiące recenzji z gotowymi etykietami (pozytywna / negatywna), a reguły rozpoznającej wydźwięk nie umiemy zapisać. Dlatego zamiast pisać reguły pokazujemy systemowi przykłady i pozwalamy mu nauczyć się, które słowa świadczą o recenzji pozytywnej, a które o negatywnej. 

## 6. Dane

|||
|---|---|
|Źródło|Filmweb+, Paweł Narolski, 2020. https://github.com/narolski/filmwebplus. Licencja MIT.|
|Rozmiar i format|27 202 recenzje filmowe po polsku z serwisów Filmweb, Film.org.pl, Kinoblog, FDB i blogfilmowy24, średnio ok. 514 słów na recenzję. Jeden plik CSV (tekst recenzji i etykieta), spakowany ok. 42 MB, po rozpakowaniu ok. 104 MB. Etykiety według oceny autora recenzji: nieoznaczone 7 710, negatywne 3 085, neutralne 9 436, pozytywne 6 971. Używamy tylko recenzji negatywnych (3 085) i pozytywnych (6 971). Po wyrównaniu liczby recenzji w obu grupach zostaje ok. 6 170 recenzji (3 085 + 3 085).|
|Czy są już dostępne?|tak|


## 7. Stos technologiczny

Wstępne założenia:
- Python 3
- NumPy
- scikit-learn
- pandas
- matplotlib
- Środowisko: Google Colab i lokalnie

## 8. Kryterium sukcesu

**Metryka:** trafność, czyli procent recenzji z części testowej, którym system przypisał właściwą etykietę (pozytywna / negatywna). Do tego macierz pomyłek: tabela, która pokazuje, ile recenzji pozytywnych system uznał za negatywne i odwrotnie.

**Punkty odniesienia, liczone na tej samej części testowej:**
1. zawsze najczęstsza klasa: ok. 50%, bo po wyrównaniu w danych jest tyle samo recenzji pozytywnych co negatywnych,
2. prosta lista słów kluczowych: ręcznie spisane polskie słowa pozytywne („świetny”, „znakomity”…) i negatywne („nuda”, „rozczarowanie”…) razem z ich najczęstszymi odmianami („świetny”, „świetna”, „świetne”). System liczy, których słów jest w recenzji więcej, i według tego przypisuje etykietę. Przy remisie wybiera najczęstszą klasę. Listę piszemy, zanim zajrzymy do części testowej.

**Próg sukcesu:**
- trafność własnego modelu **co najmniej 80%** na części testowej,
- wynik własnego modelu różni się od wyniku gotowego modelu ze scikit-learn o **nie więcej niż 5 punktów procentowych**, co pokazuje, że nasza implementacja działa poprawnie,

Do tego przegląd co najmniej 20 recenzji, przy których system się pomylił, z próbą wyjaśnienia dlaczego (np. zaprzeczenia „wcale nie jest zły”, ironia, recenzje mieszane).

## 9. Zakres minimalny

1. Skrypt pobierający dane.
2. Wybór recenzji pozytywnych i negatywnych i wyrównanie ich liczby, tak żeby obu było tyle samo. Stały podział na część uczącą i testową, ustalony raz na początku.
3. Test sprawdzający, że ta sama recenzja nie występuje jednocześnie w części uczącej i testowej.
4. Przygotowanie tekstu (małe litery, usunięcie zbędnych znaków) i zamiana każdej recenzji na listę liczb: ile razy występuje w niej każde słowo ze słownika zbudowanego na części uczącej.
5. Własna implementacja modelu uczonego na oznaczonych przykładach (konkretną metodę wybierzemy po zajęciach 10.10) i wykres, jak zmienia się liczba błędów w trakcie uczenia.
6. Oba punkty odniesienia z pkt 8.
7. Porównanie z gotowym modelem ze scikit-learn na tych samych danych.
8. Pomiar trafności, macierz pomyłek, przegląd błędów.
9. README po angielsku z instrukcją uruchomienia.

## 10. Zakres opcjonalny

- Lista słów, które najmocniej wpływają na decyzję modelu.
- Uwzględnienie par sąsiednich słów (np. „nie polecam”, „nie jest dobry”) i sprawdzenie, czy system lepiej radzi sobie z zaprzeczeniami.
- Skrócenie każdego słowa do pierwszych kilku liter, żeby odmiany liczyły się jako jedno słowo („świetny”, „świetna”, „świetne” → „świetn”), i sprawdzenie, czy to poprawia wynik.
- Sprawdzenie systemu na krótkich recenzjach z innego źródła: zbiór polish-movie-reviews-dataset (https://github.com/kamilsan/polish-movie-reviews-dataset, recenzje z Filmwebu do 160 znaków), żeby zobaczyć, jak radzi sobie z tekstami innymi niż te, na których się uczył.

## 11. Repozytorium

_Link:_ https://github.com/adriannag9/ml-sentiment-analysis-project

## 12. Główne ryzyko i plan awaryjny

**Ryzyko 1: przeciek danych.** Jeśli ta sama recenzja trafi do części uczącej i testowej, wynik będzie fałszywie dobry. _Plan:_ podział na część uczącą i testową ustalamy w sprincie 1 i więcej go nie zmieniamy. Test w `tests/` sprawdza, że żadna recenzja nie występuje w obu częściach.

**Ryzyko 2: własny model nie działa poprawnie** i daje wynik dużo gorszy niż gotowy model z biblioteki. _Plan:_ najpierw sprawdzamy go na kilku prostych recenzjach, przy których sami wiemy, jaki powinien być wynik. Potem porównujemy go z gotowym modelem ze scikit-learn na tych samych danych.

**Ryzyko 3: niejasne etykiety.** Recenzje pochodzą z kilku serwisów z różnymi skalami ocen i nie wiemy dokładnie, jak powstały etykiety. _Plan:_ sprawdzamy to w opisie zbioru. Przy przeglądzie błędów zwracamy uwagę, czy część „pomyłek” to po prostu źle oznaczone recenzje, i opisujemy to w dokumentacji.

**Ryzyko 4: inne teksty niż w opisie problemu.** W zbiorze są długie recenzje krytyków i blogerów, a recenzje użytkowników są zwykle krótsze. _Plan:_ opisujemy to w dokumentacji jako ograniczenie. Jeśli starczy czasu, sprawdzamy system na krótkich recenzjach (pkt 10).

**Ryzyko 5: za duży zakres.** _Plan:_ zostajemy przy dwóch etykietach (pozytywna / negatywna). Zakres opcjonalny zaczynamy dopiero po spełnieniu kryterium z pkt 8.

## 13. Podział pracy w czasie

Każda osoba ma swój kawałek każdego etapu, łącznie z pisaniem kodu. Koordynacja rotuje razem z dyżurem raportu tygodniowego.

|Etap|Kto|Szacowany czas na osobę|
|---|---|---|
|Dane i przygotowanie|Gilbert: skrypt pobierający i opis zbioru · Adrianna: wyrównanie klas, podział danych i test na powtórzenia · Dawid: czyszczenie tekstu i zamiana na wektory|7 h|
|Implementacja|Dawid: własny perceptron · Gilbert: punkty odniesienia i model ze scikit-learn · Adrianna: pomiar trafności i macierz pomyłek|10 h|
|Eksperymenty i ewaluacja|Adrianna: przegląd błędów · Dawid: dobór ustawień perceptronu · Gilbert: porównanie modeli i wykresy|5 h|
|Dokumentacja PL|wszyscy, każdy opisuje swoje części|4 h|
|Dokumentacja EN|wszyscy, README i streszczenie techniczne|2 h|
|Prezentacja|wszyscy, każdy prezentuje i umie wyjaśnić całość|2 h|
|**Razem na osobę**||**30 h**|

---

## Decyzja prowadzącego

_Wypełnia prowadzący — nie edytujcie tej sekcji._

- [ ] zatwierdzony
- [ ] do poprawy

**Uwagi:**
