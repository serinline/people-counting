### People counting 
projekt wykonany w ramach przedmiotu Analiza i przetwarzanie obrazów, przez: Annę Malik, Emilię Rutkowską oraz Marka Nowaka. Zespół zrealizował temat nr 3: *Zliczanie liczby osób znajdujących się na obrazie*.  

Do wykrywania osób znajdujących się na obrazie posłużono się siecią: *ssd_mobilenet_v1_coco*, zbudowaną w oparciu o framework Tensorflow. Sieć ta charakteryzuje się wykonywaniem wykrycia lokalizacji obiektów oraz ich klasyfikacji w tym samym podejściu detekcyjnym. Ponadto zapewnia ona wysoką wydajność w klasyfikacji obrazów nawet o dużej rozdzielczości. 

Skorzystano także z api: [tensorflow_object_counting_api](https://github.com/ahmetozlu/tensorflow_object_counting_api), wprowadzającego funkcje m.in. służące do wyrysowywania bounding box'ów. 

Przygotowano trzy sekwencje wideo przedstawiające przechadzających się bądź przebywających statecznie w przestrzeni miejskiej ludzi. Materiały wideo zostały manualnie oetykietowane, pod względem oczekiwanej liczby osób na danej klatce (z założeniem dokładności do około 0.25 sekundy). 

Dla tak przygotowanych trzech sekwencji wideo przeprowadzono detekcję osób. Uzyskany wynik liczby osób wykrytych przez model porównano z oczekiwaną liczbą osób zaczerpniętą z ręcznie przygotowanych etykiet.
Dla każdej klatki błąd policzony został, jako stosunek różnicy oczekiwanej liczby osób i wykrytej liczby osób do oczekiwanej liczby osób. Ponadto wyliczony został błąd dla całej sekwencji wideo uwzględniający sumaryczną liczbę osób wykrytą przez model do sumarycznej oczekiwanej liczby osób. Uzyskano następujące wyniki: 

- Average error for video people_walking_15s.mp4 = 20.69%
- Average error for video video1_800x600.mp4 = 0.52%
- Average error for video pedestrian_survaillance.mp4 = 33.21%

Wyniki różnią się znacząco w zależności od badanej sekwencji wideo. Z zaobserwowanych niedoskonałości model popełnia błędy szczególnie w sytuacji nakładających się na siebie obiektów, czy obiektów znajdujących się na dalszych planach obrazu. Niemniej, błąd na poziomie około 20% można uznać za akceptowalny.