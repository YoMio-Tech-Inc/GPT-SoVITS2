# GPT-SoVITS2

Ta nazwa została zatwierdzona przez autora GPT-SoVITS, [花儿不哭](https://space.bilibili.com/5760446?spm_id_from=333.337.0.0).

### Ten projekt jest nadal w fazie rozwoju i stanowi ulepszenie bazujące na [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). Główne ulepszenia są następujące:

1. **Natywne wsparcie dla wielu języków**: Nie ogranicza się do chińskiego, japońskiego i angielskiego, ale obsługuje dowolny język na świecie.
2. **Brak konieczności określania języka**: Jest zawsze wielojęzyczny i można swobodnie mieszać języki podczas mówienia.
3. **Wielojęzyczna ekstrakcja emocji z tekstu**: Dokładniejsza analiza emocjonalna języków, sprawiająca, że mowa jest bardziej ekspresyjna.
4. **Ulepszenie Zero Shot**: Teraz, zamiast zalecać dostrajanie modelu, zero shot jest wykonywane bezpośrednio przy użyciu zaledwie kilku sekund docelowego audio.
5. **Fuzja audio referencyjnego**: Można przesłać wiele klipów audio referencyjnego, a wynikowy głos będzie fuzją wielu klipów audio.
6. **Szybsze wnioskowanie**: Zmiana osadzania pozycyjnego na RoPE, eliminująca potrzebę ponownego obliczania osadzenia całej sekwencji dla każdego wnioskowania tokenu.

### **Zbieranie danych i prośba o współpracę**: Obecnie trwa zbieranie danych. QQ 1715069210, jeśli zestaw danych spełnia wymagania, zostanie przyznany kredyt w projekcie.

#### Obecnie organizowane są pomysły na modyfikacje w kodzie źródłowym. Szukaj # ! aby znaleźć komentarze. Jeśli jesteś zainteresowany, skontaktuj się przez powyższe QQ.

### Lista zmian

#### Zmiany w kodeksie
Pojedynczy kodeks -> 2 kodeksy/4 kodeksy
#### Zmiany w GPT
Zastąpione qwen2-0.3b
#### Zmiany w kodowaniu audio
cnhubert -> ~~w2v-bert-2.0 (tymczasowo, to najbardziej przesadzone 4,6 mln godzin wielojęzycznego wstępnego treningu wykonanego przez meta. Jeśli wynik będzie brzmiał jak obcokrajowiec mówiący po chińsku, zostanie zastąpiony cnhubert-large)~~/cnhubert-large/mHubert-147
Odkryłem, że trenowanie w2v-bert-2.0 jest trochę trudne, trenowanie mHubert-147 jest stosunkowo łatwiejsze, różnica w rozmiarze jest czterokrotna, a w rzeczywistych testach fp16 bezpośrednio się zawiesza, więc można używać tylko fp32. Ponadto, mHubert jest już wystarczająco duży (600MB).
#### Zmiany w kodowaniu tekstu
Usunięcie fonemów i odpowiadających im osadzeń
cn-roberta -> BGE-m3
#### Zmiany w kodowaniu pozycyjnym
Kodowanie tekstu i mowy każde wykonuje sinusoidalne -> globalnie wykonuje osadzenie RoPE.
#### Zmiany w osadzaniu połączonym xy (eksperymentalne)
Z oryginalnego
x1+x2+y1 -> y2
zmieniono na
x1+y1+x2 -> y2
i cała sekwencja dzieli osadzenie RoPE.
Teoretycznie jest to bardziej odpowiednie do rozszerzania większej ilości docelowego audio do fuzji linii głosowych.
Na przykład:
x1+y1+x2+y2+x3+y3+x4+y4+x5 -> y5
wydaje się bardziej naturalne niż
x1+x2+x3+x4+x5+y1+y2+y3+y4 -> y5
Nie można tego ściśle udowodnić.
#### Zmiany wymiarowe
MLP(768, 512) -> ~~Brak MLP bezpośrednio 1024 wymiary. Ponieważ zarówno w2v-bert-2.0, jak i bge-m3 mają 1024 wymiary, idealne dopasowanie.~~ MLP(1024, 768)
#### Zmiany w metodzie treningu
Czysta autoregresja -> autoregresja + regresja próbki treningowej zero shot pod tym samym mówcą
#### Zmiany w vits
Możliwe znalezienie sposobu na zwiększenie wymiarów. (256 -> 512) VITS -> VITS2 (głównie dodanie bloku transformera do modelu przepływu).
#### Format
Ujednolicona ~~połowa precyzji~~ pojedyncza precyzja (rzeczywiste testy wykazały, że połowa precyzji się zawiesza), hubert 16000 próbkowanie, vits 32000 próbkowanie, normalizacja głośności dla wszystkich audio.
#### Podsumowanie
Ogólnie rzecz biorąc, zmiany to głównie:
1. Wykorzystanie bardziej zaawansowanych wstępnie wytrenowanych modeli.
2. Ponieważ bardziej zaawansowane modele są większe, oryginalne wymiary również zostały rozszerzone.
3. Ze względu na skupienie się na zero shot, metoda treningu obejmuje trening zero shot.
4. Oryginalny kod miał tylko bert dla chińskiego, zmiana na BGE m3, wielojęzyczne osadzenie, naturalnie ekstrahuje wszystkie języki bez problemów.
5. Oryginał miał tylko pojedynczy kodeks o rozmiarze zaledwie 1024, co powodowało, że oryginalne wskazówki ekstrakcji cech hubert były niewystarczające. Zmiana na podwójne kodeksy zwiększa ilość informacji do 1024^2 = 1048576, a cztery kodeksy są jeszcze bardziej przesadzone, ale należy je wypróbować krok po kroku ze względu na ograniczone dane.
6. Jednym z powodów pierwotnej wolnej prędkości było to, że GPT musiał ponownie obliczać osadzenie całej sekwencji i osadzenie pozycyjne za każdym razem. Przejście na RoPE eliminuje tę wadę.
7. Początkowo nie brano pod uwagę fuzji linii głosowych, ale później zmodyfikowałem gałąź w oryginalnym GPT-SoVITS, aby osiągnąć fuzję linii głosowych, choć nie było to początkowym celem projektu. W filmie 花 wspomniano, czy audio referencyjne używane w części GPT i audio referencyjne w vits mogłyby być różne dla fuzji linii głosowych. Jednak moja implementacja ma wiele plików audio w obu częściach.
8. Zmienione części, które uważałem za nieuzasadnione. Na przykład, skoro hubert jest używany jako osadzenie audio, a kodeksy jako tokeny, dlaczego dodawać ar_audio_embedding. Ponadto, skoro były fonemy, trzeba było przypisać im osadzenie, co skutkowało wieloma oddzielnymi treningami osadzeń, chociaż bert i hubert były już używane. Co więcej, oddzielne osadzenia tekstu i audio nie uwzględniałyby kontekstu audio i tekstu, lepiej bezpośrednio wprowadzić do GPT i użyć uwagi do znalezienia relacji.

Jeśli dotarłeś tak daleko, oznacza to, że rozumiesz, i zapraszamy do dołączenia do tego projektu!

**QQ: 1715069210**

**WeChat: JunityZ**

#### Szybka notatka
Dzisiaj przeczytałem wiele artykułów, w tym artykuł VALLE2, i miałem wiele nowych pomysłów. Jednym ważnym problemem jest to, że obecne ar_audio_embedding i ar_text_embedding to historyczne problemy pozostałości.

Ponieważ audioLM najpierw użył hubert+kmeans do uzyskania tokenów, ale ponieważ uczenie się kwantyzacji kmeans nie musi uczyć się ogólnych danych, ale bezpośrednio uczy się z rozkładu hubert, dodano późniejsze osadzenie.

Ale jeśli używa się vq, samo vq już się uczy, więc nie ma potrzeby dodawania kolejnego osadzenia. Tutaj, problemy historyczne zawsze dodawały osadzenie, chociaż wpływ może nie być znaczący, usunięcie go byłoby znacznie bardziej uzasadnione.

Co więcej, audio lm używało zarówno semantyki, jak i akustyki poprzez odpowiednio hubert i soundstream. Ale GPT SoVITS również to ma, z meltransferencoder uzyskującym akustykę i hubert uzyskującym semantykę, co jest bardzo zbieżne.

Seria VALLE generalnie używa EnCodec, który bezpośrednio uzyskuje tokeny z audio i potrzebuje innego osadzenia, ponieważ początkowo nie generuje osadzenia. Ale używanie hubert nie potrzebuje tego, ponieważ wyjście hubert jest już osadzeniem.

Odwrotnie, uzyskujemy tokeny z osadzeniem hubert, podczas gdy EnCodec uzyskuje najpierw tokeny, a następnie wykonuje osadzenie.

Dlatego oryginalny GPTSoVITS i wcześniej wspomniany AUdio LM wydają się opierać na metodach TTS serii EnCodec, ale w rzeczywistości są inne.

#### TODO
Przepisać kwantyzację, bezpośrednio wywołać Group Residual VQ z vector-quantize-pytorch.
