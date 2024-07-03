# GPT-SoVITS2

Tento název byl schválen autorem GPT-SoVITS, [花儿不哭](https://space.bilibili.com/5760446?spm_id_from=333.337.0.0).

### Tento projekt je stále ve vývoji a je vylepšením založeným na [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). Hlavní vylepšení jsou následující:

1. **Nativní podpora více jazyků**: Není omezeno pouze na čínštinu, japonštinu a angličtinu, ale podporuje jakýkoli jazyk na světě.
2. **Není potřeba specifikovat jazyk**: Je vždy vícejazyčný a při mluvení můžete volně míchat jazyky.
3. **Vícejazyčná extrakce emocí z textu**: Přesnější emoční analýza jazyků, díky čemuž je řeč expresivnější.
4. **Vylepšení Zero Shot**: Nyní místo doporučení dolaďování modelu se provádí přímo zero shot s použitím pouze několika sekund cílového audia.
5. **Fúze referenčního audia**: Lze nahrát více referenčních audio klipů a výsledný hlas bude fúzí více audio klipů.
6. **Rychlejší inference**: Změna poziční embedding na RoPE, eliminuje potřebu přepočítávat embedding celé sekvence pro každou inferenci tokenu.

### **Žádost o data a spolupráci**: Aktuálně probíhá sběr dat. QQ 1715069210, pokud datová sada splní požadavky, bude v projektu uvedeno poděkování.

#### V současné době se organizují nápady na úpravy ve zdrojovém kódu. Hledejte # ! pro nalezení komentářů. Pokud máte zájem, neváhejte kontaktovat výše uvedené QQ.

### Seznam změn

#### Změny v codebooku
Jeden codebook -> 2 codebooky/4 codebooky

#### Změny v GPT
Nahrazeno qwen2-0.3b

#### Změny v kódování audia
cnhubert -> ~~w2v-bert-2.0 (předběžně, toto je to nejextrémnější 4,6m-hodinové vícejazyčné předtrénování provedené metou. Pokud výsledek zní jako cizinec mluvící čínsky, bude nahrazen cnhubert-large)~~/cnhubert-large/mHubert-147
Zjistil jsem, že trénování w2v-bert-2.0 je trochu obtížné, trénování mHubert-147 je relativně snazší, velikostní rozdíl je čtyřnásobný, a v reálných testech fp16 přímo padá, takže lze použít pouze fp32. Navíc mHubert je již dostatečně velký (600MB).

#### Změny v kódování textu
Odstranění fonémů a odpovídajících embeddings
cn-roberta -> BGE-m3

#### Změny v pozičním kódování
Textové a řečové kódování každé dělá sinusoidální -> globálně se dělá RoPE embedding.

#### Změny v kombinovaném embedingu xy (experimentální)
Z původního
x1+x2+y1 -> y2
změněno na
x1+y1+x2 -> y2
a celá sekvence sdílí RoPE embedding.
Teoreticky je to vhodnější pro rozšíření více cílového audia pro fúzi hlasových linií.
Například:
x1+y1+x2+y2+x3+y3+x4+y4+x5 -> y5
se zdá přirozenější než
x1+x2+x3+x4+x5+y1+y2+y3+y4 -> y5
Nelze striktně dokázat.

#### Změny v dimenzích
MLP(768, 512) -> ~~Žádné MLP přímo 1024 dimenzí. Vzhledem k tomu, že jak w2v-bert-2.0, tak bge-m3 mají 1024 dimenzí, je to perfektní shoda.~~ MLP(1024, 768)

#### Změny v metodě tréninku
Čistě autoregresivní -> autoregresivní + zero shot tréninkový vzorek regrese pod stejným mluvčím

#### Změny ve vits
Možná najít způsob, jak zvýšit dimenze. (256 -> 512) VITS -> VITS2 (hlavně přidání transformer bloku do flow modelu).

#### Formát
Sjednocená ~~poloviční přesnost~~ jednoduchá přesnost (reálné testy zjistily, že poloviční přesnost padá), hubert 16000 vzorkování, vits 32000 vzorkování, normalizace hlasitosti pro všechna audia.

#### Shrnutí
Celkově jsou změny hlavně:
1. Použití pokročilejších předtrénovaných modelů.
2. Vzhledem k tomu, že pokročilejší modely jsou větší, původní dimenze jsou také rozšířeny.
3. Vzhledem k zaměření na zero shot, metoda tréninku zahrnuje zero shot trénink.
4. Původní kód měl pouze bert pro čínštinu, změna na BGE m3, vícejazyčný embedding, přirozeně extrahuje všechny jazyky bezproblémově.
5. Původně byl pouze jeden codebook o velikosti pouze 1024, což způsobovalo nedostatečné vedení původní extrakce funkcí hubert. Změna na dvojité codebooky zvyšuje množství informací na 1024^2 = 1048576, a čtyři codebooky jsou ještě extrémnější, ale mělo by se zkoušet postupně kvůli omezeným datům.
6. Jedním z důvodů původní pomalé rychlosti je, že GPT musí pokaždé přepočítat embedding a poziční embedding celé sekvence. Přechod na RoPE eliminuje tuto nevýhodu.
7. Původně nebyla uvažována fúze hlasových linií, ale později jsem upravil větev v původním GPT-SoVITS pro dosažení fúze hlasových linií, i když to nebyl původní cíl návrhu. V 花ově videu bylo zmíněno, zda by referenční audio použité v části GPT a referenční audio ve vits mohlo být odlišné pro fúzi hlasových linií. Moje implementace však má více audií v obou částech.
8. Změněny části, které jsem považoval za nerozumné. Například, když se používá hubert jako audio embedding a codebooky jako tokeny, proč přidávat ar_audio_embedding. Také, když byly fonémy, musel jim být přiřazen embedding, což vedlo k mnoha samostatným tréninkům embeddingů, i když bert a hubert již byly použity. Navíc samostatné textové a audio embeddingy by neuvažovaly kontext audia a textu, lepší je přímo vstupovat do GPT a použít pozornost k nalezení vztahů.

Pokud jste dočetli až sem, znamená to, že chápete, a jste vítáni, abyste se připojili k tomuto projektu!

**QQ: 1715069210**

**WeChat: JunityZ**

#### Rychlá poznámka
Dnes jsem četl mnoho článků, včetně článku VALLE2, a měl jsem mnoho nových nápadů. Jedním důležitým problémem je, že současné ar_audio_embedding a ar_text_embedding jsou historické dědičné problémy.

Protože audioLM nejprve použilo hubert+kmeans k získání tokenů, ale protože učení kvantizace kmeans nepotřebuje učit celková data, ale přímo se učí z distribuce hubert, je přidán následný embedding.

Ale pokud se používá vq, vq samo o sobě se již učí, takže není potřeba přidávat další embedding. Zde historické problémy vždy přidávaly embedding, i když dopad nemusí být významný, odstranění by bylo mnohem rozumnější.

Navíc audio lm používalo jak sémantiku, tak akustiku prostřednictvím hubert a soundstream. Ale GPT SoVITS to také má, s meltransferencoderem získávajícím akustiku a hubertem získávajícím sémantiku, což je velmi náhodné.

Série VALLE obecně používají EnCodec, který přímo získává tokeny z audia a potřebuje další embedding, protože zpočátku nevydává embedding. Ale použití hubert to nepotřebuje, protože výstup hubert je již embedding.

Naopak, my získáváme tokeny s hubert embeddingem, zatímco EnCodec získává nejprve tokeny a pak provádí embedding.

Proto se zdá, že původní GPTSoVITS a předchozí odkazované AUdio LM jsou založeny na metodách TTS série EnCodec, ale ve skutečnosti jsou odlišné.

#### TODO
Přepsat kvantizaci, přímo volat Group Residual VQ z vector-quantize-pytorch.
