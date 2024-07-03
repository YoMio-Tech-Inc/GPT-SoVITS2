# GPT-SoVITS2

Tento názov bol schválený autorom GPT-SoVITS, [花儿不哭](https://space.bilibili.com/5760446?spm_id_from=333.337.0.0).

### Tento projekt je stále vo vývoji a je zlepšením založeným na [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). Hlavné vylepšenia sú nasledovné:

1. **Natívna podpora viacerých jazykov**: Nie je obmedzená len na čínštinu, japončinu a angličtinu, ale podporuje akýkoľvek jazyk na svete.
2. **Nie je potrebné špecifikovať jazyk**: Je vždy viacjazyčný a pri hovorení môžete voľne miešať jazyky.
3. **Viacjazyčná extrakcia emócií z textu**: Presnejšia analýza emócií v jazykoch, čo robí reč expresívnejšou.
4. **Vylepšenie Zero Shot**: Teraz sa namiesto odporúčania dolaďovania modelu vykonáva zero shot priamo pomocou iba niekoľkých sekúnd cieľového zvuku.
5. **Fúzia referenčného zvuku**: Je možné nahrať viacero referenčných zvukových klipov a výsledný hlas bude fúziou viacerých zvukových klipov.
6. **Rýchlejšia inferencia**: Zmena pozičného vkladania na RoPE, eliminuje potrebu prepočítavať vkladanie celej sekvencie pre každú inferenciu tokenu.

### **Výzva na poskytnutie údajov a spoluprácu**: V súčasnosti sa zbierajú údaje. QQ 1715069210, ak dátová sada spĺňa požiadavky, bude v projekte uvedený kredit.

#### Momentálne organizujem nápady na úpravy v zdrojovom kóde. Hľadajte # ! pre komentáre. Ak máte záujem, neváhajte ma kontaktovať prostredníctvom vyššie uvedeného QQ.

### Zoznam zmien

#### Zmeny v kódovej knihe
Jedna kódová kniha -> 2 kódové knihy/4 kódové knihy

#### Zmeny v GPT
Nahradené qwen2-0.3b

#### Zmeny v kódovaní zvuku
cnhubert -> ~~w2v-bert-2.0 (predbežne, toto je najextravagantnejšie 4,6m-hodinové viacjazyčné predtrénovanie vykonané spoločnosťou meta. Ak bude výsledok znieť ako cudzinec hovoriaci po čínsky, bude nahradený cnhubert-large)~~/cnhubert-large/mHubert-147
Zistil som, že trénovanie w2v-bert-2.0 je trochu náročné, trénovanie mHubert-147 je relatívne jednoduchšie, rozdiel vo veľkosti je štvornásobný a v reálnych testoch fp16 priamo padá, takže sa dá použiť len fp32. Navyše, mHubert je už dosť veľký (600MB).

#### Zmeny v kódovaní textu
Odstránenie foném a zodpovedajúcich vkladaní
cn-roberta -> BGE-m3

#### Zmeny v pozičnom kódovaní
Kódovanie textu a reči každé robí sinusoidálne -> globálne sa robí RoPE vkladanie.

#### Zmeny v kombinovanom vkladaní xy (experimentálne)
Z pôvodného
x1+x2+y1 -> y2
zmenené na
x1+y1+x2 -> y2
a celá sekvencia zdieľa RoPE vkladanie.
Teoreticky je to vhodnejšie pre rozšírenie viacerých cieľových zvukov pre fúziu hlasových liniek.
Napríklad:
x1+y1+x2+y2+x3+y3+x4+y4+x5 -> y5
sa zdá prirodzenejšie ako
x1+x2+x3+x4+x5+y1+y2+y3+y4 -> y5
Nemôže byť striktne dokázané.

#### Zmeny v rozmeroch
MLP(768, 512) -> ~~Žiadne MLP priamo 1024 rozmerov. Keďže aj w2v-bert-2.0 aj bge-m3 majú 1024 rozmerov, je to perfektná zhoda.~~ MLP(1024, 768)

#### Zmeny v metóde trénovania
Čisto autoregresívne -> autoregresívne + zero shot tréningová vzorka regresie pod rovnakým hovorcom

#### zmeny vo vits
Možno nájsť spôsob, ako zvýšiť rozmery. (256 -> 512) VITS -> VITS2 (hlavne pridanie transformačného bloku do modelu toku).

#### Formát
Jednotná ~~polovičná presnosť~~ jednoduchá presnosť (reálne testy ukázali, že polovičná presnosť padá), hubert 16000 vzorkovanie, vits 32000 vzorkovanie, normalizácia hlasitosti pre všetky zvuky.

#### Zhrnutie
Celkovo sú zmeny hlavne:
1. Použitie pokročilejších predtrénovaných modelov.
2. Keďže pokročilejšie modely sú väčšie, pôvodné rozmery sú tiež rozšírené.
3. Vzhľadom na zameranie na zero shot, metóda trénovania zahŕňa zero shot tréning.
4. Pôvodný kód mal len bert pre čínštinu, zmenou na BGE m3, viacjazyčné vkladanie, prirodzene extrahuje všetky jazyky bezproblémovo.
5. Pôvodne bola len jedna kódová kniha s veľkosťou len 1024, čo spôsobovalo nedostatočné vedenie extrakcie funkcií pôvodného hubert. Zmena na dvojité kódové knihy zvyšuje množstvo informácií na 1024^2 = 1048576 a štyri kódové knihy sú ešte extravagantnejšie, ale mali by sa skúšať postupne kvôli obmedzeným údajom.
6. Jedným z dôvodov pôvodnej pomalej rýchlosti je, že GPT musí prepočítavať vkladanie celej sekvencie a pozičné vkladanie zakaždým. Prechod na RoPE eliminuje túto nevýhodu.
7. Pôvodne sa neuvažovalo o fúzii hlasových liniek, ale neskôr som upravil vetvu v pôvodnom GPT-SoVITS na dosiahnutie fúzie hlasových liniek, hoci to nebol pôvodný cieľ návrhu. V 花ovom videu bolo spomenuté, či by referenčný zvuk použitý v časti GPT a referenčný zvuk vo vits mohli byť odlišné pre fúziu hlasových liniek. Moja implementácia však má viacero zvukov v oboch častiach.
8. Zmenené časti, ktoré som považoval za nerozumné. Napríklad, keďže sa hubert používa ako zvukové vkladanie a kódové knihy ako tokeny, prečo pridávať ar_audio_embedding. Tiež, keďže tam boli fonémy, muselo im byť priradené vkladanie, čo viedlo k mnohým samostatným tréningom vkladaní, hoci sa už používali bert a hubert. Navyše, samostatné textové a zvukové vkladania by neuvažovali kontext zvuku a textu, lepšie je priamo vstúpiť do GPT a použiť pozornosť na nájdenie vzťahov.

Ak ste dočítali až sem, znamená to, že rozumiete, a ste vítaní, aby ste sa pripojili k tomuto projektu!

**QQ: 1715069210**

**WeChat: JunityZ**

#### Rýchla poznámka
Dnes som čítal mnoho článkov, vrátane článku VALLE2, a mal som mnoho nových nápadov. Jedným dôležitým problémom je, že súčasné ar_audio_embedding a ar_text_embedding sú historické dedičné problémy.

Pretože audioLM najprv použil hubert+kmeans na získanie tokenov, ale keďže učenie kvantizácie kmeans nepotrebuje učiť celkové údaje, ale priamo sa učí z distribúcie hubert, je pridané následné vkladanie.

Ale ak sa použije vq, vq sa samo o sebe už učí, takže nie je potrebné pridávať ďalšie vkladanie. Tu historické problémy vždy pridávali vkladanie, aj keď vplyv nemusí byť významný, odstránenie by bolo oveľa rozumnejšie.

Navyše, audio lm používalo sémantické aj akustické prvky prostredníctvom hubert a soundstream. Ale GPT SoVITS to má tiež, s meltransferencoderom získavajúcim akustické a hubertom získavajúcim sémantické prvky, čo je veľmi náhodné.

Séria VALLE všeobecne používa EnCodec, ktorý priamo získava tokeny zo zvuku a potrebuje ďalšie vkladanie, pretože pôvodne nevydáva vkladanie. Ale použitie hubert to nepotrebuje, pretože výstup hubert je už vkladanie.

Naopak, my získavame tokeny s hubert vkladaním, zatiaľ čo EnCodec získava najprv tokeny a potom vykonáva vkladanie.

Preto sa zdá, že pôvodný GPTSoVITS a predchádzajúca odkazovaná AUdio LM sú založené na metódach TTS série EnCodec, ale v skutočnosti sú odlišné.

#### TODO
Prepísať kvantizáciu, priamo volať Group Residual VQ z vector-quantize-pytorch.
