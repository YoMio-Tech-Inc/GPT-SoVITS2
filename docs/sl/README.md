# GPT-SoVITS2

To ime je odobril avtor GPT-SoVITS, [花儿不哭](https://space.bilibili.com/5760446?spm_id_from=333.337.0.0).

### Ta projekt je še v razvoju in je izboljšava na podlagi [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). Glavne izboljšave so naslednje:

1. **Vgrajena podpora za več jezikov**: Ni omejena na kitajščino, japonščino in angleščino, ampak podpira kateri koli jezik na svetu.
2. **Ni potrebe po določitvi jezika**: Vedno je večjezičen in lahko prosto mešate jezike pri govoru.
3. **Večjezično izvlečenje čustev iz besedila**: Natančnejša čustvena analiza jezikov, kar naredi govor bolj izrazit.
4. **Izboljšava Zero Shot**: Zdaj namesto priporočila za fino nastavitev modela se zero shot izvede neposredno z uporabo le nekaj sekund ciljnega zvoka.
5. **Združevanje referenčnih zvočnih posnetkov**: Naloži se lahko več referenčnih zvočnih posnetkov, končni glas pa bo združitev več zvočnih posnetkov.
6. **Hitrejše sklepanje**: Sprememba pozicijskega vdelovanja v RoPE, kar odpravlja potrebo po ponovnem izračunu vdelovanja celotnega zaporedja za vsako sklepanje žetona.

### **Zbiranje podatkov in sodelovanje**: Trenutno se zbirajo podatki. QQ 1715069210, če nabor podatkov izpolnjuje zahteve, bo v projektu navedena zasluga.

#### Trenutno organiziram ideje za spremembe v izvorni kodi. Poiščite # ! za komentarje. Če vas zanima, se lahko obrnete na zgornji QQ.

### Seznam sprememb

#### Spremembe kodne knjige
Ena kodna knjiga -> 2 kodni knjigi/4 kodne knjige

#### Spremembe GPT
Zamenjano s qwen2-0.3b

#### Spremembe kodiranja zvoka
cnhubert -> ~~w2v-bert-2.0 (začasno, to je najbolj pretiran 4,6-milijonski večjezični predtrening, ki ga je opravil meta. Če bo rezultat zvenel kot tujec, ki govori kitajsko, bo zamenjan s cnhubert-large)~~/cnhubert-large/mHubert-147
Ugotovil sem, da je usposabljanje w2v-bert-2.0 malo težko, usposabljanje mHubert-147 je relativno lažje, razlika v velikosti je štirikratna, in v resničnih testih fp16 neposredno odpove, zato se lahko uporablja samo fp32. Poleg tega je mHubert že dovolj velik (600MB).

#### Spremembe kodiranja besedila
Odstranitev fonemov in pripadajočih vdelav
cn-roberta -> BGE-m3

#### Spremembe pozicijskega kodiranja
Kodiranje besedila in govora vsako naredi sinusoidno -> globalno naredi RoPE vdelavo.

#### Spremembe združenega vdelovanja xy (eksperimentalno)
Iz prvotnega
x1+x2+y1 -> y2
spremenjeno v
x1+y1+x2 -> y2
in celotno zaporedje deli RoPE vdelavo.
Teoretično je to bolj primerno za razširitev več ciljnih zvokov za združevanje glasovnih linij.
Na primer:
x1+y1+x2+y2+x3+y3+x4+y4+x5 -> y5
se zdi bolj naravno kot
x1+x2+x3+x4+x5+y1+y2+y3+y4 -> y5
Ne more biti strogo dokazano.

#### Dimenzijske spremembe
MLP(768, 512) -> ~~Brez MLP neposredno 1024 dimenzij. Ker sta tako w2v-bert-2.0 kot bge-m3 1024-dimenzionalna, je to popolno ujemanje.~~ MLP(1024, 768)

#### Spremembe metode usposabljanja
Čisto avtoregresivno -> avtoregresivno + regresija vzorca usposabljanja zero shot pod istim govornikom

#### Spremembe vits
Morda najdemo način za povečanje dimenzij. (256 -> 512) VITS -> VITS2 (predvsem dodajanje transformatorskega bloka v model pretoka).

#### Format
Poenotena ~~polovična natančnost~~ enojna natančnost (resnični testi so pokazali, da polovična natančnost odpove), hubert 16000 vzorčenje, vits 32000 vzorčenje, normalizacija glasnosti za ves zvok.

#### Povzetek
Na splošno so spremembe predvsem:
1. Uporaba naprednejših prednaučenih modelov.
2. Ker so naprednejši modeli večji, so tudi prvotne dimenzije razširjene.
3. Zaradi osredotočenosti na zero shot metoda usposabljanja vključuje zero shot usposabljanje.
4. Prvotna koda je imela samo bert za kitajščino, sprememba na BGE m3, večjezično vdelavo, naravno izvleče vse jezike brezšivno.
5. Prvotno je bila samo ena kodna knjiga z velikostjo le 1024, zaradi česar je bilo prvotno vodenje izvlečenja značilnosti hubert nezadostno. Sprememba na dvojne kodne knjige poveča količino informacij na 1024^2 = 1048576, štiri kodne knjige pa so še bolj pretirane, vendar bi jih morali preizkusiti postopoma zaradi omejenih podatkov.
6. Eden od razlogov za prvotno počasnost je, da mora GPT vsakič znova izračunati vdelavo celotnega zaporedja in pozicijsko vdelavo. Prehod na RoPE odpravi to pomanjkljivost.
7. Prvotno združevanje glasovnih linij ni bilo upoštevano, vendar sem kasneje modificiral vejo v prvotnem GPT-SoVITS za doseganje združevanja glasovnih linij, čeprav to ni bil prvotni cilj zasnove. V videu 花 je bilo omenjeno, ali bi referenčni zvok, uporabljen v delu GPT, in referenčni zvok v vits lahko bila različna za združevanje glasovnih linij. Vendar moja implementacija ima več zvokov v obeh delih.
8. Spremenil dele, za katere sem menil, da so nerazumni. Na primer, ker se hubert uporablja kot zvočno vdelovanje in kodne knjige kot žetoni, zakaj dodajati ar_audio_embedding. Prav tako, ker so bili fonemi, jim je bilo treba dodeliti vdelavo, kar je povzročilo usposabljanje številnih ločenih vdelav, čeprav sta se že uporabljala bert in hubert. Poleg tega ločena besedilna in zvočna vdelovanja ne bi upoštevala konteksta zvoka in besedila, bolje je neposredno vnesti v GPT in uporabiti pozornost za iskanje odnosov.

Če ste prebrali do sem, pomeni, da razumete, in vabljeni ste, da se pridružite temu projektu!

**QQ: 1715069210**

**WeChat: JunityZ**

#### Hitra opomba
Danes sem prebral veliko člankov, vključno s člankom VALLE2, in imam veliko novih idej. Eno pomembno vprašanje je, da sta trenutna ar_audio_embedding in ar_text_embedding zgodovinska zapuščinska problema.

Ker je audioLM najprej uporabil hubert+kmeans za pridobivanje žetonov, vendar ker se učenje kvantizacije kmeans ne potrebuje učiti celotnih podatkov, ampak se neposredno uči iz porazdelitve hubert, je dodano naknadno vdelovanje.

Toda če se uporablja vq, se vq sam že uči, zato ni potrebe po dodajanju drugega vdelovanja. Tukaj so zgodovinska vprašanja vedno dodajala vdelovanje, čeprav vpliv morda ni bil pomemben, bi bilo odstranjevanje veliko bolj razumno.

Poleg tega je audio lm uporabil tako semantiko kot akustiko prek hubert in soundstream. Ampak GPT SoVITS ima tudi to, pri čemer meltransferencoder pridobiva akustiko in hubert semantiko, kar je zelo naključno.

VALLE serije na splošno uporabljajo EnCodec, ki neposredno pridobiva žetone iz zvoka in potrebuje drugo vdelovanje, saj na začetku ne izhaja vdelovanja. Toda uporaba hubert tega ne potrebuje, ker je izhod hubert že vdelovanje.

Obratno, mi pridobivamo žetone z vdelovanjem hubert, medtem ko EnCodec najprej pridobiva žetone in nato izvaja vdelovanje.

Zato se zdi, da prvotni GPTSoVITS in prejšnji referenčni AUdio LM temeljita na metodah TTS serije EnCodec, vendar sta dejansko drugačna.

#### TODO
Ponovno napisati kvantizacijo, neposredno klicati Group Residual VQ iz vector-quantize-pytorch.
