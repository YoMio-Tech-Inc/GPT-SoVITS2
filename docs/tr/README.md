# GPT-SoVITS2

Bu isim, GPT-SoVITS'in yazarı [花儿不哭](https://space.bilibili.com/5760446?spm_id_from=333.337.0.0) tarafından izin alınmıştır.
### Bu proje hala geliştirme aşamasındadır ve [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) temel alınarak iyileştirilmiştir. Ana iyileştirme noktaları aşağıdaki gibidir:
|GPT-SoVITS|GPT-SoVITS2|
|:----:|:----:|
|**Metin**|**Metin**|
|Metin->Telefon|Metin->BPE|
|Telefon->Gömme|BPE->Gömme|
|Roberta-Çince|BGE-M3|
|**Konuşma Kodlayıcı**|**Konuşma Kodlayıcı**|
|Hubert|$S^3$|
|VQ|$S^3$->Gömme|
|1024 Konuşma Jetonu|4096 Konuşma Jetonu|
|**AR**|**AR**|
|Eski Tarz GPT|Qwen2-0.3b|
|**Konuşma Kod Çözücü**|**Konuşma Kod Çözücü**|
|VITS|VITS2|
|Gizli Boyut 192|Gizli Boyut 256|
|2 Kafa|4 Kafa|
|Ara Boyut 768|Ara Boyut 1024|
|**Eğitim**|**Eğitim**|
|Sıfır-Atış Eğitimi Yok|Aynı Konuşmacı ile Farklı Konuşma Çıkarımı|
|ZH,EN,JA|Çok Dilli|
|2000 saat|Henüz Belli Değil|

1. **Yerel çoklu dil desteği**: Sadece Çince, Japonca ve İngilizce ile sınırlı kalmayıp dünya üzerindeki herhangi bir dili destekler.
2. **Dil belirtmeye gerek yok**: Her zaman çoklu dilde olabilir ve diller arasında serbestçe karışık konuşabilirsiniz.
3. **Çoklu dil metin duygu çıkarımı**: Dilin duygu analizini daha doğru yapar, böylece konuşma daha duygusal olur.
4. **Zero Shot iyileştirmesi**: Artık model ince ayarı önerilmez, sadece birkaç saniyelik hedef ses kaydı ile doğrudan zero shot yapılır.
5. **Referans ses kaydı birleşimi**: Birden fazla referans ses kaydı yükleyebilir ve elde edilen ses, birden fazla ses kaydının birleşimi olur.
6. **Daha hızlı çıkarım**: Positional embedding'i RoPE olarak değiştirerek, her bir token çıkarımında tüm dizinin embedding'ini yeniden hesaplama gereksinimini ortadan kaldırır.

### **Veri ve iş birliği çağrısı**: Şu anda veri toplanmaktadır. QQ 1715069210, uygun veri setleri proje kredisi alacaktır.

#### Şu anda kaynak kodda değişiklik fikirleri düzenlenmektedir. # ! araması yaparak açıklamaları bulabilirsiniz. İlginiz varsa yukarıdaki QQ ile iletişime geçmekten çekinmeyin.

### Değişiklik Listesi

#### Kod kitabındaki değişiklikler
Tek kod kitabı -> 2 kod kitabı/4 kod kitabı
#### GPT değişiklikleri
qwen2-0.3b ile değiştirildi
#### Ses kodlamasındaki değişiklikler
cnhubert -> ~~w2v-bert-2.0 (geçici olarak, bu meta tarafından şu anda en çarpıcı 4.6m saatlik çok dilli ön eğitim yapılmaktadır. Sonuç yabancının Çince konuşması gibi olursa cnhubert-large ile değiştirilecek)~~/cnhubert-large/mHubert-147
w2v-bert-2.0 eğitiminin biraz zor olduğunu fark ettim, mHubert-147 eğitimi daha kolay olacak, boyutu dört kat daha küçük ve gerçek testte fp16 doğrudan çöküyor, sadece fp32 kullanılabilir. Ayrıca mHubert zaten yeterince büyük (600MB).
#### Metin kodlamasındaki değişiklikler
Fonemleri ve ilgili embedding'leri kaldırın
cn-roberta -> BGE-m3
#### Pozisyon kodlamasındaki değişiklikler
Metin ve ses kodlamasını ayrı ayrı sinusoidal -> global olarak RoPE embedding yapın.
#### xy birleşim embedding değişikliği (deneysel)
Orijinalden
x1+x2+y1 -> y2
değiştir
x1+y1+x2 -> y2
ve tüm dizi tek bir RoPE embedding'i paylaşır
Teorik olarak, bu şekilde daha fazla hedef ses kaydını genişletmek ve ses hattını birleştirmek daha uygun olur
Örneğin
x1+y1+x2+y2+x3+y3+x4+y4+x5 -> y5
hissi
x1+x2+x3+x4+x5+y1+y2+y3+y4 -> y5
daha doğal olabilir. Kesin olarak kanıtlanamaz.
#### Boyut değişiklikleri
MLP(768, 512) -> ~~MLP olmadan doğrudan 1024 boyut. w2v-bert-2.0 ve bge-m3 her ikisi de 1024 boyut olduğu için mükemmel bir uyum~~ MLP(1024, 768)
#### Eğitim yönteminde değişiklikler
Saf oto-regresif -> oto-regresif + aynı hoparlörde zero shot eğitim örneği regresyonu
#### vits değişiklikleri
Boyutu genişletmenin bir yolunu bulmak (256 -> 512) VITS -> VITS2 (özellikle akış modeline transformer block ekleyerek)
#### Format
Birleştirilmiş ~~yarım hassasiyet~~ tek hassasiyet (gerçek testten sonra yarım hassasiyet çöküyor), hubert 16000 örnekleme vits 32000 örnekleme tüm seslerde ses yüksekliği birleştirme
#### Özet
Aslında genel olarak, değişiklikler temel olarak
1. Daha gelişmiş ön eğitim modelleri kullanmak
2. Daha gelişmiş modeller daha büyük olduğundan, orijinal boyutu genişletmek
3. Zero shot'a dikkat ettiğimiz için eğitim yöntemine zero shot eğitimi eklemek
4. Orijinal kodda yalnızca Çince için bert kullanılıyordu, BGE m3 gibi çok dilli embedding'e değiştirildiğinde tüm dilleri sorunsuzca çıkarabilir
5. Orijinalde sadece tek kod kitabı vardı ve boyutu 1024'tü, bu da orijinal hubert özelliği çıkarma kılavuz yeteneğinin yetersiz olmasına neden oluyordu. Çift kod kitabı ile bilgi miktarı doğrudan 1024^2 = 1048576'ya dönüşür, dört kod kitabı daha abartılıdır ancak yeterli veri olmayabilir, adım adım deneyin
6. Orijinal hızın yavaş olmasının bir nedeni, GPT'nin her seferinde tüm dizinin embedding ve positional embedding'ini yeniden hesaplamasıydı. Ancak RoPE ile değiştirildiğinde bu dezavantaj ortadan kalkar
7. Orijinalde ses hattı birleştirme konusunda ilgi yoktu, daha sonra GPT-SoVITS üzerinde bir dal yaparak ses hattı birleştirmesini gerçekleştirdim, ancak başlangıçta bu hedef tamamen tasarımda yoktu. 花さん'ın videosunda GPT bölümünde kullanılan referans ses kaydından elde edilen ses özelliği ve vits'teki referans ses kaydı farklı olabilir, bu şekilde ses hattı birleştirilebilir. Ancak benim uygulamamda her iki bölümde de birden fazla ses kaydı var
8. Orijinalde mantıksız gördüğüm yerleri değiştirdim. Örneğin, hubert zaten sesin embedding'i olarak kullanılıyordu, token olarak kod kitabı vardı, neden ar_audio_embedding eklememiz gerekiyordu. Ayrıca orijinalde fonem olduğu için fonemlere bir embedding vermek gerekiyordu, çok sayıda ayrı embedding eğitim alıyordu ancak zaten bert ve hubert kullanıyorduk. Ayrıca, ayrı metin embedding ve ses embedding'i bağlamdaki ses ve metni dikkate almayacaktır, doğrudan GPT'ye girip attention ile ilişkileri bulmak daha iyidir

Buraya kadar okuduysanız, anladığınızı gösterir, bu projeye katılmaya davetlisiniz!

**QQ: 1715069210**

**微信: JunityZ**

#### Hızlı Not
Bugün birçok makale okudum, VALLE2 makalesi de dahil, birçok yeni fikir edindim. Şu anda ar_audio_embedding ve ar_text_embedding'in tarihsel bir kalıntı olduğu çok önemli bir konu var.

audioLM, hubert+kmeans kullanarak token elde eden ilk kişiydi, ancak kmean nicemleme eğitimi, tüm verileri öğrenmek zorunda kalmadan doğrudan hubert dağılımından öğreniyor. Bu yüzden bir sonradan embedding eklenir.

Ancak vq kullanılırsa, vq zaten öğrenmeyi gerçekleştirmiştir, bu yüzden vq'ya bir embedding eklemeye gerek yoktur. Burada tarihsel bir sorun olarak sürekli embedding eklenmiştir. Etkisi büyük olmasa da, kaldırıldığında daha mantıklı olur.

Ayrıca, audio lm hem semantic hem de akustik kullanır, bunları sırasıyla hubert ve soundstream aracılığıyla elde eder. Ancak GPT SoVITS de bu özelliğe sahiptir, meltransferencoder akustik elde ederken, hubert semantic elde eder. Bu oldukça tesadüf.

VALLE serisi genellikle EnCodec kullanır, EnCodec doğrudan sesten token elde eder, bu yüzden tekrar bir embedding yapmak gerekir, çünkü baştan embedding çıkarmaz. Ancak hubert kullanmak çok mantıklı değil çünkü hubert'in çıktısı zaten embedding'dir.

Tersine, hubert embedding kullanarak token elde ederiz, EnCodec ise token elde ettikten sonra embedding yapar.

Bu nedenle, orijinal GPTSoVITS ve önceki referans alınan AUdio LM, EnCodec serisi TTS yöntemine dayandığını gibi görünse de, aslında bu ikisi farklıdır.

#### YAPILACAKLAR
Kuantizasyonu yeniden yaz, vector-quantize-pytorch'un Group Residual VQ'sunu doğrudan çağır
