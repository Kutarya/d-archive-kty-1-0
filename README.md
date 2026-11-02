# KTY-1.0 — proje durumu ve agent devir notu

Son doğrulama: 7 Ekim 2026. Workspace: `D:\KTY-1.0`.

KTY-1.0, Türk geliştirici tarafından sıfırdan geliştirilen ve rastgele ağırlıklardan eğitilen, text-only Türkçe autoregressive dil modeli projesidir. Hazır/pretrained model ağırlığı kullanılmaz. Modelin adı KTY-1.0; v2/v3/v4 yalnız checkpoint sürümleridir.

**Mevcut model 50.055.680 parametrelidir, 4B değildir.** Eğitim adımı arttıkça parametre sayısı büyümez. 4B açık kaynak yayın, başarılı tool kullanımı, gelişmiş problem çözme, güçlü kod üretimi ve GPU verimliliği gelecekteki hedeflerdir; mevcut sürümün doğrulanmış yetenekleri olarak sunulmamalıdır.

## Güncel eğitim durumu ve devam planı

- **7 Ekim 2026: eğitim güvenli şekilde durduruldu.** Doğrulanan eğitim süreci Ctrl+C sinyali sonrası mevcut optimizer adımını tamamlayarak kapandı; zorla sonlandırma yapılmadı.
- **Son tamamlanan adım: 8.771/20.000.** Kesinti kaydı: `checkpoints/kty-1.0-v4-50m-interrupted-step008771.pt` (600.821.909 byte). Devamda ilk gerçek eğitim adımı **8.772** olmalıdır.
- Bu dosyada ZIP CRC, strict model yükleme, sonlu ağırlıklar, küçük forward, optimizer state yükleme/adım numaraları, RNG durumları, config ve mevcut dataset fingerprint eşleşmesi doğrulandı. Eğitim/resume başlatılmadı. Sonraki salt okunur çıkarım denemelerinin sonuçları aşağıda.
- Yeni kesinti kaydının `metrics` alanı boş: bu kısa devam koşusunda henüz 9.000-adım validation aralığına ulaşılmadı. Önceki 8.549-adım kaydındaki 8.500-adım validation CE **1,15165**, weighted CE **1,25130** tarihsel sonuçlardır; yeni değerlendirme yapılmadı.
- Ara kayıtlar: 2.000, 4.000, 6.000 ve 8.000 adım. Önceki `interrupted-step008549.pt` dahil eski checkpointler korunuyor. Varsayılan v4 final dosyası henüz oluşmadı.
- `TRAIN_STEPS=20000` hedefi değişmedi. Bu sayı deneysel optimum veya 50M için zorunlu bir değer değildir. Daha fazla eğitim kararı validation ve gerçek cevap değerlendirmesine dayanmalı.
- Bu işlemde yalnız güvenli kesme ve salt okunur checkpoint doğrulaması yapıldı; kod, config ve datasetler değiştirilmedi. README durum notu güncellendi. Kullanıcı istediğinde aşağıdaki komutla devam eder.

**Bu kayıt için doğrudan devam komutu:**

```powershell
cd D:\KTY-1.0
python train.py --resume "checkpoints\kty-1.0-v4-50m-interrupted-step008771.pt"
```

Yalnız `python train.py` çalıştırmak yeni rastgele ağırlıklardan başlatır. Devam sırasında config/datasetleri koru; başlangıç smoke kontrolleri sonrasında `step=8772/20000` görünmelidir. İleride tekrar durdurursan daha yeni kesinti dosyasının adını kullan.

Checkpoint SHA-256: `6cb78e99ddc3b97187d91cd7d39e65b31ed7587b8ed6869d5c3434ce6a02e029`.

### Güvenli durdurma ve kaldığın yerden devam — kısa talimat

1. **Durmak istediğin zaman**, eğitimin çalıştığı VS Code terminaline tıkla; seçili metin varsa seçimi kaldır ve **Ctrl+C'ye bir kez bas**. Ara checkpoint adımını beklemek şart değil.
2. `Kesme isteği alındı; mevcut optimizer adımı tamamlanıp kayıt alınacak.` mesajından sonra bekle. Mevcut optimizer adımı tamamlanır, model ve optimizer kaydedilir. Kayıt sırasında terminali kapatma, Debugger Stop/terminal silme düğmesini kullanma.
3. `Eğitim kesildi; resume checkpoint: D:\KTY-1.0\checkpoints\...` mesajındaki **tam dosya adını not et**. `PS D:\KTY-1.0>` geri dönünce bilgisayarı kapatabilirsin.
4. Tekrar açınca PowerShell'de aşağıdaki komutu kullan. **DOSYA_ADI.pt yerine 3. adımda yazan gerçek kesinti dosyasını koy**; step numarasını tahmin etme.
5. Başlangıçtaki dataset/smoke kontrollerinin bitmesini bekle. İlk gerçek eğitim satırı kayıtlı adımdan **bir sonraki adımla** devam etmelidir; örneğin kayıt 8.123 ise `step=8124/20000`. Kontrollerdeki tek optimizer adımı resume eğitimini sıfırlamaz.

```powershell
cd D:\KTY-1.0
python train.py --resume "checkpoints\DOSYA_ADI.pt"
```

**Yalnız `python train.py` çalıştırma; sıfırdan başlatır.** Resume model, AdamW optimizer, adım ve RNG durumlarını geri yükler. Ara verirken aynı mimari, chat formatı, dataset dosyaları ve eğitim ayarlarını koru; veri fingerprinti ve kritik ayarlar kontrol edilir. Aynı bilgisayarda internet bağlantısı gerekmez; başka bilgisayarda aynı proje dosyaları ve uyumlu Python/PyTorch ortamı gerekir.

Kesinti kaydı kullanılamıyorsa, mevcut 2.000 adımlık ara kayıt yedektir; bu dosyadan devam etmek 2.000'den sonraki kaydedilmemiş ilerlemeyi geri getirmez:

```powershell
python train.py --resume "checkpoints\kty-1.0-v4-50m-step002000.pt"
```

Çıktı veya gelecekteki ara checkpoint adı zaten varsa kod üzerine yazmaz. Bu durumda aynı gerçek resume dosyasını kullanıp `--checkpoint` ile **henüz var olmayan** bir çıkış adı seç. Aşağıdaki örnek ilk ara kayıttan yeni çıkış adıyla devam eder:

```powershell
python train.py --resume "checkpoints\kty-1.0-v4-50m-step002000.pt" --checkpoint "checkpoints\kty-1.0-v4-50m-run2.pt"
```

`run2` yalnız dosya adıdır. Var olan checkpointler korunur.

**Eğitim 20.000 adımda tamamen bittiyse:** final dosya `checkpoints/kty-1.0-v4-50m.pt` olur. Onu `generate.py --checkpoint` ile kullanabilirsin. Daha fazla eğitim istendiğinde mevcut hedefle resume yapılamaz: kullanıcı kararıyla `config.py` içindeki `TRAIN_STEPS` kayıtlı adımdan büyük seçilmeli ve var olmayan yeni çıktı adı kullanılmalı. Bu hedef değişikliği LR schedule'ını da değiştirir; bu README güncellemesinde hiçbir ayar değiştirilmedi.

```powershell
# Yalnız final eğitim bittikten ve TRAIN_STEPS > 20000 ayarlandıktan sonra:
python train.py --resume "checkpoints\kty-1.0-v4-50m.pt" --checkpoint "checkpoints\kty-1.0-v4-50m-devam.pt"
```

İnternet kesilmesi yerel CPU eğitimini etkilemez. Ani elektrik kesilmesi veya zorla kapatma, son tamamlanmış checkpointten sonraki ilerlemeyi kaybettirebilir. Ara kayıtlar her 2.000 adımda alınır; Ctrl+C kaydı herhangi bir tamamlanan optimizer adımında alınabilir.

## Mimari ve eğitim düzeni

Byte token embedding + learned position embedding → 10 pre-LayerNorm Transformer blok → final LayerNorm → LM head. Her blok multi-head causal self-attention, residual bağlantılar, GELU MLP içerir. Attention `scaled_dot_product_attention` kullanır. Çıktı `[B,T,256]`; gelecekteki byte'lara bakmadan next-token tahmini yapılır.

| Ayar | Mevcut değer |
| --- | ---: |
| VOCAB_SIZE | 256 |
| CONTEXT_SIZE | 768 byte |
| EMBEDDING_DIM | 640 |
| NUM_HEADS | 10 |
| NUM_LAYERS | 10 |
| FFN_DIM | 2560 |
| Parametre | 50.055.680 |
| BATCH_SIZE | 2 |
| GRAD_ACCUM_STEPS | 2 |
| Etkin batch | 4 sequence |
| Pozisyon / optimizer adımı | 3.072 |
| TRAIN_STEPS | 20.000 |
| LEARNING_RATE / MIN_LEARNING_RATE | 0,0002 / 0,00002 |
| WARMUP_STEPS | 500 |
| WEIGHT_DECAY / GRAD_CLIP | 0,01 / 1,0 |
| EVAL_INTERVAL / EVAL_BATCHES | 500 / 4 |
| CHECKPOINT_INTERVAL | 2.000 |
| CPU_THREADS | 4 |

Tokenizer UTF-8 byte değerlerini 0–255 kullanır. Token burada kelime değildir; Türkçe karakter birden fazla byte olabilir. Tokenizer değiştirilmedi. 768 byte context uzun belgeler ve büyük kod dosyaları için dar bir sınırdır.

Her adımda tüm corpus işlenmez. Ağırlıklı rastgele kaynak seçimi yapılır; aynı kaynaktan kayıtlar context penceresine paketlenir. Uzun kayıtlarda bazen rastgele başlangıç seçilir. Aynı örnek tekrar gelebilir; sabit sırayla bir epoch bitirme düzeni yoktur. Temel selamlaşma/kimlik/2+2 örnekleri bulunan havuzlarda %25 anchor seçimi vardır.

`X=token[i:i+T]`, `Y=token[i+1:i+T+1]`. Bütün pozisyonlarda cross entropy hesaplanır; kullanıcı/prompt hedefleri 0,35, cevap/sonlandırma hedefleri 1,0 ağırlık alır. İki microbatch loss'u ortalanarak gradient biriktirilir; AdamW, gradient clipping, warmup + cosine LR kullanılır.

Normalize edilmiş soru grupları hash ile train/validation'a ayrılır; sayısal varyasyonlar aynı gruba alınır. Hedef oran %5'tir; gerçek kayıt oranı grup büyüklüklerine göre değişir. Anchor sorular train'de tutulur. Validation 4 sabit batch üzerinden CE ve weighted CE bildirir; kapsamlı kalite benchmarkı değildir.

İlk terminal çıktısında hız yaklaşık 0,21 optimizer adımı/s ve toplam süre tahmini yaklaşık 26 saatti. Daha sonraki 2.000/2.050-adım ekran görüntüsünde hız 0,239 optimizer adımı/s, geçen süre yaklaşık 140 dakika ve kalan ETA 20,9 saatti. Bunlar geçmiş ölçümlerdir; güncel kalan süre için çalışan terminalin ETA değerini esas al. FP32 ağırlık + AdamW durumları bir checkpointte yaklaşık 573 MiB; varsayılan ara/final kayıt bütçesi yaklaşık 5,6 GiB. Kesinti kayıtları ayrıca yer kaplar.

## Veriler: neye göre hazırlandı?

Veriler önceki taleplerdeki Türkçe sohbet, temel matematik, yalnız Python, KTY kimliği ve ton çeşitliliğine göre hazırlanmıştır. Yeni genişletmeler ağırlıklı olarak sentetiktir: programatik soru aileleri, sayısal varyasyonlar, kod örnekleri ve yazılmış sohbet cevapları. Satır sayısı, aynı sayıda bağımsız kavram veya geniş dünya bilgisi demek değildir. Tam kayıt tekrarı olmaması, anlamsal/template tekrarının olmadığı anlamına gelmez.

| Dosya | Kayıt/satır | UTF-8 byte | Kaynak seçim ağırlığı | Kapsam |
| --- | ---: | ---: | ---: | --- |
| data/train.txt | 206 corpus satırı | 32.030 | %1 | Mevcut genel corpus |
| data/kty_chat.jsonl | 1.831 kayıt | 235.980 | %2 | Eski sohbet, kimlik, teknik cevaplar |
| data/sertlik.jsonl | 2.464 kayıt | 560.839 | %4 | Argo, alay, itiraz, sert ama yardımcı cevaplar |
| data/sohbet.jsonl | 2.796 kayıt | 443.168 | %35 | Selamlaşma, günlük hayat, duygu, okul/iş, kimlik |
| data/matematik.jsonl | 19.767 kayıt | 1.787.075 | %20 | Temel işlemler, kesir/yüzde, denklem, günlük problemler |
| data/python.jsonl | 10.925 kayıt | 2.817.257 | %18 | Python öğretimi, kod üretimi, hata düzeltme, temel algoritmalar |
| data/reasoning.jsonl | 12.765 kayıt | 2.620.907 | %20 | Kısa matematik/mantık çözümleri, Python çıktı açıklaması |
| **Toplam** | **50.548 JSONL kaydı + corpus** | **8.497.256** | **%100** | Yaklaşık 8,5 MB ham veri |

Ağırlıklar satır oranları değil, sequence için kaynak seçimi olasılıklarıdır. Eğitim öncesi önceki raporda chat formatına dönüştürülmüş toplam yaklaşık 8.274.244 byte pozisyonu bildirildi; ham dosya boyutundan farklıdır. Yükleme sırasında kaynaklar arası 18 aynı soru/cevap atlanır; dosyalar silinmez/değiştirilmez.

Tüm JSONL satırları tam olarak `{"user":"...","assistant":"..."}` şemasındadır. Eğitimde JSON noktalama işaretleri yerine şu format kullanılır:

```text
Kullanıcı: selam
KTY: Selam, ne yapıyorsun?
<|son|>
```

Matematikte temel aritmetik, negatif/ondalık sayılar, kesir, yüzde, oran, üs/kök, işlem önceliği, basit denklem, para/yaş/zaman/uzunluk/alan/ortalama örnekleri bulunur. Önceki hazırlıkta matematik ve reasoning cevapları programatik denetlendi.

Python içeriği print, değişken/tipler, input, koşul/döngü, koleksiyonlar, fonksiyon/class/inheritance, dosya/pathlib/json, exception, type hints, comprehension, generator, context manager, temel async, temiz kod ve algoritmaları kapsar. Diğer programlama dillerini öğretmek hedeflenmedi. Syntax doğruluğu, her kodun bütün girdilerde doğru çalıştığı anlamına gelmez.

Sertlik verisinde sansürsüz küfür bulunur; amaç her cevapta küfür değil, istenen tona uyumdur. Korunan gruplara nefret söylemi üretmemek hazırlama kuralıdır; eğitimli çıktının bunu her zaman sağladığı henüz ölçülmedi.

### Reasoning'in şu anki anlamı

`[Çözümü açıkla]` etiketli sorularda görünür, kısa bir `Çözüm: ...\nCevap: ...` cevabı öğretilir. Bu bir çıktı biçimi ve problem çözme örnekleri kümesidir; doğrulanmış derin düşünme yeteneği, gizli bir reasoning motoru veya otomatik doğrulama sistemi değildir.

### Tool kullanımı şu an yok

Mevcut datasetler doğal soru/cevap kayıtlarıdır. Model tarafından seçilen araç adı/argümanları, tool sonucu ve takip cevabı içeren çok turlu tool eğitim protokolü yoktur. Araç çalıştıran runtime, sandbox, sonuç doğrulama ve hata sonrası toparlanma döngüsü de uygulanmadı. Python kodu yazmak, modelin o kodu gerçekten çalıştırdığı anlamına gelmez.

## Checkpointler ve deneme komutları

| Dosya | Durum |
| --- | --- |
| checkpoints/kty-1.0.pt | Korunan eski tek-head model |
| checkpoints/kty-1.0-v2.pt | Korunan 491.776 parametreli causal tek-head model |
| checkpoints/kty-1.0-v3.pt | Korunan 2.566.400 parametreli, 2.000 adım eğitilmiş Transformer |
| checkpoints/kty-1.0-v4-50m-step002000.pt | İlk 50M ara kayıt mevcut; 600.821.973 byte; bu kayıt ayrıca değerlendirilmedi |
| checkpoints/kty-1.0-v4-50m-interrupted-step008771.pt | Doğrulanan son kesinti kaydı; devam için kullan |
| checkpoints/kty-1.0-v4-50m.pt | 20.000 adımlık hedef tamamlanırsa final kayıt |

v3 eğitimi kullanıcı raporunda yaklaşık 371 saniyede tamamlandı; train loss 5,53 → 0,69 düştü. Buna rağmen gerçek selamlaşma/kimlik/matematik cevapları zayıftı. Loss düşüşü tek başına kaliteli sohbet kanıtı değildir. v4 yeni rastgele ağırlıklardan başladı; v3 ağırlıkları büyütülerek kullanılmadı.

Checkpoint model/optimizer state, step, mimari/config, eğitim ayarları, RNG durumları, dataset fingerprinti, son validation metrikleri ve chat/sonlandırma/reasoning metadata içerir. Kayıtlar yalnız workspace içindeki `checkpoints/*.pt` konumuna, var olan dosyayı ezmeden yazılır.

İlk checkpointi değerlendirme ve kullanma:

```powershell
python train.py --evaluate-checkpoint "checkpoints\kty-1.0-v4-50m-step002000.pt"
python generate.py --checkpoint "checkpoints\kty-1.0-v4-50m-step002000.pt"
python generate.py --checkpoint "checkpoints\kty-1.0-v4-50m-step002000.pt" --greedy --prompt "selam"
python generate.py --checkpoint "checkpoints\kty-1.0-v4-50m-step002000.pt" --reasoning --prompt "2+2"
```

`--evaluate-checkpoint` eğitim başlatmaz; 10 temel soru üretir, 7 kimlik/kısa aritmetik kontrolü yapar. Selamlaşmalar insan değerlendirmesine bırakılır. Bu küçük kontrol kapsamlı kod/tool/reasoning benchmarkı değildir.

**Ara kaydı açıkça belirt.** Yalnız `python generate.py`, v4 final dosyası yoksa eski v3'e döner; en yeni ara kaydı otomatik seçmez.

Generator promptu içeride ekler, yalnız cevabı gösterir. Temperature 0,6/top_k 10 varsayılandır; greedy, seed, max-new-tokens ve reasoning seçenekleri vardır. UTF-8 byte kısıtı bozuk karakterleri azaltır; anlamsız cevapları düzeltmez. KV cache kullanılır; context dolunca kayan pencere yeniden hesaplanır. Sohbet geçmişi tutulmaz; her mesaj bağımsızdır. Yeni kullanıcı/KTY turu ve v4 sonlandırma işareti generation'ı durdurur.

## 7 Ekim 2026 — 8.771-adım checkpointinin gerçek cevap denemesi

Dosya: `kty-1.0-v4-50m-interrupted-step008771.pt`. Eğitim başlatılmadan CPU/4 thread ile 13 greedy soru ve 3 sampling denemesi yapıldı. Sorular temel kullanım kontrolüdür; eğitimden bağımsız held-out benchmark olduğu iddia edilmez.

| Soru | Gerçek cevap / gözlem | Sonuç |
| --- | --- | --- |
| selam | Selam! Nasıl gidiyor? | Doğal selamlaşma |
| merhaba | Merhaba. Bugün nasılsın? | Doğal selamlaşma |
| nasılsın | Hoş geldin. Ne konuşalım? | Akıcı ama soruya doğrudan cevap değil |
| Sen kimsin? | Ben KTY-1.0. | Doğru kimlik |
| Adın ne? | KTY-1.0. | Doğru kimlik |
| 2+2 | 4 | Doğru |
| 17+28 | 17+28 = 10. | Yanlış; doğru 45 |
| Ali'nin 8 kalemi vardı, 3 tanesini verdi | 8 - 3 = 1. Ali'nin 1 kalemi kaldı. | Yanlış; doğru 5 |
| Bugün çok yorgunum, biraz sohbet edelim. | Dinlenmeye alan açman iyi olabilir. Şu an yenisi kalan işleri başlangıç olabilir. | Başlangıç ilgili, devam bozuk |
| 147 TL bütçe, 3×19 TL ürün, 8 TL kargo; kalan? (reasoning) | Çözüm: Tutar 149 TL. Para üstü 800 - 1499 = 61 TL. Cevap: 61 | Girdileri değiştirmiş; doğru 82 |
| 3x + 5 = 20 (reasoning) | Çözüm: Önce 3 çıkar: 20. Sonra 2'e böl: 10. Cevap: 10 | Yanlış; doğru x=5 |
| Python ile Merhaba yazdıran kod | Optimizer ve model ağırlıkları hakkında ilgisiz metin; kod yok | Görev başarısız |
| Python: listedeki çift sayıları döndüren fonksiyon | Bozuk fonksiyon/class parçaları | Kod bloğu AST/compile kontrolünde SyntaxError |

Sampling temperature=0,6/top_k=10 ile `selam` (seed 42), `Sen kimsin?` (seed 43), `2+2` (seed 44) aynı doğru kısa cevapları verdi. Bu üç deneme daha geniş sampling başarısı kanıtı değildir.

Önceki v3'ün bozuk selamlaşma/kimlik cevaplarına göre ilerleme var; matematik, tutarlı açıklama ve Python üretimi bu küçük denemede güvenilir değil. Reasoning formatını üretmek doğru çözüm ürettiğini göstermedi. Daha fazla adımın bu sorunları gidereceği doğrulanmadı; sonraki karar bu ölçümler ve veri/eğitim değerlendirmesine dayanmalı.

Kendin denemek için (eğitim başlatmaz):

```powershell
python generate.py --checkpoint "checkpoints\kty-1.0-v4-50m-interrupted-step008771.pt"
```

## GPU ve 4B yolunun sınırları

Şu an `train.py` ve `generate.py` CPU'ya açıkça bağlıdır; CUDA device seçimi, mixed precision, GPU bellek ayarı, quantization ve GPU performans doğrulaması hazır değildir. Model katmanları PyTorch ile GPU'ya taşınabilir yapıdadır; mevcut komutlar GPU'da çalışacak şekilde tamamlanmış değildir. KV cache ve SDPA mevcut optimizasyonlardır; bunlar hedef ekran kartında ölçülmüş performans değildir.

50M deney bitince model kendiliğinden 4B'ye dönüşmez. 4B için ayrı mimari ve ayrı eğitim koşusu gerekir; mevcut ağırlıklar doğrudan büyük mimariye strict-load edilemez. Sıfırdan geliştirme/pretrained kullanmama kuralı devam eder.

Ölçek fikri için yalnız ağırlık belleği: 4 milyar parametre FP32 yaklaşık 16 GB, FP16/BF16 yaklaşık 8 GB; ideal 4-bit ham değerler yaklaşık 2 GB. Cache, aktivasyonlar, format metadata, eğitim gradient/optimizer durumları bu sayılara dahil değildir. 4-bit export/inference burada uygulanmadı. Kart/VRAM ve gerçek ölçüm olmadan “4B GPU için optimize” iddiası kurulmaz.

Mevcut yaklaşık 8,5 MB ağırlıklı sentetik veri, güçlü genel amaçlı 4B model iddiasını destekleyen bir corpus değildir. Büyük boyuta geçmeden önce doğal Türkçe dil çeşitliliği, lisansı belli kaynaklar, geniş kod/problemler ve bağımsız kalite ölçümleri gerekir.

## Sonraki agent için çalışma sırası

1. Bu README ve mevcut `config.py`, `tokenizer.py`, `model.py`, `train.py`, `generate.py` dosyalarını oku; checkpoint envanterini yeniden kontrol et. Son doğrulanan durum 8.771. adımda güvenli durma; yeni bir kullanıcı koşusu varsa envanteri/süreci yeniden kontrol et.
2. 8.771-adımlık kesinti kaydını koru; kullanıcı istediğinde bu dosyadan devam eder. Çalışan süreci otomatik öldürme; uzun eğitimi kendiliğinden başlatma.
3. 2.000 adımlık gerçek checkpointte selamlaşma/kimlik, görülmemiş aritmetik, farklı problem ifadeleri, çalıştırılabilir Python testleri ve kısa çözüm tutarlılığını ölç. Önceki v3 ile karşılaştır. Validation eğrisini incele; düşen train loss'u başarı diye sunma.
4. Kalite sonuçlarından sonra adım bütçesini kararlaştır. `TRAIN_STEPS` değişimi LR schedule'ını da değiştirir. Çalışan Python süreci config değişikliklerini yeniden okumaz. Resume hedefi kayıtlı adımdan büyük olmalı; eşitse kod devamı reddeder.
5. Tool hedefi için ayrı ve net bir protokol tasarla: araç şeması → çağrı/argümanlar → araç sonucu → takip cevabı. Araç gerektirmeyen sorular, bozuk sonuç, timeout, başarısız çağrı ve çok adımlı toparlanma örnekleri ekle; sonuca göre görev başarısını ölçen testler kur. Model çıktısını çalıştıran uygulama tarafını ayrıca geliştir.
6. Reasoning/kod hedeflerini görülmemiş çok adımlı görevler ve gerçek kod testleriyle ölç. Daha uzun açıklamayı doğru muhakeme kabul etme. Eksik bilgi isteme, hatasını düzeltme ve farklı problem aileleri ekle.
7. GPU yolunu gerçek GPU/VRAM ve kurulu PyTorch build'ini ölçerek hazırla: device yönetimi, desteklenen dtype, mixed precision ve resume için gerekli RNG/device durumları. Mevcut CPU checkpointlerini koru; GPU smoke ve gerçek bellek/hız ölçümü olmadan hazır deme.
8. Ancak yeterli veri, kaynak ve ölçüm planı oluşunca 4B mimari/eğitim bütçesini belirle. Tokenizer/context değişimi gerekiyorsa ayrı sürüm ve geçiş planı hazırla; byte checkpointlerini uyumsuz yeni tokenizer ile kullanma.
9. GitHub açık kaynak yayını için kod lisansını, veri kullanım/dağıtım izinlerini, model ağırlığı lisansını, büyük dosya depolama yöntemini ve model kartını kullanıcıyla netleştir. Bu projede şu an lisans dosyası yok; yayınlanmış veya lisanslanmış sayma. Sürüm adı, gerçek parametre sayısı, checkpoint SHA-256, değerlendirme sonuçları ve sınırlamaları yayınla. Kullanıcı açıkça istemeden repo/push/yayın yapma.

Yeni eğitim verisi eklemek mevcut dataset fingerprintini değiştirir ve mevcut resume kontrolünü engeller. Önce mevcut koşuyu koru; veri büyütmeyi yeni koşu veya açıkça tasarlanmış devam stratejisi olarak ele al. Metadata kontrolünü sessizce kaldırarak devam etme.

## Doğrulama durumu

İlk README hazırlığında **yeniden doğrulananlar**: 50.548 JSONL kaydı; UTF-8 strict okuma; tam user/assistant şeması; boş alan olmaması; her dosyada tam kayıt tekrarı 0; Python dosyasındaki 10.925 ve reasoning girdilerindeki 2.508 Python kod bloğunun AST/compile syntax kontrolü. Parametre sayısı model katman şekillerinden yeniden hesaplandı. Mevcut üç checkpointin boyut ve hash envanteri okundu.

**Önceki hazırlıkta geçen smoke kontrolleri**: gerçek 50M shape/CE/backward/tek optimizer adımı, causal prefix/geleceğe sıfır gradient, KV cache eşitliği; küçük test mimarisinde checkpoint serialization + optimizer/RNG resume ve accumulation eşitliği; validation/split/RNG; eski checkpoint strict-load ve overwrite koruması; generation biçimi/UTF-8/EOS. Checkpoint save/load smoke testi RAM nedeniyle küçük mimarideydi, gerçek 50M diske yazma testi değildi. Bu dokümantasyon işi sırasında ağır smoke yeniden çalıştırılmadı ve eğitim başlatılmadı.

**Henüz ölçülmeyenler**: kapsamlı v4 cevap kalitesi benchmarkı, tool görev başarısı, gelişmiş reasoning/kod benchmarkları, GPU bellek/hızı ve 4B model kalitesi. 8.771-adımlık kaydın aşağıdaki küçük çıkarım denemesi tamamlandı; genelleme benchmarkı olarak sunulmamalı.

Son README güncellemesinde durdurma/resume talimatları mevcut `train.py` ile karşılaştırıldı, config ve checkpoint dosya envanteri salt okunur incelendi. Yeni 2.000-adım checkpointi tespit edildi; ağır smoke, generation veya resume çalıştırılmadı. Dataset ve Python dosyalarının değişmezliği hash ile kontrol edildi.

Eğitim başlamadan doğrulama komutu:

```powershell
python train.py --check
```

Bu komut uzun eğitim başlatmaz; ancak ağır 50M smoke forward/backward yapar. Aktif eğitim varken gereksiz yere paralel çalıştırma.

## Dosya görevleri ve korunacak kurallar

- `config.py`: mimari, eğitim, veri ağırlıkları ve generation ayarları.
- `tokenizer.py`: mevcut UTF-8 byte tokenizer.
- `model.py`: decoder Transformer, KV cache, eski checkpoint yükleme.
- `train.py`: parse/validation, örnekleme, smoke, eğitim, güvenli kesme/resume, cevap kontrolü.
- `generate.py`: terminal çıkarımı, sampling/greedy, UTF-8 ve tur sonlandırma.
- `data/`: korunacak corpus ve datasetler.
- `checkpoints/`: korunacak eski ve yeni eğitim kayıtları.
- `README.md`: bu devir notu. İlk devir notu olarak eklendi; son işlemde yalnız bu dosya güncellendi.

Yalnız mevcut workspace içinde çalış. Yeni proje/repository/kopya/temp/masaüstü klasörü oluşturma, workspace dışında dosya yazma. Eski dataset ve checkpointleri silme/ezme. Gereksiz utility/setup/installer ekleme. Mevcut dosyaları incelemeden yeniden yazma. Kullanıcı eğitimi kendisi başlatır.

6 Ekim 2026 veri kimlikleri; resume uyumu için referans:

```text
kty_chat.jsonl 790aeb949ddad1eef58c6ace4e10264e4fe8ac9b636117ac733be09c14c727bc
matematik.jsonl 6646aee1610d04d3ceeda5bbf97eda5f94e20886f948ff85546163aff16492fc
python.jsonl a5c6c2530857cea92b738e41c2415222fe11dcb351b2da5f83be7fa375502971
reasoning.jsonl ee3a51d72ee0730d4d4c23c13af356c5c3f92eda719c136cdc320a414906860d
sertlik.jsonl 93a49a5fb4ddb1dd9022a306a809df66df283d1d0748f4ae42b1f8b3b6c6140f
sohbet.jsonl 26d104939c5bcece906385e21d58f07df91fb022b5a167f5b0e2ab09aef46684
train.txt 34545cb89837d57434da4bed4775f6cb8c72f02073afe1b236ecc306f69e2393
```
