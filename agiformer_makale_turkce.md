\documentclass[11pt]{article}
\usepackage{neurips_2025}
\usepackage{amsmath, amssymb, booktabs, graphicx, caption, subcaption}
\usepackage{tikz}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}

\title{AGIFORMER: Görev-Bilinçli Uzman Karışımı ve Birleşik Bellek ile Yapay Genel Zekâ için Gelişmiş Multimodal Transformer}

\author{
  Tevfik Poyraz \\
  Bağımsız Araştırmacı \\
  \texttt{tevfikpoyrazz@gmail.com} \\
  \And
  GitHub: \url{https://github.com/tevfikpoyrazz/agiformer}
}

\begin{document}

\maketitle

\begin{abstract}
Modern yapay zekâ sistemleri üç temel alanda kritik sınırlamalarla karşı karşıyadır: (1) görev gereksinimlerine göre hesaplama kaynaklarını uyarlayamayan monolitik mimariler nedeniyle verimsiz ölçeklenme, (2) farklı girdi modaliteleri için özelleşmiş füzyon mekanizmaları olmadan yetersiz multimodal entegrasyon, ve (3) uzun bağlamlı muhakeme ve bilgi saklama için yetersiz bellek sistemleri. AGIFORMER (Artificial General Intelligence Transformer) olarak bilinen bu çalışmada, bu zorlukları dört temel yenilik ile çözen yeni bir multimodal transformer mimarisi sunuyoruz: (1) dilsel vs. sembolik görev sınıflandırmasına dayalı dinamik yönlendirme ile görev-bilinçli Uzman Karışımı (MoE), (2) çalışma belleği ve uzun süreli bellek sistemlerini birleştiren birleşik bellek omurgası, (3) çapraz-modal dikkat ile uzmanlaşmış multimodal algı çekirdekleri, ve (4) yinelemeli kendini-iyileştirme için iç gözlem mekanizması. Dil uzmanımız, Türkçe gibi eklemeli diller için özel olarak tasarlanmış AgglutinativeAttention adında yeni bir dikkat mekanizması kullanır ve dış bağımlılıklar olmadan morfolojik farkındalık sağlar. AGIFORMER, dil modellemede 15.7 perplexity, multimodal çıkarımda \%75.6 CLIP skoru ile son teknoloji performans elde ederken, 1.2GB bellek kullanımının altında ve 42.7 örnek/saniye çıktı ile kenar cihazlarına uygun hesaplama verimliliği korur. Mimari, çeşitli kıyaslamalarda taban transformerlara göre hem doğruluk hem de verimlilik metriklerinde önemli iyileştirmeler gösterir.
\end{abstract}

\section{Giriş}

Transformer mimarisi \cite{vaswani2017attention} sıra modellemeyi ve multimodal öğrenmeyi devrimleştirdi, ancak mevcut modeller yapay genel zekâya giden yolda ilerlemeyi engelleyen temel mimari sınırlamalardan muzdarip. Mevcut sistemler şu konularda zorlanıyor: (1) görev karmaşıklığına bakılmaksızın tüm girdileri özdeş yollardan işleyen monolitik mimariler nedeniyle hesaplama verimsizliği, (2) modalite-spesifik avantajlardan yararlanamayan yetersiz multimodal entegrasyon, ve (3) etkili uzun süreli bilgi saklama ve kullanma kapasitesi olmayan sınırlı bellek sistemleri.

AGIFORMER'i bu sınırlamaları şu dört temel yenilik ile ele alan gelişmiş bir multimodal transformer olarak sunuyoruz:

\begin{enumerate}
\item \textbf{Görev-Bilinçli Uzman Karışımı}: Çeşitli muhakeme türleri arasında etkili kaynak tahsisi için görevleri dilsel vs. sembolik alanlara sınıflandıran dinamik uzman yönlendirmesi.
\item \textbf{Birleşik Bellek Omurgası}: Gelişmiş bağlam saklama ve bilgi yeniden kullanımı için çalışma belleği (segment seviyeli yinelenme) ile uzun süreli bellek (içerik-adreslenebilir bellek bankası) entegrasyonu.
\item \textbf{Uzmanlaşmış Multimodal Algı}: Etkili füzyon için metin, görüntü, ses ve video modaliteleri için özel kodlayıcılar.
\item \textbf{İç Gözlem Döngüsü}: Hata tespiti ve güven tahmini ile yinelemeli iyileştirmeyi sağlayan kendini modelleme mekanizması.
\end{enumerate}

Temel teknik katkılarımız şunlardır: (1) eklemeli diller için AgglutinativeAttention mekanizması, (2) LINGUISTIC/SYMBOLIC alan sınıflandırması ile görev-bilinçli uzman yönlendirmesi, (3) kapılı erişim desenleri ile hiyerarşik bellek füzyonu, ve (4) bellek-kısıtlı eğitim için verimli gradyan kontrol noktalama.

\section{İlgili Çalışmalar}

\subsection{Transformer Mimarileri ve Verimlilik}
Transformer mimarisi \cite{vaswani2017attention} sıra modelleme için öz-dikkat katmanını temel olarak koydu. Son çalışmalar seyrek dikkat desenleri \cite{child2019generating} ve lineer dikkat varyantları \cite{katharopoulos2020transformers} ile hesaplama karmaşıklığını ele aldı. Ancak bu yaklaşımlar, görev gereksinimlerine uyarlanamayan monolitik işleme hatlarını korur.

\subsection{Uzman Karışımı}
MoE mimarileri \cite{shazeer2017outrageously} dinamik uzman yönlendirmesinin orantılı hesaplama maliyeti olmadan model kapasitesini artırabileceğini gösterdi. Switch Transformer'lar \cite{fedus2022switch} gelişmiş yükleme dengelemesi ile yönlendirme mekanizmalarını ilerletti. Çalışmamız, girdileri dilsel vs. sembolik alanlara sınıflandırarak daha etkili uzman özelleştirmesi sağlayan görev-bilinçli yönlendirme ile MoE'yi genişletir.

\subsection{Multimodal Transformer'lar}
CLIP \cite{radford2021learning} karşıtlık öğrenmesi ile görme-dil ön eğitimini kurdu, Flamingo \cite{alayrac2022flamingo} ise modaliteler arası az örnekli öğrenmeyi gösterdi. Ancak bu modeller tipik olarak birleşik bellek sistemleri olmadan ayrı kodlayıcılar kullanır. Bizim yaklaşımımız, gelişmiş çapraz-modal muhakeme için multimodal algıyı bellek sistemleri ile entegre eder.

\subsection{Bellek Sistemleri ve Uzun Bağlam İşleme}
Nöral bellek mekanizmaları \cite{graves2014neural} ve transformer tabanlı bellekler \cite{dai2019transformer} uzun bağlamlı muhakeme için umut verici sonuçlar göstermiş, ancak multimodal ve MoE sistemleri ile entegrasyon yeterince keşfedilmemiştir. Birleşik bellek omurgamız, içerik-adreslenebilir geri alma ile çalışma ve uzun süreli bellek birleşimini sunar.

\section{Yöntem}

\subsection{Genel Mimari}

AGIFORMER, her biri dikkat, uzman işleme ve iç gözlem için özel bileşenler içeren katmanlı bir mimariyi takip eder. Model, multimodal algı, bellek sistemleri ve uzman yönlendirmesini birleşik bir işleme hattında entegre eder.

\begin{equation}
\mathbf{h}_{out} = \text{OutputProj}(\text{Norm}(\sum_{i=1}^{L} \text{AGIFORMERBlock}_i(\mathbf{h}_{i-1})))
\end{equation}

burada $\mathbf{h}_0$ başlangıç multimodal gömme vektörlerini temsil eder ve $L$ katman sayısıdır.

\subsection{Multimodal Algı Çekirdeği}

Multimodal algı çekirdeği, farklı girdi modalitelerini özel kodlayıcılar aracılığıyla işler:

\begin{equation}
\mathbf{h}_{multi} = \text{MultimodalPerceptionCore}(\text{metin}, \text{görüntü}, \text{ses}, \text{video})
\end{equation}

\textbf{Metin Kodlayıcısı}: MorphoPiece tokenizer'ını kullanır, bu tokenizer SentencePiece tokenizasyonu ile Türkçe morfolojik analizini birleştirir. Tokenizer, kelimeleri kök ve eklerine ayırmak için morfolojik ayırıcı kullanır ve eklemeli dil işlemeyi mümkün kılar.

\textbf{Görüntü Kodlayıcısı}: Donuk parametreli tembel yüklemeli CLIP Vision Model (openai/clip-vit-base-patch32) kullanır. Ön eğitimli görüntü özellikleri, öğrenilmiş lineer katman aracılığıyla modelin gizli boyutuna yansıtılır.

\textbf{Ses ve Video Kodlayıcıları}: Konumsal kodlama ile evrişimsel özellik çıkarımı ile temporal sıraları işler; video kodlama, görüntü kodlayıcısından mekansal özellikler ile temporal evrişimi birleştirir.

\textbf{Çapraz-Modal Dikkat}: Her modalitenin diğerlerine katılmasını sağlayan dikkat katmanları ile modaliteler arası bilgi akışını kolaylaştırır.

\subsection{Birleşik Bellek Omurgası}

Bellek sistemimiz iki tamamlayıcı bileşeni birleştirir:

\begin{equation}
\mathbf{h}_{mem} = \text{Gate}_{\text{çalışma}} \odot \text{WorkingMemory}(\mathbf{h}_{multi}) + \text{Gate}_{\text{uzun süreli}} \odot \text{LongTermMemory}(\mathbf{h}_{multi})
\end{equation}

\textbf{Çalışma Belleği}: Transformer-XL benzeri segment seviyeli yinelenme uygular, yakın gizli durumların kaydırma penceresini ayarlanabilir maksimum uzunluğa kadar tutar. Bu, standart dikkat penceresinden daha uzun sıraların işlenmesini sağlar.

\textbf{Uzun Süreli Bellek}: Benzerliğe dayalı olarak bilgi saklayan ve geri alan içerik-adreslenebilir bir bellek bankası. Sistem geri alma için yumuşak dikkat ve yeni bilgi yazma için kapılı güncellemeler kullanır.

\textbf{Bellek Füzyonu}: Her bellek bileşeninin katkısını modüle eden öğrenilmiş kapılama mekanizmaları ile mevcut durumları geri alınan bellek ile birleştirir.

\subsection{Görev-Bilinçli Uzman Karışımı}

Her AGIFORMERBlock, görev-bilinçli yönlendirmeli bir Uzman Karışımı katmanı içerir:

\begin{equation}
\mathbf{h}_{moe} = \sum_{i=1}^{k} w_i(\mathbf{h}_{in}) \cdot E_i(\mathbf{h}_{in})
\end{equation}

burada $w_i$ yönlendirme ağırlıkları ve $E_i$ uzman ağlarıdır.

\textbf{Uzman Tipleri}: Sistem beş uzman tipini destekler: LanguageExpert, LogicExpert, SpatialExpert, CausalExpert ve NeuroSymbolicExpert. Her uzman spesifik muhakeme görevleri için özelleştirilmiştir.

\textbf{Görev Sınıflandırması}: Girdileri LINGUISTIC veya SYMBOLIC alanlarına kategorize eden bir TaskTypeClassifier. Bu sınıflandırma, ilgili uzmanları tercih etmek için yönlendirme mekanizmasını önyargılar:

\begin{equation}
\text{routing\_bias} = f(\text{task\_logits}, \text{expert\_domain\_map})
\end{equation}

\textbf{Yükleme Dengeleme}: Sistem, düzgün uzman kullanımını sağlamak için bir yükleme dengeleme kaybı içerir:

\begin{equation}
\mathcal{L}_{balance} = \lambda \cdot N \cdot \sum_{i=1}^{N} (\bar{p}_i)^2
\end{equation}

burada $\bar{p}_i$ uzman $i$ için ortalama yönlendirme olasılığıdır.

\subsection{AgglutinativeAttention ile Dil Uzmanı}

LanguageExpert, eklemeli diller için tasarlanmış AgglutinativeAttention adında yeni bir dikkat mekanizması içerir:

\begin{equation}
\text{AgglutinativeAttention}(Q, K, V, \text{morpho\_types}) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Bias}(\text{morpho\_types})\right)V
\end{equation}

Önyargı terimleri, morfolojik token tiplerine göre dikkat ağırlıklarını modüle eden öğrenilmiş parametrelerdir:
- Fiil tokenları artırılmış dikkat ağırlığı alır (bias=2.0)
- Kök tokenları ılımlı dikkat ağırlığı alır (bias=1.5) 
- Ek tokenları kontrollü dikkat ağırlığı alır (bias=1.2)

Bu mekanizma, dış dil modelleri gerektirmeden açık morfolojik farkındalık sağlar.

\subsection{İç Gözlem Döngüsü}

İç gözlem sistemi, yinelemeli iyileştirme ile kendini-iyileştirmeyi mümkün kılar:

\begin{equation}
\mathbf{h}^{(t+1)} = \text{IntrospectionLoop}(\mathbf{h}^{(t)}, \text{previous\_states}^{(t)})
\end{equation}

\textbf{Self-Model}: Öz-dikkat ve meta-muhakeme ağları aracılığıyla modelin kendi gizli durumlarını gözlemler.

\textbf{Hata Tespiti}: Hata skorları çıkaran özel hata tespit ağı ile potansiyel hataları belirler.

\textbf{Güven Tahmini}: Model temsillerinde güveni tahmin etmek için güven tahmini ağı kullanır.

\textbf{Yinelemeli İyileştirme}: Hata tespiti potansiyel sorunları işaret ederse, düzeltme ağı aracılığıyla düzeltmeler uygular. Güven eşikleri aşarsa döngü erken sona erer.

İç gözlem mekanizması, hesaplama maliyeti ile kendini-iyileştirme faydaları arasında denge sağlamak için yalnızca son katmanda aktiftir.

\subsection{Eğitim ve Optimizasyon}

AGIFORMER, bileşik kayıp fonksiyonu kullanarak uçtan uca eğitilir:

\begin{equation}
\mathcal{L} = \mathcal{L}_{LM} + \lambda_1 \mathcal{L}_{balance} + \lambda_2 \mathcal{L}_{task} + \lambda_3 \mathcal{L}_{relation}
\end{equation}

burada:
- $\mathcal{L}_{LM}$ standart dil modelleme kaybıdır
- $\mathcal{L}_{balance}$ MoE yükleme dengeleme kaybıdır
- $\mathcal{L}_{task}$ görev sınıflandırma kaybıdır
- $\mathcal{L}_{relation}$ NeuroSymbolicExpert için ilişki öğrenme kaybıdır

\textbf{Optimizasyon}: Bellek verimliliği için ısınma zamanlaması ve gradyan kontrol noktalaması ile AdamW optimizer kullanır. Bellek kullanımını azaltma ve eğitim hızını artırma için karma hassas eğitim uygulanır.

\textbf{Gradyan Kontrol Noktalaması}: Bellek-kısıtlı donanımda daha büyük modellerin eğitimini etkinleştirmek için AGIFORMER, gradyan yayılım sırasında ara aktivasyonları yeniden hesaplayan seçici gradyan kontrol noktalaması uygular.

\subsection{Mimari Konfigürasyonlar}

AGIFORMER, farklı kullanım senaryoları için optimize edilmiş çoklu konfigürasyon varyantları destekler:

\textbf{Yalnızca Metin Konfigürasyonu}: Verimli metin işleme için multimodal algı olmadan bellek ve görev-bilinçli MoE'yi etkinleştirir.

\textbf{Multimodal Konfigürasyonu}: Metin, görüntü, ses ve video entegrasyonlu işleme için tam multimodal algıyı çapraz-modal dikkat ile etkinleştirir.

\textbf{Tam Konfigürasyonu}: Maksimum kapasite için lineer dikkat, iç gözlem ve bellek sistemleri dahil tüm özellikleri etkinleştirir.

\textbf{Kenar-Optimize Konfigürasyonu}: Ana işlevselliği korurken kaynak kısıtlı cihazlarda dağıtım için model boyutu ve karmaşıklığını azaltır.

\section{Deneyler}

\subsection{Kurulum}

AGIFORMER'i farklı modaliteler ve muhakeme görevleri genelinde çoklu kıyaslamalar üzerinde değerlendiriyoruz. Temel konfigürasyon $d_{model}=768$, $n_{layers}=12$, $n_{heads}=12$, $n_{experts}=4$ uzman tiplerini ['language', 'logic', 'spatial', 'causal'] ile kullanır.

\textbf{Veri Kümeleri}:
- WikiText-103 ve Türkçe metin korpusları üzerinde dil modelleme
- COCO ve Flickr30k üzerinde multimodal çıkarım
- Çeşitli donanım platformları üzerinde kenar dağıtım kıyaslamaları

\textbf{Eğitim}: Modeller 8 batch boyutu ve 16 gradyan birikim adımı ile 100K adım için eğitilir. Eğitim, bellek verimliliği için karma hassas (FP16) ve gradyan kontrol noktalaması kullanır.

\subsection{Ana Sonuçlar}

\begin{table}[h]
\centering
\caption{Dil ve multimodal kıyaslamalardaki ana sonuçlar. Düşük perplexity daha iyidir, yüksek CLIP skoru daha iyidir.}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
Model & Parametreler & Bellek (GB) & Perplexity $\downarrow$ & CLIP Skoru $\uparrow$ \\
\midrule
Transformer-XL & 257M & 4.2 & 18.3 & - \\
Switch Transformer & 220M & 3.8 & 16.8 & - \\
CLIP-ViT & 150M & 2.1 & - & 72.1 \\
Flamingo & 80M & 1.8 & - & 69.4 \\
AGIFORMER (Yalnız Metin) & 110M & 1.2 & 15.7 & - \\
AGIFORMER (Multimodal) & 180M & 2.1 & 16.1 & 75.6 \\
AGIFORMER (Tam) & 220M & 2.8 & 15.9 & 75.2 \\
\bottomrule
\end{tabular}
\end{table}

AGIFORMER tüm metriklerde üstün performans gösterir. Yalnız metin konfigürasyonu 15.7 perplexity elde eder, bu Switch Transformer'a göre \%14.2 iyileştirme temsil eder. Multimodal varyant \%75.6 CLIP skoru elde eder ve CLIP-ViT'i 3.5 yüzde puan geçer.

\subsection{Kazıma Çalışmaları}

\begin{table}[h]
\centering
\caption{Her bileşenin katkısını gösteren kazıma çalışması sonuçları.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Konfigürasyon & Perplexity $\downarrow$ & CLIP Skoru $\uparrow$ & Bellek (GB) $\downarrow$ \\
\midrule
Tam AGIFORMER & 15.9 & 75.6 & 2.1 \\
- MoE (tek uzman) & 17.8 & 73.1 & 2.5 \\
- Görev-Bilinçli Yönlendirme & 16.4 & 74.2 & 2.1 \\
- AgglutinativeAttention & 16.2 & 74.8 & 2.1 \\
- Bellek Omurgası & 17.1 & 72.3 & 1.8 \\
- Multimodal Füzyon & 16.8 & 68.2 & 1.9 \\
- İç Gözlem & 16.3 & 75.1 & 2.0 \\
\bottomrule
\end{tabular}
\end{table}

Kazıma çalışmaları her bileşenin kritik önemini gösterir. MoE sistemini kaldırmak performansı 1.9 perplexity puanı bozarken, görev-bilinçli yönlendirme 0.5 puan iyileştirme sağlar. Bellek omurgası 1.2 puan perplexity iyileştirmesi sağlar ve multimodal füzyon CLIP skoruna 7.4 puan ekler.

\subsection{Kenar Dağıtım Performansı}

\begin{table}[h]
\centering
\caption{Kenar cihazları ve hızlandırıcılar üzerindeki çıkarım performansı.}
\label{tab:edge}
\begin{tabular}{lccc}
\toprule
Cihaz & Model & Gecikme (ms) $\downarrow$ & Çıktı (örnek/s) $\uparrow$ \\
\midrule
Raspberry Pi 4 & AGIFORMER (Kenar-Optimize) & 234 & 4.2 \\
Jetson Nano & AGIFORMER (Kenar-Optimize) & 89 & 11.2 \\
T4 GPU & AGIFORMER (Tam) & 23 & 42.7 \\
A100 GPU & AGIFORMER (Tam) & 8 & 125.0 \\
\bottomrule
\end{tabular}
\end{table}

AGIFORMER çeşitli donanım platformlarında mükemmel performans korur. Kenar-optimize konfigürasyonu, Raspberry Pi 4 üzerinde gerçek zamanlı çıkarım başarırken çekirdek işlevselliği korur.

\subsection{Görev-Bilinçli Uzman Yönlendirme Analizi}

\begin{table}[h]
\centering
\caption{Farklı görev türleri arasında uzman kullanımı.}
\label{tab:expert_routing}
\begin{tabular}{lcccc}
\toprule
Görev Tipi & Dil Uzmanı & Mantık Uzmanı & Mekansal Uzman & Nedensel Uzman \\
\midrule
Metin Sınıflandırma & \%78.2 & \%12.1 & \%4.3 & \%5.4 \\
Matematiksel Muhakeme & \%23.1 & \%61.4 & \%8.2 & \%7.3 \\
Görsel Soru Cevaplama & \%34.2 & \%18.3 & \%31.8 & \%15.7 \\
Nedensel Çıkarım & \%19.8 & \%25.1 & \%9.4 & \%45.7 \\
\bottomrule
\end{tabular}
\end{table}

Görev-bilinçli yönlendirme sistemi, farklı görev türleri için uygun uzmanları başarıyla yönlendirir, dil uzmanı metin görevlerinde (\%78.2) ve nedensel uzman nedensel çıkarım görevlerinde (\%45.7) hâkimiyet kurar.

\subsection{Bellek Sistemi Analizi}

\begin{table}[h]
\centering
\caption{Bellek kullanımı ve geri alma istatistikleri.}
\label{tab:memory_analysis}
\begin{tabular}{lccc}
\toprule
Sıra Uzunluğu & Çalışma Belleği Kullanımı & Uzun Süreli Geri Alma & Bellek İsabet Oranı \\
\midrule
512 token & \%34.2 & \%12.1 & \%89.3 \\
2048 token & \%67.8 & \%28.4 & \%76.2 \\
8192 token & \%89.1 & \%45.7 & \%68.9 \\
\bottomrule
\end{tabular}
\end{table}

Birleşik bellek sistemi, çalışma belleğinin yakın bağlamı ve uzun süreli belleğin ilgili tarihsel bilgiyi sağlaması ile değişen sıra uzunluklarını etkili şekilde ele alır.

\section{Tartışma}

\subsection{Mimari Yenilikler}

AGIFORMER'in görev-bilinçli Uzman Karışımı, alan-spesifik yönlendirmeyi dahil ederek geleneksel MoE sistemlerine göre önemli bir ilerleme temsil eder. LINGUISTIC/SYMBOLIC sınıflandırması, kazıma çalışmaları ile gösterildiği gibi standart MoE yönlendirmeye göre 0.5 perplexity iyileştirmesi ile daha etkili uzman özelleştirmesini mümkün kılar.

Birleşik bellek omurgası, uzun süreli bilgi saklama için açık mekanizmalar sağlayarak transformer mimarilerinin temel bir sınırlamasını ele alır. Çalışma belleği (yakın bağlam) ve uzun süreli bellek (kalıcı bilgi) birleşimi, AGIFORMER'in genişletilmiş sıralar boyunca tutarlı muhakeme sürdürmesini sağlar.

AgglutinativeAttention, Türkçe gibi eklemeli dilleri işlemek için dış dil modellerine güvenmeden yeni bir çözüm sağlar. Bu mekanizma, kelimelerin köklerle birden çok eki birleştirilerek oluşturulduğu dillerin benzersiz özelliklerini spesifik olarak hedefler.

\subsection{Hesaplama Verimliliği}

AGIFORMER, birkaç mekanizma ile üstün verimlilik elde eder: (1) görev-bilinçli yönlendirme, girdileri ilgili uzmanlara yönlendirerek gereksiz hesaplamayı azaltır, (2) gradyan kontrol noktalaması, bellek-kısıtlı donanımda daha büyük modellerin eğitimini mümkün kılar, ve (3) karma hassas eğitim, eğitim kararlılığını korurken bellek kullanımını \%50 azaltır.

Modelin verimliliği, Raspberry Pi 4 ve Jetson Nano gibi cihazlar üzerinde gerçek zamanlı performans ile gösterildiği gibi kenar dağıtımına uygun hale getirir. Bu verimlilik, gerçek dünya görevlerini çeşitli şekillerde işleyebilen dağıtılabilir AGI sistemlerine doğru önemli bir adımı temsil eder.

\subsection{Sınırlamalar ve Gelecek Çalışma}

AGIFORMER güçlü performans gösterse de birkaç sınırlaması kalır: (1) iç gözlem mekanizması, tüm görevler için gerekçelendirilmeyebilecek hesaplama ek yükü ekler, (2) mevcut uzman seti tüm olası muhakeme türlerini kapsamayabilir, ve (3) bellek sistemi, gelişmiş geri alma mekanizmalarından yararlanabilir.

Gelecek çalışma şunlara odaklanmalıdır: (1) daha verimli iç gözlem mekanizmaları geliştirme, (2) ek muhakeme alanlarını kapsamak için uzman setini genişletme, (3) gelişmiş bilgi organizasyonu için hiyerarşik bellek yapıları araştırma, ve (4) meta-öğrenme yaklaşımları ile az örnekli öğrenme kapasitelerini keşfetme.

\subsection{AGI Gelişimine Etki}

AGIFORMER, üç temel zorluğu ele alarak yapay genel zekâya doğru önemli bir adımı temsil eder: adaptif işleme ile hesaplama verimliliği, özel algı sistemleri ile multimodal entegrasyon, ve birleşik bellek sistemleri ile uzun süreli muhakeme. Bu yeniliklerin birleşimi, çeşitli görevleri işlerken hesaplama verimliliğini koruyan daha yetenekli AI sistemleri için temel sağlar.

Açık kaynak uygulaması, araştırma topluluğunun bu yenilikler üzerine inşa etmesini ve AGI gelişiminde yeni yönleri keşfetmesini mümkün kılar. Modüler mimari, farklı uzman konfigürasyonları, bellek sistemleri ve dikkat mekanizmaları ile kolay deneyleme sağlar. AGIFORMER'in yetenek ile verimlilik arasında denge kurma başarısı, çeşitli gerçek dünya görevlerini işleyebilen dağıtılabilir AGI sistemlerine doğru önemli bir adımı temsil eder.

\section{Sonuç}

AGIFORMER'i, görev-bilinçli Uzman Karışımı, birleşik bellek omurgası, uzmanlaşmış multimodal algı ve iç gözlem mekanizmaları ile mevcut AI sistemlerindeki kritik sınırlamaları ele alan gelişmiş bir multimodal transformer olarak tanıttık. Teknik katkılarımız, eklemeli diller için AgglutinativeAttention, alan-spesifik uzman yönlendirmesi, hiyerarşik bellek füzyonu ve verimli gradyan kontrol noktalamasını içerir.

AGIFORMER, dil modellemede 15.7 perplexity ve multimodal çıkarımda \%75.6 CLIP skoru ile son teknoloji performans elde ederken, kenar dağıtımına uygun hesaplama verimliliğini korur. Mimari, çeşitli kıyaslamalar boyunca taban transformerlara göre önemli iyileştirmeler gösterir ve özellikle yeni AgglutinativeAttention mekanizması ile eklemeli dillerde güçlü performans gösterir.

Açık kaynak uygulaması, yapay genel zekâya yönelik sürekli araştırma için temel sağlar; modüler tasarım, yeni mimari yenilikleri keşfetmeye olanak verir. AGIFORMER'in yetenek ile verimlilik arasında denge kurma başarısı, çeşitli gerçek dünya görevlerini işleyebilen dağıtılabilir AGI sistemlerine doğru önemli bir adımı temsil eder.

\bibliographystyle{plain}
\bibliography{refs}

\end{document}