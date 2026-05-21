# 🎥 TrustTube — AI-Powered YouTube Video & Trust Analyzer
> **Yapay Zeka Destekli YouTube Güvenilirlik ve Performans Analizörü**

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![UI Framework](https://img.shields.io/badge/UI-CustomTkinter-darkblue.svg?style=for-the-badge&logo=python&logoColor=white)](https://github.com/TomSchimansky/CustomTkinter)
[![AI Engine](https://img.shields.io/badge/AI%20Engine-HuggingFace%20RoBERTa-orange.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
[![Deep Learning](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## 🇹🇷 Türkçe Açıklama

**TrustTube**, aradığınız herhangi bir konudaki en doğru, güvenilir ve popüler YouTube videolarını bulup analiz eden modern bir masaüstü uygulamasıdır. Klasik YouTube arama algoritmasının aksine TrustTube; izlenme sayıları, etkileşim oranları ve en önemlisi **yorumların yapay zeka tabanlı duygu analizi (Sentiment Analysis)** sonuçlarını birleştirerek videoları akıllı bir şekilde puanlar ve size en kaliteli **ilk 5** videoyu önerir.

### 🌟 Öne Çıkan Özellikler

- **Modern & Şık Arayüz:** CustomTkinter ile oluşturulmuş, karanlık mod (Dark Mode) destekli, göz yormayan modern tasarım.
- **YouTube Veri API Entegrasyonu:** Gerçek zamanlı arama sonuçları, izlenme sayıları, beğeni oranları ve tarihler.
- **Yapay Zeka Yorum Analizi:** Hugging Face `transformers` altyapısı ve `cardiffnlp/twitter-roberta-base-sentiment-latest` RoBERTa modeliyle en popüler 15 yorumun duygu durumunu (Olumlu, Nötr, Olumsuz) derinlemesine analiz eder.
- **Shorts Filtreleme:** Sadece kaliteli ve uzun içeriklere odaklanmak amacıyla 60 saniyeden kısa olan YouTube Shorts videolarını otomatik olarak eler.
- **Akıllı Skorlama Formülü:** İzlenme hızını, beğeni oranını ve yapay zeka analiz skorunu harmanlayarak her videoya 0-100 arası nihai bir **Güven Puanı (Total Score)** verir.
- **En Beğenilen Yorum Gösterimi:** Her videonun en çok beğeni alan (faydalı) yorumunu doğrudan arayüzde gösterir.
- **Çevrimdışı Model Desteği:** `local_model` klasörü içinde yerel ağırlıklar varsa doğrudan oradan yüklenir; yoksa HuggingFace Hub'dan otomatik indirilir.
- **PyInstaller Uyumlu:** Taşınabilir tek bir `.exe` dosyası olarak paketlenmeye hazır yapı.

---

## 🇬🇧 English Description

**TrustTube** is a sleek desktop application that finds, analyzes, and ranks the most reliable and high-quality YouTube videos for any search query. Unlike standard YouTube search, TrustTube evaluates videos using a custom algorithm that combines view counts, engagement rates, and **AI-powered sentiment analysis of video comments** using state-of-the-art RoBERTa models. It filters out low-effort videos and presents the **Top 5** recommended videos with detailed stats.

### 🌟 Key Features

- **Modern & Premium GUI:** Built with CustomTkinter, featuring a high-fidelity dark-themed user interface.
- **Real-Time YouTube API Integration:** Fetches search results, view counts, likes, and published dates.
- **AI Sentiment Analysis:** Uses Hugging Face `transformers` and a RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to perform sentiment analysis (Positive, Neutral, Negative) on the top 15 comments.
- **Shorts Filtering:** Automatically excludes YouTube Shorts (videos < 60 seconds) to ensure long-form, comprehensive content.
- **Smart Scoring Formula:** Harmonizes view velocity, like/engagement rates, and AI sentiment scores to assign a final **Trust Score (0-100)**.
- **Highlighting Top Comments:** Pinpoints and displays the single most-liked comment for quick context.
- **Offline Model Support:** Automatically loads weights from a local folder (`local_model`) if present, or downloads online weights dynamically.
- **PyInstaller Ready:** Fully compatible with PyInstaller for building single-executable packages (`.exe`).

---

## 📸 Arayüz Önizlemesi / UI Preview

Aşağıda uygulamanın modern ve estetik arayüz tasarımını görebilirsiniz:

![TrustTube Preview](assets/preview.png)

---

## 🧠 Akıllı Skorlama Algoritması / Smart Scoring Formula

TrustTube, videoları sadece izlenme sayılarına göre değil, topluluk geri bildirimlerine göre de sıralar. Sıralama formülü şu şekildedir:

### 🇹🇷 Formül Detayları
- **Duygu Analizi Skoru (Ağırlık: %60):** Yorumların yapay zeka analizinden elde edilen olumluluk oranı.
- **Normalize Günlük İzlenme Hızı (Ağırlık: %20):** $\text{Günlük İzlenme} = \frac{\text{Toplam İzlenme}}{\text{Video Yaşı (Gün)}}$. En yüksek izlenmeye sahip videoya göre 0-100 arası normalize edilir.
- **Normalize Etkileşim Oranı (Ağırlık: %20):** $\text{Etkileşim Oranı} = \frac{\text{Beğeni}}{\text{İzlenme}}$. En yüksek etkileşim alan videoya göre 0-100 arası normalize edilir.

$$\text{Toplam Skor} = (\text{Duygu Skoru} \times 0.60) + (\text{Normalize Günlük İzlenme} \times 0.20) + (\text{Normalize Etkileşim} \times 0.20)$$

### 🇬🇧 Formula Details
- **AI Sentiment Score (Weight: 60%):** The positivity ratio derived from comment AI processing.
- **Normalized Daily Velocity (Weight: 20%):** Calculated as $\frac{\text{Total Views}}{\text{Video Age (Days)}}$ and normalized relative to the top performer.
- **Normalized Engagement Rate (Weight: 20%):** Calculated as $\frac{\text{Likes}}{\text{Views}}$ and normalized relative to the top performer.

---

## 🚀 Kurulum ve Çalıştırma / Installation & Execution

### 📋 Gereksinimler / Prerequisites
- **Python 3.8** veya daha üzeri bir sürüm.
- **YouTube Data API v3 Anahtarı (API Key)**.

### 📥 Adım 1: Depoyu Klonlayın / Clone the Repository
```bash
git clone https://github.com/KULLANICI_ADINIZ/trust-tube.git
cd trust-tube
```

### 📦 Adım 2: Gerekli Paketleri Yükleyin / Install Dependencies
Bir sanal ortam (virtual environment) oluşturup paketleri yüklemeniz önerilir:

```bash
# Sanal ortam oluşturma
python -m venv venv

# Sanal ortamı aktifleştirme (Windows)
.\venv\Scripts\activate

# Sanal ortamı aktifleştirme (macOS/Linux)
source venv/bin/activate

# Bağımlılıkları yükleme
pip install -r requirements.txt
```

> [!NOTE]
> `torch` ve `transformers` kütüphanelerinin boyutu büyüktür. Yükleme işlemi internet hızınıza bağlı olarak birkaç dakika sürebilir.

### 🔑 Adım 3: YouTube API Anahtarınızı Tanımlayın / Setup YouTube API Key
Uygulamayı çalıştırmadan önce `TrustTube (1).py` dosyasını açarak 29. satırdaki `API_KEY` değerine kendi YouTube API anahtarınızı yapıştırın:

```python
# TrustTube (1).py - Satır 29
API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"
```

> [!WARNING]
> **Güvenlik Uyarısı:** API anahtarınızı doğrudan GitHub gibi genel (public) depolarda paylaşmamaya dikkat edin. Kodu GitHub'a yüklemeden önce API anahtarınızı silmeniz veya çevre değişkeni (environment variable) kullanmanız şiddetle tavsiye edilir.

### 💻 Adım 4: Uygulamayı Başlatın / Run the Application
```bash
python "TrustTube (1).py"
```

Uygulama açıldığında ilk seferde yapay zeka modelini HuggingFace üzerinden indirecektir. Bu işlem bir defaya mahsus olup, sonraki açılışlarda hazır olan yerel model kullanılacaktır. Model hazır olduğunda durum göstergesi yeşile dönecek ve **"ANALYZE"** butonu aktifleşecektir.

---

## 📦 PyInstaller ile Exe Paketleme / Building .exe with PyInstaller

Uygulamayı herhangi bir Python kurulumuna ihtiyaç duymadan çalıştırılabilecek tek bir `.exe` haline getirmek için aşağıdaki adımları takip edebilirsiniz:

1. **PyInstaller kütüphanesini yükleyin:**
   ```bash
   pip install pyinstaller
   ```

2. **Aşağıdaki komutu kullanarak projeyi derleyin:**
   ```bash
   pyinstaller --noconfirm --onedir --windowed --add-data "app_icon.ico;." "TrustTube (1).py"
   ```

3. Derleme işlemi tamamlandıktan sonra oluşturulan `dist/` klasörü altından uygulamanızı doğrudan çalıştırabilirsiniz.

---

## 🛠 Kullanılan Teknolojiler / Technologies Used

- **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter):** Modern, şık ve koyu mod destekli masaüstü kullanıcı arayüzü tasarımı.
- **[Hugging Face Transformers](https://huggingface.co/docs/transformers):** Yorum duygu analizi için gelişmiş doğal dil işleme (NLP) altyapısı.
- **[CardiffNLP Twitter RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest):** Sosyal medya yorumlarını yüksek doğrulukla analiz etmek için özel olarak eğitilmiş dil modeli.
- **[PyTorch](https://pytorch.org/):** Yapay zeka modelinin arka planda hızlı ve performanslı çalışmasını sağlayan derin öğrenme kütüphanesi.
- **[Google API Python Client](https://github.com/googleapis/google-api-python-client):** YouTube Data API v3 üzerinden canlı video verileri ve yorum akışı çekmek için resmi kütüphane.


