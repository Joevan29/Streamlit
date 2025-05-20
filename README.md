<h1 align="center">🚀 Sentiment Analysis Gojek App - Streamlit</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-%E2%9D%A4-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/SVM-Model-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" />
</p>

---

## 🧠 Deskripsi

Proyek ini merupakan aplikasi **analisis sentimen** terhadap ulasan pengguna aplikasi **Gojek**, dibuat dengan **Python** dan **Streamlit**. Model klasifikasi yang digunakan adalah **Support Vector Machine (SVM)**, dan aplikasi ini dirancang untuk memberikan visualisasi interaktif serta prediksi real-time berdasarkan input pengguna.

---

## 🎯 Fitur

- Upload file CSV berisi ulasan untuk analisis batch
- Prediksi sentimen dari input teks secara langsung
- Visualisasi WordCloud per kategori sentimen
- Statistik sebaran data: positif, netral, negatif
- Evaluasi performa model klasifikasi

---

## 📸 Tampilan Aplikasi

> Ganti tautan gambar di bawah dengan screenshot asli dari aplikasi

<p align="center">
  <img src="https://via.placeholder.com/800x400.png?text=Screenshot+Aplikasi+Streamlit" alt="Tampilan Aplikasi Streamlit" />
</p>

---

## ⚙️ Teknologi

- Python 3.10
- Streamlit
- Scikit-learn
- Sastrawi (NLP Bahasa Indonesia)
- Pandas, Matplotlib, Seaborn
- WordCloud
- google-play-scraper

---

## 📁 Struktur Repositori
📦 Streamlit
├── sentiment_gojek_app.py # Aplikasi utama Streamlit
├── README.md # Dokumentasi proyek
├── requirements.txt # Daftar dependensi


---

## 🚀 Cara Menjalankan

1. **Clone repositori ini**
```bash
git clone https://github.com/Joevan29/Streamlit.git
cd Streamlit
```
2. **Buat virtual environment (opsional)**
```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

3. **Install dependensi**
```bash
pip install -r requirements.txt
```
4. **Jalankan aplikasi**
```bash
streamlit run sentiment_gojek_app.py
```

**📊 Evaluasi Model**
| Metrik    | Nilai |
| --------- | ----- |
| Akurasi   | 85.2% |
| Precision | 83.4% |
| Recall    | 84.0% |
| F1-Score  | 83.7% |

    Catatan: Nilai di atas berdasarkan dataset 10.000 ulasan Gojek dari Google Play Store

    
**👤 Kontak**
Joevan Pramana Achmad

📧 joevanpan@outlook.com
🔗 https://www.linkedin.com/in/jvnprmnachmd/

Aplikasi ini dikembangkan sebagai bagian dari penelitian untuk mendukung pengambilan keputusan berbasis analisis sentimen terhadap layanan digital transportasi dan keuangan di Indonesia.
---
