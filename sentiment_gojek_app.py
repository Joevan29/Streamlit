import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import nltk
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from io import StringIO

# Konfigurasi halaman Streamlit - HARUS MENJADI PERINTAH STREAMLIT PERTAMA
st.set_page_config(page_title="Analisis Sentimen Gojek", layout="wide")

# Inisialisasi variabel global
df = None
svm_model = None
vectorizer = None
models_loaded = False

# Download NLTK resources jika belum ada
@st.cache_resource
def download_nltk_resources():
    try:
        stopwords.words('indonesian')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
    return "NLTK resources loaded"

# Memuat resources NLTK
download_status = download_nltk_resources()

# Fungsi Preprocessing teks yang lebih robust
def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Inisialisasi stemmer dan stopwords
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('indonesian'))
        
        # Langkah-langkah preprocessing
        text = text.lower()  # Mengubah teks menjadi lowercase
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Menghapus URL
        text = re.sub(r'@\w+', '', text)  # Menghapus username
        text = re.sub(r'#\w+', '', text)  # Menghapus hashtag
        text = re.sub(r'\d+', '', text)  # Menghapus angka
        text = text.translate(str.maketrans('', '', string.punctuation))  # Menghapus tanda baca
        text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
        
        # Menghapus stopwords
        words = [word for word in text.split() if word not in stop_words]
        
        # Stemming
        stemmed_words = [stemmer.stem(word) for word in words]
        
        # Menggabungkan kembali kata-kata
        processed_text = " ".join(stemmed_words)
        
        return processed_text
    except Exception as e:
        st.error(f"Error dalam preprocessing: {e}")
        return text

# Fungsi untuk memuat model dan vectorizer
@st.cache_resource
def load_models():
    try:
        svm_model = joblib.load('svm_model_gojek.joblib')
        vectorizer = joblib.load('vectorizer_gojek.joblib')
        return svm_model, vectorizer, True
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None, False
    except Exception as e:
        st.error(f"Unexpected error when loading models: {e}")
        return None, None, False

# Fungsi untuk memuat data
@st.cache_data
def load_data(file):
    try:
        if isinstance(file, str):
            # Jika file adalah path string
            data = pd.read_csv(file)
        else:
            # Jika file adalah file yang diupload
            data = pd.read_csv(file)
            
        # Mengisi nilai NaN
        data = data.fillna('')
        
        return data, True
    except Exception as e:
        st.error(f"Error saat memuat data: {e}")
        return pd.DataFrame(), False

# Fungsi untuk melatih model baru
def train_model(X_train, y_train):
    try:
        # Membuat vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        
        # Melatih model SVM
        svm_model = SVC(kernel='linear', probability=True)
        svm_model.fit(X_train_vec, y_train)
        
        return svm_model, vectorizer, True
    except Exception as e:
        st.error(f"Error saat melatih model: {e}")
        return None, None, False

# Fungsi untuk memprediksi sentimen
def predict_sentiment(text, model, vectorizer):
    try:
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        
        sentiment_mapping = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        sentiment = sentiment_mapping.get(prediction, 'Unknown')
        
        return sentiment, processed_text, probabilities
    except Exception as e:
        st.error(f"Error saat memprediksi: {e}")
        return "Error", "", []

# UI utama
st.title("🚖 Analisis Sentimen Ulasan Aplikasi Gojek")

# Sidebar Navigasi
st.sidebar.title("Navigasi")
options = st.sidebar.radio("Pilih Halaman", [
    "📥 Data Management", 
    "🧠 Model Training & Evaluation", 
    "🔍 Prediksi Sentimen", 
    "📊 Visualisasi Data",
    "🔄 Batch Processing"
])

# Halaman Data Management
if options == "📥 Data Management":
    st.header("📥 Data Management")
    
    # Tambahkan opsi upload file
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload CSV file ulasan", type=['csv'])
    
    if uploaded_file is not None:
        # Muat data dari file yang diupload
        df, success = load_data(uploaded_file)
        
        if success and not df.empty:
            # Simpan ke session state
            st.session_state['df'] = df
            st.success(f"Data berhasil dimuat! ({len(df)} baris)")
            
            # Lihat data
            st.subheader("Preview Data")
            st.dataframe(df.head())
            
            # Cek kolom yang ada
            st.subheader("Struktur Data")
            st.write(f"Kolom dalam dataset: {', '.join(df.columns)}")
            
            # Tambahkan kolom yang diperlukan jika belum ada
            cols_to_add = []
            if 'review' not in df.columns and 'text' not in df.columns and 'review_text' not in df.columns:
                st.warning("Tidak ada kolom ulasan terdeteksi. Silakan tentukan kolom ulasan.")
                review_col = st.selectbox("Pilih kolom yang berisi ulasan:", df.columns)
                if review_col:
                    df.rename(columns={review_col: 'review'}, inplace=True)
                    st.success(f"Kolom {review_col} diubah menjadi 'review'")
            
            # Preprocessing data
            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    if 'review' in df.columns:
                        df['processed_text'] = df['review'].apply(preprocess_text)
                    elif 'text' in df.columns:
                        df['processed_text'] = df['text'].apply(preprocess_text)
                    elif 'review_text' in df.columns:
                        df['processed_text'] = df['review_text'].apply(preprocess_text)
                    else:
                        st.error("Tidak dapat menemukan kolom ulasan.")
                    
                    # Update session state
                    st.session_state['df'] = df
                    st.success("Preprocessing selesai!")
                    st.dataframe(df[['review' if 'review' in df.columns 
                                    else 'text' if 'text' in df.columns 
                                    else 'review_text', 'processed_text']].head())
            
            # Lihat distribusi sentimen jika kolom sentiment ada
            if 'sentiment' in df.columns or 'sentiment_text' in df.columns or 'label' in df.columns:
                st.subheader("Distribusi Sentimen")
                
                sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'sentiment_text' if 'sentiment_text' in df.columns else 'label'
                
                # Hitung distribusi
                sentiment_counts = df[sentiment_col].value_counts()
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_counts.plot(kind='bar', ax=ax, color=['firebrick', 'goldenrod', 'forestgreen'])
                plt.title("Distribusi Sentimen")
                plt.xlabel("Sentimen")
                plt.ylabel("Jumlah")
                plt.xticks(rotation=0)
                st.pyplot(fig)
                
                # Tampilkan statistik
                st.write("Statistik Sentimen:")
                st.write(sentiment_counts)
            
            # Opsi simpan data yang telah dipreprocess
            if 'processed_text' in df.columns:
                if st.button("Simpan Data Preprocessed"):
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Data Preprocessed",
                        data=csv,
                        file_name="gojek_reviews_preprocessed.csv",
                        mime="text/csv"
                    )
    else:
        # Coba muat file default jika tidak ada upload
        if os.path.exists('gojek_reviews_preprocessed.csv'):
            if st.button("Muat Data Default"):
                df, success = load_data('gojek_reviews_preprocessed.csv')
                if success:
                    st.session_state['df'] = df
                    st.success(f"Data default berhasil dimuat! ({len(df)} baris)")
                    st.dataframe(df.head())
        else:
            st.info("Silakan upload file CSV atau masukkan ulasan secara manual.")
            
            # Opsi untuk memasukkan data secara manual
            st.subheader("Atau Masukkan Data Secara Manual")
            manual_data = st.text_area("Masukkan ulasan (satu ulasan per baris):")
            sentiment_options = st.multiselect("Sentimen yang tersedia:", ["Positif", "Netral", "Negatif"], default=["Positif", "Netral", "Negatif"])
            
            if manual_data and st.button("Buat Dataset"):
                reviews = manual_data.strip().split('\n')
                manual_df = pd.DataFrame({
                    'review': reviews,
                    'processed_text': [preprocess_text(review) for review in reviews]
                })
                st.session_state['df'] = manual_df
                st.success(f"Dataset berhasil dibuat dengan {len(manual_df)} ulasan")
                st.dataframe(manual_df)

# Halaman Model Training & Evaluation
elif options == "🧠 Model Training & Evaluation":
    st.header("🧠 Model Training & Evaluation")
    
    # Cek apakah data tersedia
    if 'df' not in st.session_state or st.session_state['df'] is None or st.session_state['df'].empty:
        st.warning("Data belum dimuat. Silakan kembali ke halaman 'Data Management'.")
    else:
        df = st.session_state['df']
        
        # Load model yang sudah ada
        st.subheader("Model Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Muat Model Yang Sudah Ada"):
                svm_model, vectorizer, models_loaded = load_models()
                if models_loaded:
                    st.session_state['svm_model'] = svm_model
                    st.session_state['vectorizer'] = vectorizer
                    st.session_state['models_loaded'] = models_loaded
                    st.success("Model dan vectorizer berhasil dimuat!")
                else:
                    st.error("Gagal memuat model. Model atau vectorizer tidak ditemukan.")
        
        with col2:
            # Latih model baru
            if st.button("Latih Model Baru"):
                # Cek kolom yang diperlukan
                text_col = None
                label_col = None
                
                # Mencari kolom teks
                if 'processed_text' in df.columns:
                    text_col = 'processed_text'
                elif 'stemmed' in df.columns:
                    text_col = 'stemmed'
                elif 'review' in df.columns:
                    text_col = 'review'
                
                # Mencari kolom label/sentimen
                if 'sentiment' in df.columns:
                    label_col = 'sentiment'
                elif 'sentiment_text' in df.columns:
                    label_col = 'sentiment_text'
                elif 'label' in df.columns:
                    label_col = 'label'
                
                if text_col is None or label_col is None:
                    st.error("Dataset tidak memiliki kolom teks atau label yang diperlukan.")
                else:
                    # Split data
                    st.write(f"Menggunakan '{text_col}' sebagai fitur dan '{label_col}' sebagai target")
                    X = df[text_col].fillna('')
                    y = df[label_col]
                    
                    # Convert sentiment ke numeric jika perlu
                    if y.dtype == 'object':
                        # Map string labels ke numerik
                        label_mapping = {}
                        unique_labels = y.unique()
                        
                        if set(unique_labels) == set(['Positif', 'Netral', 'Negatif']):
                            label_mapping = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
                        elif set(unique_labels) == set(['positive', 'neutral', 'negative']):
                            label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
                        else:
                            # Membuat mapping numerik generik
                            for i, label in enumerate(unique_labels):
                                label_mapping[label] = i
                        
                        y = y.map(label_mapping)
                        st.write(f"Label mapping: {label_mapping}")
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    st.write(f"Training dengan {len(X_train)} sampel, Testing dengan {len(X_test)} sampel")
                    
                    # Train model
                    with st.spinner("Melatih model..."):
                        svm_model, vectorizer, training_success = train_model(X_train, y_train)
                        
                        if training_success:
                            # Evaluasi model
                            X_test_vec = vectorizer.transform(X_test)
                            y_pred = svm_model.predict(X_test_vec)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                y_test, y_pred, average='weighted'
                            )
                            
                            # Simpan model ke session state
                            st.session_state['svm_model'] = svm_model
                            st.session_state['vectorizer'] = vectorizer
                            st.session_state['models_loaded'] = True
                            
                            # Tampilkan metrik
                            st.success("Model berhasil dilatih!")
                            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                            metrics_col1.metric("Accuracy", f"{accuracy:.2%}")
                            metrics_col2.metric("Precision", f"{precision:.2%}")
                            metrics_col3.metric("Recall", f"{recall:.2%}")
                            metrics_col4.metric("F1 Score", f"{f1:.2%}")
                            
                            # Opsi simpan model
                            if st.button("Simpan Model"):
                                joblib.dump(svm_model, 'svm_model_gojek.joblib')
                                joblib.dump(vectorizer, 'vectorizer_gojek.joblib')
                                st.success("Model dan vectorizer berhasil disimpan!")
                        else:
                            st.error("Gagal melatih model.")
        
        # Evaluasi model jika sudah dimuat atau dilatih
        if 'models_loaded' in st.session_state and st.session_state['models_loaded']:
            st.subheader("Evaluasi Model")
            
            # Ambil model dan vectorizer dari session state
            svm_model = st.session_state['svm_model']
            vectorizer = st.session_state['vectorizer']``
            
            # Cari kolom yang diperlukan
            text_col = None
            label_col = None
            
            if 'processed_text' in df.columns:
                text_col = 'processed_text'
            elif 'stemmed' in df.columns:
                text_col = 'stemmed'
            elif 'review' in df.columns:
                text_col = 'review'
            
            if 'sentiment' in df.columns:
                label_col = 'sentiment'
            elif 'sentiment_text' in df.columns:
                label_col = 'sentiment_text'
            elif 'label' in df.columns:
                label_col = 'label' 
            
            if text_col is not None and label_col is not None:
                # Evaluasi model pada dataset
                X = df[text_col].fillna('')
                y = df[label_col]
                
                # Convert y ke numeric jika perlu
                if y.dtype == 'object':
                    # Map string labels ke numerik
                    label_mapping = {}
                    unique_labels = y.unique()
                    
                    if set(unique_labels) == set(['Positif', 'Netral', 'Negatif']):
                        label_mapping = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
                    elif set(unique_labels) == set(['positive', 'neutral', 'negative']):
                        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
                    else:
                        # Membuat mapping numerik generik
                        for i, label in enumerate(unique_labels):
                            label_mapping[label] = i
                    
                    y_numeric = y.map(label_mapping)
                else:
                    y_numeric = y
                
                # Reverse mapping (numeric ke string)
                reverse_mapping = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
                
                # Transform dan prediksi
                X_vec = vectorizer.transform(X)
                y_pred_numeric = svm_model.predict(X_vec)
                
                # Convert prediksi ke label string jika diperlukan
                y_pred = [reverse_mapping.get(pred, str(pred)) for pred in y_pred_numeric]
                
                # Evaluasi
                if y.dtype == 'object':
                    # Jika target adalah string, bandingkan dengan y_pred yang sudah dikonversi ke string
                    accuracy = sum(y.values == y_pred) / len(y)
                else:
                    # Jika target numerik, bandingkan dengan y_pred_numeric
                    accuracy = accuracy_score(y, y_pred_numeric)
                
                # Tampilkan hasil evaluasi
                st.metric("Akurasi Model", f"{accuracy:.2%}")
                
                # Confusion Matrix
                st.write("### Confusion Matrix")
                
                if y.dtype == 'object':
                    # Jika label berupa string
                    cm = confusion_matrix(y, y_pred)
                    labels = sorted(y.unique())
                else:
                    # Jika label numerik
                    cm = confusion_matrix(y, y_pred_numeric)
                    labels = [reverse_mapping.get(i, str(i)) for i in range(len(reverse_mapping))]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.xlabel("Prediksi")
                plt.ylabel("Aktual")
                st.pyplot(fig)
                
                # Classification Report
                st.write("### Classification Report")
                if y.dtype == 'object':
                    report = classification_report(y, y_pred, output_dict=True)
                else:
                    report = classification_report(y, y_pred_numeric, output_dict=True)
                
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # Contoh kesalahan klasifikasi
                if st.checkbox("Lihat Contoh Kesalahan Klasifikasi"):
                    review_col = 'review' if 'review' in df.columns else text_col
                    
                    # Buat DataFrame hasil
                    results_df = df.copy()
                    results_df['predicted'] = y_pred
                    
                    # Filter ulasan yang salah klasifikasi
                    misclassified = results_df[results_df[label_col] != results_df['predicted']]
                    
                    if len(misclassified) > 0:
                        st.write(f"Jumlah ulasan yang salah klasifikasi: {len(misclassified)} dari {len(df)} ({len(misclassified)/len(df):.2%})")
                        st.dataframe(misclassified[[review_col, label_col, 'predicted']].head(10))
                    else:
                        st.success("Tidak ada kesalahan klasifikasi!")
            else:
                st.warning("Dataset tidak memiliki kolom teks atau label yang diperlukan untuk evaluasi.")

# Halaman Prediksi Sentimen
elif options == "🔍 Prediksi Sentimen":
    st.header("🔍 Prediksi Sentimen")
    
    # Cek apakah model tersedia
    if 'models_loaded' in st.session_state and st.session_state['models_loaded']:
        svm_model = st.session_state['svm_model']
        vectorizer = st.session_state['vectorizer']
        
        st.subheader("Prediksi Ulasan Tunggal")
        input_text = st.text_area("Masukkan ulasan untuk diprediksi:", height=150)
        
        if st.button("Analisis Sentimen"):
            if input_text:
                with st.spinner("Menganalisis..."):
                    sentiment, processed_text, probabilities = predict_sentiment(input_text, svm_model, vectorizer)
                    
                    # Tampilkan hasil dengan visualisasi yang lebih baik
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if sentiment == 'Positif':
                            st.success("Sentimen: POSITIF")
                            st.markdown("### 😀 Positif")
                        elif sentiment == 'Netral':
                            st.info("Sentimen: NETRAL")
                            st.markdown("### 😐 Netral")
                        elif sentiment == 'Negatif':
                            st.error("Sentimen: NEGATIF")
                            st.markdown("### 😠 Negatif")
                        else:
                            st.warning(f"Sentimen: {sentiment}")
                    
                    with col2:
                        # Tampilkan probabilitas dalam bentuk bar chart
                        st.write("Distribusi Probabilitas:")
                        probs_df = pd.DataFrame({
                            'Sentimen': ['Negatif', 'Netral', 'Positif'],
                            'Probabilitas': probabilities
                        })
                        
                        fig, ax = plt.subplots(figsize=(8, 3))
                        sns.barplot(data=probs_df, x='Sentimen', y='Probabilitas', 
                                    palette=['firebrick', 'goldenrod', 'forestgreen'], ax=ax)
                        plt.ylabel('Probabilitas')
                        plt.xlabel('Sentimen')
                        plt.ylim(0, 1)
                        st.pyplot(fig)
                    
                    # Tampilkan hasil preprocessing
                    with st.expander("Detail Analisis"):
                        st.write("**Teks Asli:**")
                        st.write(input_text)
                        st.write("**Setelah preprocessing:**")
                        st.write(processed_text)
                        
                        # Highlight kata-kata signifikan
                        if processed_text:
                            st.write("**Kata-kata Signifikan:**")
                            
                            # Dapatkan feature names dari vectorizer
                            try:
                                feature_names = vectorizer.get_feature_names_out()
                                
                                # Dapatkan koefisien dari model SVM
                                coefs = svm_model.coef_
                                
                                # Ambil kata-kata dari teks yang diproses
                                words = processed_text.split()
                                word_importance = {}
                                
                                # Hitung skor kepentingan untuk setiap kata
                                for word in set(words):
                                    try:
                                        idx = list(feature_names).index(word)
                                        # Ambil koefisien positif, negatif dan netral
                                        scores = [coefs[0][idx], coefs[1][idx] if coefs.shape[0] > 1 else 0, 
                                                 coefs[2][idx] if coefs.shape[0] > 2 else 0]
                                        # Ambil koefisien dengan magnitude tertinggi
                                        max_score = max(scores, key=abs)
                                        word_importance[word] = max_score
                                    except (ValueError, IndexError):
                                        # Kata tidak ada dalam vocabulary
                                        continue
                                
                                # Urutkan kata berdasarkan pentingnya
                                sorted_words = sorted(word_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                                
                                # Tampilkan kata-kata penting
                                important_words = []
                                for word, score in sorted_words[:10]:  # Ambil 10 kata teratas
                                    if score > 0:
                                        color = "green"
                                        label = "positif"
                                    else:
                                        color = "red"
                                        label = "negatif"
                                    
                                    important_words.append(f"<span style='color:{color};font-weight:bold'>{word}</span> ({label}, skor: {score:.4f})")
                                
                                if important_words:
                                    st.markdown(", ".join(important_words), unsafe_allow_html=True)
                                else:
                                    st.write("Tidak dapat menentukan kata-kata signifikan.")
                            except Exception as e:
                                st.write(f"Tidak dapat menampilkan kata-kata signifikan: {e}")
            else:
                st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        st.warning("Model belum dimuat. Silakan muat atau latih model terlebih dahulu di halaman 'Model Training & Evaluation'.")
        
        # Opsi untuk muat model langsung
        if st.button("Muat Model Sekarang"):
            svm_model, vectorizer, models_loaded = load_models()
            if models_loaded:
                st.session_state['svm_model'] = svm_model
                st.session_state['vectorizer'] = vectorizer
                st.session_state['models_loaded'] = models_loaded
                st.success("Model berhasil dimuat! Anda dapat melakukan prediksi sekarang.")
                st.experimental_rerun()
            else:
                st.error("Gagal memuat model. Pastikan file model tersedia.")

# Halaman Visualisasi Data
elif options == "📊 Visualisasi Data":
    st.header("📊 Visualisasi Data")
    
    # Cek apakah data tersedia
    if 'df' not in st.session_state or st.session_state['df'] is None or st.session_state['df'].empty:
        st.warning("Data belum dimuat. Silakan kembali ke halaman 'Data Management'.")
    else:
        df = st.session_state['df']
        
        # Pilihan visualisasi
        viz_option = st.selectbox(
            "Pilih Visualisasi",
            ["Wordcloud", "Distribusi Sentimen", "Panjang Ulasan", "Kata Frekuensi Tinggi"]
        )
        
        # Tetapkan kolom yang akan digunakan
        text_col = None
        label_col = None
        
        # Identifikasi kolom teks
        if 'processed_text' in df.columns:
            text_col = 'processed_text'
        elif 'stemmed' in df.columns:
            text_col = 'stemmed'
        elif 'review' in df.columns:
            text_col = 'review'
        
        # Identifikasi kolom sentimen
        if 'sentiment' in df.columns:
            label_col = 'sentiment'
        elif 'sentiment_text' in df.columns:
            label_col = 'sentiment_text'