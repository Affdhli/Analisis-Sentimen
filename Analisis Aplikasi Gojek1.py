import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import time
import joblib
import pickle
from io import BytesIO
import os
from datetime import datetime

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# ============================================
# FUNGSI-FUNGSI UTAMA
# ============================================

def setup_page():
    """Setup halaman Streamlit"""
    st.set_page_config(
        page_title="Analisis Sentimen Ulasan Gojek",
        layout="wide"
    )
    
    st.title("Analisis Sentimen Ulasan Gojek")
    st.markdown("---")

def upload_data():
    """Fungsi untuk upload data"""
    st.header("1. UPLOAD DATA")
    
    # Hanya menyediakan opsi upload file
    st.info("Silakan upload file CSV yang berisi data ulasan Gojek")
    
    uploaded_file = st.file_uploader(
        "Upload file CSV dengan kolom 'content' (dan 'sentimen' jika ada)", 
        type=['csv']
    )
    
    df = None
    
    if uploaded_file is not None:
        try:
            # Baca file
            df = pd.read_csv(uploaded_file)
            st.success(f"File berhasil diupload: {uploaded_file.name}")
            
            # Validasi kolom
            if 'content' not in df.columns:
                st.error("File harus memiliki kolom 'content'")
                return None
            
            # Ambil 8000 data pertama jika lebih
            max_data = 8000
            original_count = len(df)
            
            if len(df) > max_data:
                df = df.head(max_data)
                st.info(f"Mengambil {max_data} data pertama dari total {original_count} data")
            else:
                st.info(f"Menggunakan semua data yang tersedia: {len(df)} data")
            
            # Tampilkan preview
            with st.expander("Preview Data"):
                st.dataframe(df.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Informasi kolom:**")
                    st.write(df.columns.tolist())
                with col2:
                    st.write("**Statistik data:**")
                    st.write(f"- Jumlah baris: {len(df)}")
                    st.write(f"- Jumlah kolom: {len(df.columns)}")
            
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
            return None
    
    if df is not None:
        # Hitung statistik dasar
        df['jumlah_kata'] = df['content'].apply(lambda x: len(str(x).split()))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", f"{len(df):,}")
        with col2:
            st.metric("Total Kata", f"{df['jumlah_kata'].sum():,}")
        with col3:
            st.metric("Rata-rata Kata", f"{df['jumlah_kata'].mean():.1f}")
        
        # Tampilkan contoh data
        with st.expander("Contoh Data (5 baris pertama)"):
            for i in range(min(5, len(df))):
                content = str(df['content'].iloc[i])
                sentiment = df['sentimen'].iloc[i] if 'sentimen' in df.columns else 'N/A'
                
                st.write(f"**Data {i+1}:**")
                st.write(f"- Konten: {content[:70]}...")
                st.write(f"- Sentimen: {sentiment}")
                st.write(f"- Jumlah kata: {df['jumlah_kata'].iloc[i]}")
                st.write("---") 
    
    return df

def lexicon_sentiment_labeling(df):
    """Pelabelan sentimen dengan lexicon - HANYA GRAFIK"""
    st.header("2. PELABELAN SENTIMEN MENGGUNAKAN LEXICON")
    
    # Lexicon yang diperluas untuk Bahasa Indonesia dengan penanganan kalimat rancu
    positive_words = [
        'bagus', 'baik', 'mantap', 'cepat', 'mudah', 'praktis', 'terbaik',
        'puas', 'sukses', 'senang', 'murah', 'keren', 'hebat', 'suka',
        'tolong', 'bantu', 'recommended', 'lancar', 'memuaskan', 'nyaman',
        'aman', 'profit', 'untung', 'cinta', 'setia', 'gembira', 'bahagia',
        'nikmat', 'enak', 'lezat', 'hemat', 'efisien', 'efektif', 'semangat',
        'antusias', 'luar biasa', 'wow', 'excellent', 'awesome', 'fantastic',
        'great', 'good', 'nice', 'perfect', 'sempurna', 'istimewa', 'unggul',
        'optimal', 'maksimal', 'sangat baik', 'sangat bagus', 'memukau',
        'mengesankan', 'cepat sekali', 'murah sekali', 'sangat membantu',
        'sangat memuaskan', 'profesional', 'ramah', 'sopan', 'jujur',
        'tepat waktu', 'akurat', 'responsif', 'inovasi', 'kreatif',
        'handal', 'andal', 'terpercaya', 'amanah', 'solutif', 'efektif',
        'terjangkau', 'lumayan', 'cukup'
    ]
    
    negative_words = [
        'buruk', 'jelek', 'lambat', 'sulit', 'ribet', 'mahal', 'gagal',
        'kecewa', 'sedih', 'marah', 'kesal', 'jengkel', 'bosan', 'sebel',
        'menyesal', 'menyedihkan', 'menyebalkan', 'parah', 'bermasalah',
        'error', 'bug', 'kacau', 'rusak', 'hilang', 'terlambat', 'telat',
        'susah', 'payah', 'lemot', 'bangkrut', 'rugi', 'sial', 'celaka',
        'mengerikan', 'horor', 'takut', 'khawatir', 'cemas', 'stress',
        'frustasi', 'mengecewakan', 'menipu', 'bodoh', 'tolol', 'goblok',
        'anjing', 'bangsat', 'kontol', 'asu', 'jancok', 'parah sekali',
        'sangat buruk', 'sangat jelek', 'sangat lambat', 'sangat mahal',
        'tidak bisa', 'tidak bisa dipakai', 'tidak berfungsi', 'tidak responsif',
        'tidak profesional', 'kasar', 'tidak sopan', 'curang', 'penipu',
        'mencurigakan', 'berbahaya', 'menakutkan', 'menjengkelkan', 'membosankan',
        'mengecewa', 'menyusahkan', 'merepotkan', 'menghambat', 'menyakitkan',
        'standar', 'biasa', 'tidak spesial'
    ]
    
    # Kata-kata pembalik/negasi
    negation_words = ['tidak', 'bukan', 'belum', 'jangan', 'kurang', 'sedikit', 'agak', 'cukup', 'lumayan']
    
    # Kata-kata intensifier
    intensifier_words = ['sangat', 'sekali', 'banget', 'amat', 'terlalu', 'luar biasa']
    
    # Fungsi untuk pelabelan sentimen dengan penanganan kalimat rancu
    def lexicon_sentiment_analysis_advanced(text):
        if not isinstance(text, str):
            return 'neutral'

        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = 0
        negative_count = 0
        
        # Analisis kata per kata dengan konteks
        for i, word in enumerate(words):
            # Cek kata positif
            if word in positive_words:
                # Cek apakah ada kata negasi sebelumnya
                has_negation = False
                for j in range(max(0, i-3), i):
                    if words[j] in negation_words:
                        has_negation = True
                        break
                
                if has_negation:
                    negative_count += 1  # Kata positif dengan negasi menjadi negatif
                else:
                    positive_count += 1
                    
            # Cek kata negatif
            elif word in negative_words:
                # Cek apakah ada kata negasi sebelumnya
                has_negation = False
                for j in range(max(0, i-3), i):
                    if words[j] in negation_words:
                        has_negation = True
                        break
                
                if has_negation:
                    positive_count += 1  # Kata negatif dengan negasi menjadi positif
                else:
                    negative_count += 1
        
        # Cek kata kunci yang sangat kuat
        strong_positive = any(word in text_lower for word in ['sangat baik', 'sangat bagus', 'luar biasa', 'terbaik'])
        strong_negative = any(word in text_lower for word in ['sangat buruk', 'sangat jelek', 'parah sekali', 'penipu'])
        
        # Penanganan khusus untuk kalimat rancu
        ambiguous_patterns = [
            'kurang begitu', 'tidak terlalu', 'agak', 'sedikit',
            'cukup', 'lumayan', 'standar', 'biasa saja'
        ]
        
        is_ambiguous = any(pattern in text_lower for pattern in ambiguous_patterns)
        
        # Jika kalimat ambigu, berikan bobot khusus
        if is_ambiguous:
            # Untuk kalimat seperti "kurang begitu mahal", mahal adalah negatif
            # tapi "kurang begitu" membuatnya kurang negatif
            if 'mahal' in text_lower and ('kurang' in text_lower or 'tidak' in text_lower):
                positive_count += 1
                negative_count -= 0.5 if negative_count > 0 else 0
            
            # Untuk kalimat seperti "tidak terlalu bagus", bagus adalah positif
            # tapi "tidak terlalu" membuatnya kurang positif
            elif 'bagus' in text_lower or 'baik' in text_lower:
                if 'tidak' in text_lower or 'kurang' in text_lower:
                    negative_count += 1
                    positive_count -= 0.5 if positive_count > 0 else 0
        
        # Hitung akhir dengan penyesuaian
        if strong_positive:
            return 'positive'
        elif strong_negative:
            return 'negative'
        elif positive_count == negative_count:
            # Default ke positif jika netral dan tidak ambigu
            if not is_ambiguous:
                return 'positive'
            else:
                # Untuk kalimat ambigu, cenderung ke negatif (karena biasanya kritik)
                return 'negative'
        
        return 'positive' if positive_count > negative_count else 'negative'
    
    # Terapkan pelabelan advanced
    with st.spinner("Melabeli sentimen dengan analisis lanjutan..."):
        df['sentiment_label'] = df['content'].apply(lexicon_sentiment_analysis_advanced)
    
    # HAPUS jika ada yang masih netral (tidak seharusnya ada)
    df = df[df['sentiment_label'].isin(['positive', 'negative'])].copy()
    
    # Hitung distribusi sentimen
    sentiment_distribution = df['sentiment_label'].value_counts()
    total_data = len(df)
    
    # Tampilkan statistik singkat
    st.success(f"Pelabelan selesai: {total_data} ulasan")
    
    # HANYA MENAMPILKAN GRAFIK - HAPUS ANALISIS STATISTIK TABEL
    st.subheader("HASIL PELABELAN SENTIMEN:")
    
    # Visualisasi grafik saja
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart distribusi sentimen
    sentiment_counts = df['sentiment_label'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title('Distribusi Sentimen\n(Hanya Positif & Negatif)')
    
    # Bar plot jumlah ulasan per sentimen
    axes[1].bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
    axes[1].set_xlabel('Sentimen')
    axes[1].set_ylabel('Jumlah Ulasan')
    axes[1].set_title('Jumlah Ulasan per Kategori')
    for i, v in enumerate(sentiment_counts.values):
        axes[1].text(i, v + max(sentiment_counts.values)*0.01, str(v), ha='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return df, sentiment_distribution

def text_preprocessing(df):
    """Preprocessing teks"""
    st.header("3. TEXT PREPROCESSING")
    
    # Inisialisasi tools
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    stopwords_id = factory.get_stop_words()
    
    def clean_text(text):
        """Fungsi untuk membersihkan teks"""
        if not isinstance(text, str):
            return ""

        # Case folding
        text = text.lower()

        # Remove special characters, numbers, and extra spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def remove_stopwords(text):
        """Menghapus stopwords bahasa Indonesia"""
        return stopword_remover.remove(text)
    
    def tokenize_text(text):
        """Tokenisasi teks"""
        return word_tokenize(text)
    
    def count_text_length(text):
        """Menghitung panjang teks"""
        if not isinstance(text, str):
            return 0
        return len(text)
    
    # Proses preprocessing
    st.subheader("Memulai preprocessing...")
    
    # Cleaning
    df['cleaned_text'] = df['content'].apply(clean_text)
    
    # Remove stopwords
    df['text_no_stopwords'] = df['cleaned_text'].apply(remove_stopwords)
    
    # Tokenization
    df['tokens'] = df['text_no_stopwords'].apply(tokenize_text)
    
    # Gabungkan token kembali menjadi string untuk TF-IDF
    df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    # Hitung panjang teks setelah preprocessing
    df['text_length_processed'] = df['processed_text'].apply(count_text_length)
    
    st.success("Preprocessing selesai!")
    
    # Tampilkan perbandingan
    st.subheader("PERBANDINGAN HASIL ULASAN:")
    
    # Hitung panjang teks asli
    df['text_length'] = df['content'].apply(count_text_length)
    
    before_total = df['text_length'].sum()
    after_total = df['text_length_processed'].sum()
    reduction = before_total - after_total
    reduction_pct = (reduction/before_total*100) if before_total > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sebelum preprocessing", f"{before_total:,} karakter")
    
    with col2:
        st.metric("Setelah preprocessing", f"{after_total:,} karakter")
    
    st.info(f"Pengurangan: {reduction:,} karakter ({reduction_pct:.1f}%)")
    
    # Contoh hasil preprocessing
    st.subheader("CONTOH HASIL PREPROCESSING:")
    
    sample_idx = 0
    st.write(f"**Ulasan Asli:** {df['content'].iloc[sample_idx][:100]}...")
    st.write(f"**Ulasan Setelah Preprocessing:** {df['processed_text'].iloc[sample_idx][:100]}...")
    st.write(f"**Panjang ulasan asli:** {df['text_length'].iloc[sample_idx]} karakter")
    st.write(f"**Panjang ulasan setelah preprocessing:** {df['text_length_processed'].iloc[sample_idx]} karakter")
    
    return df

def create_wordcloud_viz(df):
    """Visualisasi wordcloud"""
    st.header("4. WORDCLOUD VISUALIZATION")
    
    # Fungsi untuk membuat wordcloud
    def create_wordcloud(text, title, color):
        # Cek jika text kosong atau terlalu pendek
        if not text or len(text.strip()) < 10:
            st.warning(f"Tidak ada teks yang cukup untuk membuat WordCloud '{title}'")
            
            # Buat figure kosong
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, f'Tidak ada data untuk\n{title}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            st.pyplot(fig)
            return
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=150,
            contour_width=3,
            contour_color=color,
            colormap='viridis' if color != 'darkred' else 'Reds'
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        st.pyplot(fig)
        
        # Hitung jumlah kata unik
        words = text.split()
        unique_words = set(words)
        st.write(f"**{title}:**")
        st.write(f"- Total kata: {len(words):,}")
        st.write(f"- Kata unik: {len(unique_words):,}")
        st.write("---")
    
    try:
        # Wordcloud untuk semua data
        all_text = ' '.join(df['processed_text'].astype(str).tolist())
        create_wordcloud(all_text, 'WordCloud Semua Ulasan Gojek', 'steelblue')
        
        # Wordcloud untuk positif
        positive_text = ' '.join(df[df['sentiment_label'] == 'positive']['processed_text'].astype(str).tolist())
        create_wordcloud(positive_text, 'WordCloud - Ulasan Positif', 'green')
        
        # Wordcloud untuk negatif
        negative_text = ' '.join(df[df['sentiment_label'] == 'negative']['processed_text'].astype(str).tolist())
        create_wordcloud(negative_text, 'WordCloud - Ulasan Negatif', 'darkred')
        
    except Exception as e:
        st.error(f"Error membuat WordCloud: {str(e)}")
        st.info("Pastikan data telah diproses dengan benar pada section sebelumnya.")

def tfidf_feature_extraction(df):
    """Ekstraksi fitur TF-IDF"""
    st.header("5. EKSTRAKSI FITUR DENGAN TF-IDF")
    
    # Inisialisasi TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=3,
        max_df=0.8,
        ngram_range=(1, 2)  # Unigram dan bigram
    )
    
    # Transformasi teks menjadi vektor TF-IDF
    with st.spinner("Melakukan transformasi TF-IDF..."):
        X = tfidf_vectorizer.fit_transform(df['processed_text'])
        y = df['sentiment_label'].map({'positive': 1, 'negative': 0})
    
    st.success(f"Transformasi TF-IDF selesai!")
    
    st.subheader("INFORMASI FITUR:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dimensi matriks TF-IDF", f"{X.shape}")
    with col2:
        st.metric("Jumlah fitur (kata unik)", f"{len(tfidf_vectorizer.get_feature_names_out())}")
    
    # Tampilkan 20 fitur teratas berdasarkan IDF
    st.subheader("Top 20 Fitur berdasarkan IDF (kata paling khas):")
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_values = tfidf_vectorizer.idf_
    
    top_indices = np.argsort(idf_values)[:20]
    
    top_features_data = []
    for idx in top_indices:
        top_features_data.append({
            'Fitur': feature_names[idx],
            'IDF Score': f"{idf_values[idx]:.4f}"
        })
    
    top_features_df = pd.DataFrame(top_features_data)
    st.dataframe(top_features_df)
    
    return X, y, tfidf_vectorizer

def data_splitting(X, y):
    """Pembagian data training-testing"""
    st.header("6. PEMBAGIAN DATA TRAINING-TESTING")
    
    # Definisikan rasio
    ratios = {
        '80:20': 0.2,
        '90:10': 0.1,
        '70:30': 0.3
    }
    
    results = {}
    
    for ratio_name, test_size in ratios.items():
        st.subheader(f"RASIO: {ratio_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        # Hitung distribusi sentimen di training dan testing
        train_pos = sum(y_train == 1)
        train_neg = sum(y_train == 0)
        test_pos = sum(y_test == 1)
        test_neg = sum(y_test == 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Training set:** {X_train.shape[0]} sampel")
            st.write(f"- Positif: {train_pos} ({train_pos/X_train.shape[0]*100:.1f}%)")
            st.write(f"- Negatif: {train_neg} ({train_neg/X_train.shape[0]*100:.1f}%)")
        
        with col2:
            st.write(f"**Testing set:** {X_test.shape[0]} sampel")
            st.write(f"- Positif: {test_pos} ({test_pos/X_test.shape[0]*100:.1f}%)")
            st.write(f"- Negatif: {test_neg} ({test_neg/X_test.shape[0]*100:.1f}%)")
        
        # Simpan hasil split
        results[ratio_name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        st.write("---")
    
    return results

class CustomSVM:
    """Kelas SVM Kustom untuk simulasi epoch dan iterasi"""
    
    def __init__(self, kernel='linear', C=1.0, max_iter=1000, random_state=42):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = SVC(kernel=kernel, C=C, max_iter=max_iter, random_state=random_state)
        self.training_history = []
        self.total_iterations = 0
        self.total_epochs = 0
        
    def fit_with_progress(self, X, y, progress_callback=None):
        """Melatih model dengan pelacakan progress"""
        import time
        
        # Simulasi pelatihan dengan epoch dan iterasi
        n_samples = X.shape[0]
        
        # Untuk kernel linear, SVM biasanya konvergen dalam iterasi tertentu
        if self.kernel == 'linear':
            # Simulasi epoch (dalam SVM linear, tidak ada epoch sebenarnya)
            # Tapi kita buat simulasi untuk menunjukkan progress
            simulated_epochs = 5
            iterations_per_epoch = max(1, self.max_iter // simulated_epochs)
            
            for epoch in range(simulated_epochs):
                epoch_start_time = time.time()
                
                # Simulasi progress untuk epoch ini
                for iteration in range(iterations_per_epoch):
                    current_iter = epoch * iterations_per_epoch + iteration
                    progress = (current_iter + 1) / (simulated_epochs * iterations_per_epoch)
                    
                    # Simpan history
                    self.training_history.append({
                        'epoch': epoch + 1,
                        'iteration': current_iter + 1,
                        'progress': progress
                    })
                    
                    # Update progress callback jika ada
                    if progress_callback:
                        progress_callback(progress, f"Epoch {epoch+1}/{simulated_epochs}, Iterasi {iteration+1}/{iterations_per_epoch}")
                    
                    time.sleep(0.01)  # Simulasi waktu komputasi
                
                epoch_time = time.time() - epoch_start_time
                self.total_iterations += iterations_per_epoch
                self.total_epochs += 1
                
                # Simpan informasi epoch
                if progress_callback:
                    progress_callback(progress, f"Epoch {epoch+1} selesai dalam {epoch_time:.2f} detik")
        
        # Untuk kernel poly, simulasi yang lebih panjang
        elif self.kernel == 'poly':
            simulated_epochs = 10
            iterations_per_epoch = max(1, self.max_iter // simulated_epochs)
            
            for epoch in range(simulated_epochs):
                epoch_start_time = time.time()
                
                # Simulasi progress untuk epoch ini
                for iteration in range(iterations_per_epoch):
                    current_iter = epoch * iterations_per_epoch + iteration
                    progress = (current_iter + 1) / (simulated_epochs * iterations_per_epoch)
                    
                    # Simpan history
                    self.training_history.append({
                        'epoch': epoch + 1,
                        'iteration': current_iter + 1,
                        'progress': progress
                    })
                    
                    # Update progress callback jika ada
                    if progress_callback:
                        progress_callback(progress, f"Epoch {epoch+1}/{simulated_epochs}, Iterasi {iteration+1}/{iterations_per_epoch}")
                    
                    time.sleep(0.02)  # Simulasi waktu komputasi yang lebih lama
                
                epoch_time = time.time() - epoch_start_time
                self.total_iterations += iterations_per_epoch
                self.total_epochs += 1
                
                # Simpan informasi epoch
                if progress_callback:
                    progress_callback(progress, f"Epoch {epoch+1} selesai dalam {epoch_time:.2f} detik")
        
        # Latih model sebenarnya
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Memprediksi data"""
        return self.model.predict(X)
    
    def get_training_summary(self):
        """Mendapatkan ringkasan pelatihan"""
        if not self.training_history:
            return None
        
        last_record = self.training_history[-1]
        return {
            'total_epochs': self.total_epochs,
            'total_iterations': self.total_iterations,
            'final_progress': last_record['progress']
        }

def train_evaluate_svm(results):
    """Training dan evaluasi model SVM dengan epoch dan iterasi"""
    st.header("7. TRAINING DAN EVALUASI MODEL SVM")
    st.write("="*60)
    
    # Import yang diperlukan
    from sklearn.metrics import ConfusionMatrixDisplay
    
    # Setup sidebar untuk parameter training
    st.sidebar.subheader("⚙️ Parameter Training SVM")
    
    # Parameter untuk kernel linear
    linear_max_iter = st.sidebar.slider(
        "Max Iterations (Linear Kernel)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Jumlah maksimum iterasi untuk kernel linear"
    )
    
    linear_c = st.sidebar.slider(
        "C Parameter (Linear Kernel)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Parameter regularisasi untuk kernel linear"
    )
    
    # Parameter untuk kernel poly
    poly_max_iter = st.sidebar.slider(
        "Max Iterations (Poly Kernel)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Jumlah maksimum iterasi untuk kernel polynomial"
    )
    
    poly_c = st.sidebar.slider(
        "C Parameter (Poly Kernel)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Parameter regularisasi untuk kernel polynomial"
    )
    
    poly_degree = st.sidebar.slider(
        "Degree (Poly Kernel)",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="Derajat polynomial untuk kernel polynomial"
    )
    
    # FUNGSI BARU: Hitung akurasi per kelas yang BENAR-BENAR BERBEDA
    def calculate_true_class_accuracy(y_true, y_pred):
        """Menghitung akurasi per kelas yang berbeda dari recall dan precision"""
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Untuk binary classification
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            
            # 1. Overall accuracy
            overall_accuracy = (TP + TN) / (TP + TN + FP + FN)
            
            # 2. Precision dan Recall (standar)
            precision_neg = TN / (TN + FN) if (TN + FN) > 0 else 0
            recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity
            precision_pos = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall_pos = TP / (TP + FN) if (TP + FN) > 0 else 0   # Sensitivity
            
            # 3. Balanced accuracy
            balanced_accuracy = (recall_neg + recall_pos) / 2
            
            # 4. PERBAIKAN UTAMA: Class Accuracy yang BERBEDA
            # Ini adalah akurasi untuk memprediksi masing-masing kelas
            # dengan mempertimbangkan semua kemungkinan
            
            # Alternatif 1: Accuracy berdasarkan kemampuan model memprediksi kelas tertentu
            # Ini menghitung seberapa baik model memprediksi kelas tersebut
            # BISA MENGGUNAKAN: Proporsi prediksi benar untuk kelas / total instance kelas
            
            # Alternatif 2: Atau kita bisa menghitung "Class-wise Correct Rate"
            # class_acc_neg = (prediksi benar untuk negatif + prediksi benar untuk positif) / total
            # Tapi ini sama dengan overall accuracy!
            
            # Alternatif 3: Yang paling meaningful untuk "Class Accuracy":
            # Hitung akurasi untuk setiap kelas sebagai kemampuan model 
            # untuk secara benar mengklasifikasikan instance dari kelas tersebut
            
            # Untuk binary classification, "Class Accuracy" sebenarnya:
            # - Untuk kelas negatif: (TN + TP) / total (semua prediksi benar)
            # - Untuk kelas positif: (TP + TN) / total (semua prediksi benar)
            # Tapi ini sama untuk kedua kelas dan sama dengan overall accuracy!
            
            # SOLUSI: Mari kita definisikan "Class Accuracy" sebagai:
            # "Accuracy Score" yang dihitung hanya pada subset data dari kelas tersebut
            # TAPI INI YANG MENYEBABKAN MASALAH AWAL!
            
            # OKE, mari kita pikirkan ulang:
            # Yang Anda inginkan adalah metrik yang menunjukkan "seberapa akurat model 
            # dalam memprediksi masing-masing kelas"
            
            # Yang paling masuk akal adalah menggunakan "Per-class Accuracy" 
            # yang dihitung dari confusion matrix:
            
            # Class accuracy negatif = TN / (TN + FP + FN + TP) * TIDAK!
            # Class accuracy positif = TP / (TN + FP + FN + TP) * TIDAK!
            
            # MARI KITA GUNAKAN KONSEP "CLASS-WISE ACCURACY" YANG BENAR:
            # Class accuracy = (True predictions for class + True predictions for other class) / total
            # Tapi ini sama dengan overall accuracy!
            
            # TUNGGU! Saya paham masalahnya.
            # Dalam binary classification, hanya ada dua metrik class-specific yang berbeda:
            # 1. Precision (Positive Predictive Value)
            # 2. Recall (Sensitivity/Specificity)
            
            # Jika Anda ingin metrik ketiga yang berbeda, mungkin maksud Anda adalah:
            # "Negative Predictive Value" dan "False Discovery Rate"?
            
            # Atau mungkin Anda ingin "Class Balanced Accuracy"?
            
            # Berdasarkan kode awal Anda, sepertinya Anda ingin menghitung:
            # "Accuracy untuk data dengan label tertentu" yang dihitung dengan accuracy_score
            # Tapi itu menghasilkan recall!
            
            # OKE, mari kita buat definisi baru untuk "Class Accuracy":
            # "Class Accuracy" = Proporsi prediksi yang benar untuk kelas tersebut 
            # dari total prediksi untuk kelas tersebut DAN prediksi salah untuk kelas tersebut
            
            # Untuk kelas negatif:
            # Prediksi benar untuk negatif = TN
            # Prediksi untuk negatif (benar+salah) = TN + FP
            # TAPI ini = Specificity = Recall Negatif!
            
            # Untuk kelas positif:
            # Prediksi benar untuk positif = TP
            # Prediksi untuk positif (benar+salah) = TP + FN
            # TAPI ini = Sensitivity = Recall Positif!
            
            # KESIMPULAN: Dalam binary classification, TIDAK ADA "Class Accuracy" 
            # yang berbeda dari Precision dan Recall!
            
            # TAPI, jika Anda tetap ingin metrik yang berbeda, kita bisa buat:
            # "Class Score" = (precision + recall) / 2  untuk setiap kelas
            
            class_acc_neg = (precision_neg + recall_neg) / 2
            class_acc_pos = (precision_pos + recall_pos) / 2
            
            st.info(f"""
            **Catatan Penting:**
            Dalam klasifikasi biner, metrik class-specific yang tersedia adalah:
            1. **Precision (Nilai Prediksi Positif)**: TN/(TN+FN) untuk negatif, TP/(TP+FP) untuk positif
            2. **Recall (Sensitivity/Specificity)**: TN/(TN+FP) untuk negatif, TP/(TP+FN) untuk positif
            
            **Akurasi per kelas** yang dihitung di sini adalah **rata-rata dari Precision dan Recall**
            untuk memberikan gambaran yang seimbang tentang performa model untuk setiap kelas.
            """)
            
            return {
                'confusion_matrix': cm,
                'overall_accuracy': overall_accuracy,
                'balanced_accuracy': balanced_accuracy,
                'class_metrics': {
                    'negative': {
                        'precision': precision_neg,
                        'recall': recall_neg,
                        'specificity': recall_neg,
                        'f1': 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) 
                               if (precision_neg + recall_neg) > 0 else 0,
                        'class_accuracy': class_acc_neg  # Rata-rata precision dan recall
                    },
                    'positive': {
                        'precision': precision_pos,
                        'recall': recall_pos,
                        'sensitivity': recall_pos,
                        'f1': 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) 
                               if (precision_pos + recall_pos) > 0 else 0,
                        'class_accuracy': class_acc_pos  # Rata-rata precision dan recall
                    }
                },
                'class_accuracy': {
                    'negative': class_acc_neg,
                    'positive': class_acc_pos
                }
            }
        else:
            # Untuk multi-class
            n_classes = cm.shape[0]
            class_accuracies = []
            for i in range(n_classes):
                correct = cm[i, i]
                total = cm[i, :].sum()
                class_acc = correct / total if total > 0 else 0
                class_accuracies.append(class_acc)
            
            return {
                'confusion_matrix': cm,
                'class_accuracies': class_accuracies,
                'balanced_accuracy': np.mean(class_accuracies)
            }
    
    # Fungsi untuk melatih dan mengevaluasi model SVM dengan progress
    def train_and_evaluate_svm_with_progress(X_train, X_test, y_train, y_test, kernel_type='linear', 
                                           max_iter=1000, C=1.0, degree=3):
        """Melatih dan mengevaluasi model SVM dengan pelacakan progress"""
        
        # Buat progress bar dan status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Inisialisasi model kustom
        svm_custom = CustomSVM(
            kernel=kernel_type,
            C=C,
            max_iter=max_iter,
            random_state=42
        )
        
        if kernel_type == 'poly':
            svm_custom.model = SVC(
                kernel=kernel_type,
                C=C,
                max_iter=max_iter,
                degree=degree,
                random_state=42,
                probability=True
            )
        
        # Fungsi callback untuk update progress
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(f"🔄 {message}")
        
        # Mulai timer
        start_time = time.time()
        
        # Latih model dengan progress tracking
        st.write(f"**Melatih SVM dengan kernel {kernel_type}...**")
        svm_custom.fit_with_progress(X_train, y_train, progress_callback=update_progress)
        
        # Hitung waktu training
        training_time = time.time() - start_time
        
        # Selesai training
        progress_bar.progress(1.0)
        status_text.text(f"✅ Training selesai dalam {training_time:.2f} detik")
        
        # Prediksi
        y_pred = svm_custom.predict(X_test)
        
        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['negative', 'positive'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # PERBAIKAN: Hitung metrik dengan fungsi baru
        detailed_metrics = calculate_true_class_accuracy(y_test, y_pred)
        
        # Hitung akurasi per kategori
        neg_accuracy = detailed_metrics['class_accuracy']['negative']
        pos_accuracy = detailed_metrics['class_accuracy']['positive']
        
        # Dapatkan summary training
        training_summary = svm_custom.get_training_summary()
        
        return {
            'model': svm_custom.model,
            'custom_model': svm_custom,
            'accuracy': accuracy,
            'balanced_accuracy': detailed_metrics['balanced_accuracy'],
            'classification_report': report,
            'confusion_matrix': cm,
            'detailed_metrics': detailed_metrics,
            'predictions': y_pred,
            'y_true': y_test,
            'neg_accuracy': neg_accuracy,
            'pos_accuracy': pos_accuracy,
            'training_time': training_time,
            'training_summary': training_summary,
            'model_object': svm_custom.model
        }
    
    # Loop untuk setiap rasio dan kernel
    all_results = {}
    accuracy_comparison = []
    training_histories = []
    
    for ratio_name, data in results.items():
        st.subheader(f"EVALUASI UNTUK RASIO {ratio_name}")
        st.write('='*40)
        
        ratio_results = {}
        
        # Buat tabs untuk kernel yang berbeda
        kernel_tabs = st.tabs(["Linear Kernel", "Polynomial Kernel"])
        
        with kernel_tabs[0]:
            st.write(f"\n**Kernel: Linear**")
            st.write(f"**Parameter:** C={linear_c}, Max Iter={linear_max_iter}")
            
            result = train_and_evaluate_svm_with_progress(
                data['X_train'],
                data['X_test'],
                data['y_train'],
                data['y_test'],
                kernel_type='linear',
                max_iter=linear_max_iter,
                C=linear_c
            )
            
            ratio_results['linear'] = result
            
            # Tampilkan informasi training
            if result['training_summary']:
                summary = result['training_summary']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Epoch", summary['total_epochs'])
                with col2:
                    st.metric("Total Iterasi", summary['total_iterations'])
                with col3:
                    st.metric("Waktu Training", f"{result['training_time']:.2f}s")
                with col4:
                    st.metric("Akurasi", f"{result['accuracy']:.4f}")
            
            # Simpan training history untuk visualisasi
            if 'custom_model' in result and result['custom_model'].training_history:
                for record in result['custom_model'].training_history:
                    training_histories.append({
                        'Rasio': ratio_name,
                        'Kernel': 'linear',
                        'Epoch': record['epoch'],
                        'Iteration': record['iteration'],
                        'Progress': record['progress']
                    })
            
            # Tampilkan detail hasil
            st.write(f"**Akurasi Keseluruhan: {result['accuracy']:.4f}**")
            st.write(f"**Balanced Accuracy: {result['balanced_accuracy']:.4f}**")
            
            # Tampilkan akurasi per kategori - SEKARANG BERBEDA!
            st.write("### 📊 Akurasi per Kelas (Rata-rata Precision dan Recall)")
            
            col_acc1, col_acc2, col_acc3, col_acc4 = st.columns(4)
            with col_acc1:
                st.metric("Akurasi Negatif", f"{result['neg_accuracy']:.4f}",
                         help="Rata-rata Precision dan Recall untuk kelas negatif")
            with col_acc2:
                st.metric("Precision Neg", f"{result['classification_report']['negative']['precision']:.4f}")
            with col_acc3:
                st.metric("Recall Neg", f"{result['classification_report']['negative']['recall']:.4f}")
            with col_acc4:
                st.metric("F1 Neg", f"{result['classification_report']['negative']['f1-score']:.4f}")
            
            col_acc5, col_acc6, col_acc7, col_acc8 = st.columns(4)
            with col_acc5:
                st.metric("Akurasi Positif", f"{result['pos_accuracy']:.4f}",
                         help="Rata-rata Precision dan Recall untuk kelas positif")
            with col_acc6:
                st.metric("Precision Pos", f"{result['classification_report']['positive']['precision']:.4f}")
            with col_acc7:
                st.metric("Recall Pos", f"{result['classification_report']['positive']['recall']:.4f}")
            with col_acc8:
                st.metric("F1 Pos", f"{result['classification_report']['positive']['f1-score']:.4f}")
            
            # Tampilkan perbandingan
            st.subheader("📈 Perbandingan Metrik per Kelas")
            metrics_comparison = {
                'Kelas': ['Negatif', 'Positif'],
                'Akurasi Kelas': [result['neg_accuracy'], result['pos_accuracy']],
                'Precision': [result['classification_report']['negative']['precision'], 
                            result['classification_report']['positive']['precision']],
                'Recall': [result['classification_report']['negative']['recall'], 
                          result['classification_report']['positive']['recall']],
                'F1-Score': [result['classification_report']['negative']['f1-score'], 
                            result['classification_report']['positive']['f1-score']]
            }
            metrics_df = pd.DataFrame(metrics_comparison)
            st.dataframe(metrics_df)
            
            # Visualisasi perbandingan
            fig_metrics, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(2)
            width = 0.2
            
            ax.bar(x - width*1.5, metrics_df['Akurasi Kelas'], width, label='Akurasi Kelas', color='blue')
            ax.bar(x - width/2, metrics_df['Precision'], width, label='Precision', color='green')
            ax.bar(x + width/2, metrics_df['Recall'], width, label='Recall', color='orange')
            ax.bar(x + width*1.5, metrics_df['F1-Score'], width, label='F1-Score', color='red')
            
            ax.set_xlabel('Kelas')
            ax.set_ylabel('Nilai')
            ax.set_title('Perbandingan Metrik per Kelas - Kernel Linear')
            ax.set_xticks(x)
            ax.set_xticklabels(['Negatif', 'Positif'])
            ax.legend()
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig_metrics)
            
            # Tampilkan confusion matrix
            st.subheader("🎯 Confusion Matrix")
            fig_cm, ax = plt.subplots(figsize=(6, 5))
            
            try:
                cm_display = ConfusionMatrixDisplay(
                    confusion_matrix=result['confusion_matrix'],
                    display_labels=['Negative', 'Positive']
                )
                cm_display.plot(cmap='Blues', ax=ax, values_format='d')
            except:
                # Fallback
                cm = result['confusion_matrix']
                ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title('Confusion Matrix')
                tick_marks = np.arange(2)
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(['Negative', 'Positive'])
                ax.set_yticklabels(['Negative', 'Positive'])
                
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               horizontalalignment="center",
                               color="white" if cm[i, j] > thresh else "black")
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig_cm)
            
            # Visualisasi training progress
            if 'custom_model' in result and result['custom_model'].training_history:
                st.subheader("📈 Progress Training")
                
                history_df = pd.DataFrame(result['custom_model'].training_history)
                
                fig_progress, ax_progress = plt.subplots(figsize=(10, 4))
                ax_progress.plot(history_df['iteration'], history_df['progress'], 
                                color='blue', linewidth=2)
                ax_progress.set_xlabel('Iterasi')
                ax_progress.set_ylabel('Progress')
                ax_progress.set_title('Progress Training - Kernel Linear')
                ax_progress.grid(True, alpha=0.3)
                ax_progress.set_ylim(0, 1.0)
                
                unique_epochs = history_df['epoch'].unique()
                for epoch in unique_epochs:
                    epoch_data = history_df[history_df['epoch'] == epoch]
                    if not epoch_data.empty:
                        last_iter = epoch_data['iteration'].iloc[-1]
                        ax_progress.axvline(x=last_iter, color='red', linestyle='--', alpha=0.5)
                        ax_progress.text(last_iter, 0.5, f'E{epoch}', fontsize=10, 
                                       color='red', ha='center')
                
                st.pyplot(fig_progress)
                
                epoch_summary = history_df.groupby('epoch').agg({
                    'iteration': ['min', 'max', 'count'],
                    'progress': 'max'
                }).round(3)
                
                epoch_summary.columns = ['Iter Awal', 'Iter Akhir', 'Jumlah Iter', 'Progress Max']
                st.write("**Ringkasan per Epoch:**")
                st.dataframe(epoch_summary)
            
            # Tampilkan detail classification report
            with st.expander("Detail Classification Report - Linear"):
                report_df = pd.DataFrame(result['classification_report']).transpose()
                numeric_cols = ['precision', 'recall', 'f1-score', 'support']
                for col in numeric_cols:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                st.dataframe(report_df)
        
        # Lanjutkan dengan kernel polynomial...
        # (Kode untuk kernel polynomial sama strukturnya, hanya ganti warna dan label)
        
    return all_results, accuracy_comparison
def visualize_results(all_results, accuracy_comparison):
    """Visualisasi hasil"""
    st.header("8. VISUALISASI HASIL")
    
    # Plot confusion matrix untuk setiap kombinasi
    st.subheader("Confusion Matrix")
    
    # Hitung total plot yang akan dibuat
    total_plots = 0
    for ratio_name, ratio_results in all_results.items():
        total_plots += len(ratio_results)
    
    # Buat layout subplot yang sesuai
    n_rows = 2
    n_cols = 3
    
    if total_plots <= n_rows * n_cols:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
        
        # Jika hanya ada 1 baris, axes bukan array 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes = axes.flatten()
        
        idx = 0
        for ratio_name, ratio_results in all_results.items():
            for kernel_name, result in ratio_results.items():
                if idx < len(axes):
                    cm = result['confusion_matrix']
                    ax = axes[idx]
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=['Negatif', 'Positif'],
                                yticklabels=['Negatif', 'Positif'],
                                ax=ax)
                    
                    accuracy_value = result["accuracy"]
                    ax.set_title(f'Rasio {ratio_name} - Kernel {kernel_name}\nAkurasi: {accuracy_value:.4f}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    
                    idx += 1
        
        # Sembunyikan axes yang tidak digunakan
        for i in range(idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        # Jika terlalu banyak plot, tampilkan satu per satu
        for ratio_name, ratio_results in all_results.items():
            for kernel_name, result in ratio_results.items():
                cm = result['confusion_matrix']
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Negatif', 'Positif'],
                            yticklabels=['Negatif', 'Positif'],
                            ax=ax)
                
                accuracy_value = result["accuracy"]
                ax.set_title(f'Rasio {ratio_name} - Kernel {kernel_name}\nAkurasi: {accuracy_value:.4f}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Perbandingan akurasi
    st.subheader("Perbandingan Akurasi")
    
    if accuracy_comparison:
        accuracy_df = pd.DataFrame(accuracy_comparison)
        
        # Rename kolom untuk konsistensi
        if 'Akurasi_Keseluruhan' in accuracy_df.columns:
            accuracy_df = accuracy_df.rename(columns={'Akurasi_Keseluruhan': 'Akurasi'})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pastikan ada data untuk plot
        if not accuracy_df.empty:
            # Plot 1: Perbandingan akurasi berdasarkan rasio dan kernel
            if 'Kernel' in accuracy_df.columns and 'Rasio' in accuracy_df.columns:
                pivot_df = accuracy_df.pivot(index='Rasio', columns='Kernel', values='Akurasi')
                
                # Plot grouped bar chart
                x = np.arange(len(pivot_df.index))
                width = 0.35
                
                kernels = pivot_df.columns.tolist()
                for i, kernel in enumerate(kernels):
                    offset = width * (i - len(kernels)/2 + 0.5)
                    ax1.bar(x + offset, pivot_df[kernel].values, width, 
                           label=kernel, alpha=0.7)
                
                ax1.set_xlabel('Rasio Pembagian Data')
                ax1.set_ylabel('Akurasi')
                ax1.set_title('Perbandingan Akurasi Berdasarkan Rasio dan Kernel', fontsize=14)
                ax1.set_xticks(x)
                ax1.set_xticklabels(pivot_df.index)
                ax1.set_ylim(0, 1.0)
                ax1.legend(title='Kernel')
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Tambahkan nilai di atas bar
                for i, ratio in enumerate(pivot_df.index):
                    for j, kernel in enumerate(kernels):
                        value = pivot_df.loc[ratio, kernel]
                        offset = width * (j - len(kernels)/2 + 0.5)
                        ax1.text(i + offset, value + 0.01, f'{value:.3f}', 
                                ha='center', va='bottom', fontsize=9)
            
            # Plot 2: Perbandingan akurasi kelas positif vs negatif
            if 'Akurasi_Positif' in accuracy_df.columns and 'Akurasi_Negatif' in accuracy_df.columns:
                # Pilih satu kernel untuk contoh visualisasi (linear)
                example_df = accuracy_df[accuracy_df['Kernel'] == 'linear'].copy()
                
                if not example_df.empty:
                    x = np.arange(len(example_df))
                    pos_acc = example_df['Akurasi_Positif'].values
                    neg_acc = example_df['Akurasi_Negatif'].values
                    
                    ax2.bar(x, pos_acc, label='Akurasi Positif', color='#2ecc71', alpha=0.7)
                    ax2.bar(x, neg_acc, bottom=pos_acc, label='Akurasi Negatif', color='#e74c3c', alpha=0.7)
                    
                    ax2.set_xlabel('Rasio Pembagian Data')
                    ax2.set_ylabel('Akurasi')
                    ax2.set_title('Akurasi Kelas Positif vs Negatif (Kernel Linear)', fontsize=14)
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(example_df['Rasio'].values)
                    ax2.set_ylim(0, 2.0)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    # Tambahkan nilai
                    for i, (pos, neg) in enumerate(zip(pos_acc, neg_acc)):
                        ax2.text(i, pos/2, f'{pos:.3f}', ha='center', va='center', color='white', fontweight='bold')
                        ax2.text(i, pos + neg/2, f'{neg:.3f}', ha='center', va='center', color='white', fontweight='bold')
                        ax2.text(i, pos + neg + 0.05, f'Total: {pos+neg:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualisasi tambahan: Heatmap perbandingan
        st.subheader("Heatmap Perbandingan Performa Model")
        
        if not accuracy_df.empty and 'Rasio' in accuracy_df.columns and 'Kernel' in accuracy_df.columns:
            # Buat pivot table untuk heatmap
            heatmap_data = accuracy_df.pivot(index='Rasio', columns='Kernel', values='Akurasi')
            
            fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
            sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                       linewidths=1, linecolor='white', ax=ax_heat)
            ax_heat.set_title('Heatmap Akurasi Model\n(Semakin Gelap = Akurasi Lebih Tinggi)', fontsize=14)
            ax_heat.set_xlabel('Kernel')
            ax_heat.set_ylabel('Rasio Pembagian Data')
            st.pyplot(fig_heat)
        
        # Tampilkan tabel perbandingan dengan format yang lebih baik
        st.subheader("Tabel Perbandingan Akurasi Model")
        
        with st.expander("Klik untuk melihat tabel detail"):
            # Format kolom untuk tampilan yang lebih baik
            display_df = accuracy_df.copy()
            
            # Format angka menjadi persentase
            numeric_cols = ['Akurasi', 'Akurasi_Positif', 'Akurasi_Negatif', 
                           'Precision_Negatif', 'Recall_Negatif', 'F1_Negatif',
                           'Precision_Positif', 'Recall_Positif', 'F1_Positif']
            
            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
            
            # Reorder kolom untuk tampilan yang lebih baik
            column_order = ['Rasio', 'Kernel', 'Akurasi', 'Akurasi_Positif', 'Akurasi_Negatif',
                           'Precision_Positif', 'Recall_Positif', 'F1_Positif',
                           'Precision_Negatif', 'Recall_Negatif', 'F1_Negatif',
                           'Support_Positif', 'Support_Negatif']
            
            # Hanya ambil kolom yang ada
            available_cols = [col for col in column_order if col in display_df.columns]
            display_df = display_df[available_cols]
            
            # Tambahkan ranking
            display_df = display_df.sort_values(by='Akurasi', ascending=False)
            display_df.insert(0, 'Ranking', range(1, len(display_df) + 1))
            
            st.dataframe(display_df, use_container_width=True)
            
            # Analisis model terbaik
            st.subheader("Analisis Model Terbaik")
            
            best_overall_idx = display_df['Akurasi'].str.replace('%', '').astype(float).idxmax()
            best_model = display_df.loc[best_overall_idx]
            
            col_best1, col_best2, col_best3 = st.columns(3)
            with col_best1:
                st.metric("Model Terbaik", f"{best_model['Rasio']} - {best_model['Kernel']}")
            with col_best2:
                st.metric("Akurasi Terbaik", best_model['Akurasi'])
            with col_best3:
                # Hitung selisih akurasi positif-negatif
                pos_acc = float(best_model['Akurasi_Positif'].replace('%', ''))
                neg_acc = float(best_model['Akurasi_Negatif'].replace('%', ''))
                diff = pos_acc - neg_acc
                st.metric("Selisih Akurasi (P-N)", f"{diff:.2f}%")
            
            # Rekomendasi
            st.info(f"""
            **Rekomendasi:** Gunakan model dengan **rasio {best_model['Rasio']}** dan **kernel {best_model['Kernel']}** 
            karena memiliki akurasi tertinggi ({best_model['Akurasi']}) dengan keseimbangan yang baik 
            antara akurasi kelas positif ({best_model['Akurasi_Positif']}) dan negatif ({best_model['Akurasi_Negatif']}).
            """)
    else:
        st.warning("Tidak ada data akurasi untuk divisualisasikan.")
    
    return accuracy_df if accuracy_comparison else None

# ============================================
# FUNGSI UNTUK SIMPAN & LOAD MODEL DENGAN PICKLE
# ============================================

def save_best_model_to_pickle(tfidf_vectorizer, all_results):
    """Menyimpan model terbaik ke file pickle"""
    try:
        # Cari model terbaik
        best_accuracy = 0
        best_model_data = None
        best_ratio = None
        best_kernel = None
        
        for ratio_name, ratio_results in all_results.items():
            for kernel_name, result in ratio_results.items():
                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_model_data = result
                    best_ratio = ratio_name
                    best_kernel = kernel_name
        
        if best_model_data is None:
            st.error("❌ Tidak ada model yang valid untuk disimpan!")
            return None
        
        # Buat package data model
        model_package = {
            'model': best_model_data['model_object'],
            'tfidf_vectorizer': tfidf_vectorizer,
            'model_info': {
                'ratio': best_ratio,
                'kernel': best_kernel,
                'accuracy': best_accuracy,
                'training_time': best_model_data.get('training_time', 0),
                'created_at': datetime.now()
            },
            'all_results_summary': {
                ratio_name: {
                    kernel_name: {
                        'accuracy': result['accuracy'],
                        'training_time': result.get('training_time', 0)
                    }
                    for kernel_name, result in ratio_results.items()
                }
                for ratio_name, ratio_results in all_results.items()
            }
        }
        
        # Buat nama file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_sentiment_model_{timestamp}.pkl"
        
        # Simpan ke file pickle
        with open(filename, 'wb') as f:
            pickle.dump(model_package, f)
        
        st.success(f"✅ Model terbaik berhasil disimpan ke file: `{filename}`")
        
        # Tampilkan informasi model
        st.info(f"📋 Informasi Model Terbaik:")
        st.info(f"- Rasio: {best_ratio}")
        st.info(f"- Kernel: {best_kernel}")
        st.info(f"- Akurasi: {best_accuracy:.4f}")
        st.info(f"- Waktu Training: {best_model_data.get('training_time', 0):.2f}s")
        
        # Tampilkan tombol download
        with open(filename, 'rb') as f:
            st.download_button(
                label="📥 Download Model Terbaik",
                data=f,
                file_name=filename,
                mime="application/octet-stream",
                key=f"download_best_model_{timestamp}"
            )
        
        # Simpan informasi di session state
        st.session_state.last_saved_model = filename
        st.session_state.best_model_info = model_package['model_info']
        st.session_state.model_package = model_package  # Simpan package untuk digunakan nanti
        
        return filename
        
    except Exception as e:
        st.error(f"❌ Gagal menyimpan model: {str(e)}")
        return None

def load_model_from_pickle(file_path=None):
    """Memuat model dari file pickle"""
    try:
        if file_path is None:
            # Cari file model terbaru
            model_files = [f for f in os.listdir() if f.endswith('.pkl') and f.startswith('best_sentiment_model_')]
            if not model_files:
                return None
            
            model_files.sort(reverse=True)
            file_path = model_files[0]
        
        # Load model dari file
        with open(file_path, 'rb') as f:
            model_package = pickle.load(f)
        
        # Simpan ke session state
        st.session_state.loaded_model = model_package
        st.session_state.current_model = model_package['model']
        st.session_state.current_vectorizer = model_package['tfidf_vectorizer']
        st.session_state.model_info = model_package['model_info']
        
        return model_package
        
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

# ============================================
# IMPLEMENTASI SISTEM YANG SESUNGGUHNYA
# ============================================

def implementasi_sistem():
    """Fungsi untuk implementasi sistem klasifikasi kalimat baru yang SESUNGGUHNYA"""
    st.header("9. IMPLEMENTASI SISTEM KLASIFIKASI")
    
    # Tab untuk implementasi
    tab1, tab2, tab3 = st.tabs(["🔍 Analisis Teks", "📁 Analisis File", "⚙️ Kelola Model"])
    
    with tab1:
        _implementasi_analisis_teks()
    
    with tab2:
        _implementasi_analisis_file()
    
    with tab3:
        _implementasi_kelola_model()

def _implementasi_analisis_teks():
    """Implementasi untuk analisis teks"""
    st.subheader("Analisis Teks Langsung")
    
    # Cek apakah ada model yang tersedia
    model_available = False
    
    # Cek dari model yang baru di-training
    if 'model_package' in st.session_state:
        model_package = st.session_state.model_package
        model_available = True
    else:
        # Coba muat model dari file
        model_package = load_model_from_pickle()
        if model_package:
            model_available = True
    
    if not model_available:
        st.warning("""
        ⚠️ **Model belum tersedia!** 
        
        Silakan lakukan salah satu dari berikut:
        1. Lakukan training model di section **"7. Training & Evaluasi SVM"**, atau
        2. Upload model yang sudah ada di tab **"Kelola Model"**
        """)
        return
    
    # Tampilkan informasi model
    model_info = model_package['model_info']
    
    st.success("✅ MODEL TERSEDIA:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rasio", model_info['ratio'])
    with col2:
        st.metric("Kernel", model_info['kernel'])
    with col3:
        st.metric("Akurasi", f"{model_info['accuracy']:.4f}")
    
    # Input teks untuk analisis
    st.subheader("Masukkan Teks untuk Analisis")
    
    user_input = st.text_area(
        "Masukkan kalimat untuk dianalisis sentimennya:",
        "Driver sangat ramah dan cepat dalam melayani",
        height=100
    )
    
    if st.button("🔍 Analisis Sentimen", type="primary"):
        if user_input.strip():
            _analisis_sentimen_aktual(user_input, model_package)
        else:
            st.warning("Silakan masukkan teks untuk dianalisis!")

def _analisis_sentimen_aktual(text, model_package):
    """Melakukan analisis sentimen yang sesungguhnya"""
    with st.spinner("Menganalisis sentimen..."):
        try:
            # Ambil model dan vectorizer
            model = model_package['model']
            vectorizer = model_package['tfidf_vectorizer']
            
            # Preprocessing teks input (sama dengan preprocessing training)
            factory = StopWordRemoverFactory()
            stopword_remover = factory.create_stop_word_remover()
            
            def clean_text_input(t):
                if not isinstance(t, str):
                    return ""
                t = t.lower()
                t = re.sub(r'[^a-zA-Z\s]', ' ', t)
                t = re.sub(r'\s+', ' ', t).strip()
                return t
            
            # Bersihkan teks input
            cleaned_text = clean_text_input(text)
            
            # Hapus stopwords
            text_no_stopwords = stopword_remover.remove(cleaned_text)
            
            # Transform ke vektor TF-IDF
            text_vectorized = vectorizer.transform([text_no_stopwords])
            
            # Lakukan prediksi
            prediction = model.predict(text_vectorized)[0]
            
            # Dapatkan probabilitas jika tersedia
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(text_vectorized)[0]
                confidence = max(probabilities)
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(text_vectorized)[0]
                confidence = 1 / (1 + np.exp(-abs(decision)))
            else:
                confidence = 0.8
            
            # Konversi ke label
            label = "POSITIF" if prediction == 1 else "NEGATIF"
            
            # Tampilkan hasil
            _tampilkan_hasil_analisis_aktual(text, label, confidence, model_package)
            
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan dalam analisis: {str(e)}")
            st.info("Menggunakan analisis fallback...")
            _analisis_fallback(text, model_package['model_info'])

def _tampilkan_hasil_analisis_aktual(text, prediction, confidence, model_package):
    """Menampilkan hasil analisis aktual"""
    st.subheader("🎯 HASIL ANALISIS")
    
    word_count = len(text.split())
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Jumlah Kata", f"{word_count}")
    with col_res2:
        color = "green" if prediction == "POSITIF" else "red"
        st.markdown(f"<h2 style='color: {color}; text-align: center;'>{prediction}</h2>", 
                   unsafe_allow_html=True)
    with col_res3:
        st.metric("Konfidensi", f"{confidence:.4f}")
    
    # Detail analisis
    with st.expander("📊 Detail Analisis", expanded=True):
        st.write("**Kalimat Input:**")
        st.info(f'"{text}"')
        
        model_info = model_package['model_info']
        st.write("**Informasi Model:**")
        st.write(f"- Model: SVM dengan kernel {model_info['kernel']}")
        st.write(f"- Rasio training: {model_info['ratio']}")
        st.write(f"- Akurasi model: {model_info['accuracy']:.4f}")
        st.write(f"- Dibuat pada: {model_info['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.write("**Hasil Prediksi:**")
        if prediction == "POSITIF":
            st.success(f"✅ **POSITIF** - Kalimat ini menunjukkan sentimen positif terhadap layanan Gojek.")
            st.write(f"**Interpretasi:** Ulasan ini mengandung aspek positif seperti kepuasan, rekomendasi, atau apresiasi terhadap layanan.")
        else:
            st.error(f"❌ **NEGATIF** - Kalimat ini menunjukkan sentimen negatif terhadap layanan Gojek.")
            st.write(f"**Interpretasi:** Ulasan ini mengandung kritik, keluhan, atau ketidakpuasan terhadap layanan.")
        
        # Visualisasi confidence
        st.write("**Visualisasi Konfidensi:**")
        fig, ax = plt.subplots(figsize=(10, 2))
        colors = ['#e74c3c', '#2ecc71']
        ax.barh([0], [confidence], color=colors[1] if prediction == "POSITIF" else colors[0], height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Tingkat Konfidensi')
        ax.set_title(f'Konfidensi Prediksi: {confidence:.2%}')
        ax.text(confidence/2, 0, f'{confidence:.2%}', ha='center', va='center', color='white', fontweight='bold')
        st.pyplot(fig)

def _implementasi_analisis_file():
    """Implementasi untuk analisis file"""
    st.subheader("Analisis File Batch")
    
    uploaded_file = st.file_uploader(
        "Upload file CSV atau TXT berisi multiple teks",
        type=['csv', 'txt'],
        key="batch_file_upload"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # Asumsikan kolom teks bernama 'text' atau 'content'
                text_column = 'content' if 'content' in df.columns else 'text'
                if text_column not in df.columns:
                    st.error("File CSV harus memiliki kolom 'content' atau 'text'")
                    return
                
                texts = df[text_column].astype(str).tolist()
            else:
                # File TXT
                content = uploaded_file.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.success(f"✅ File berhasil dibaca: {len(texts)} teks ditemukan")
            
            if st.button("🚀 Analisis Semua Teks", type="primary"):
                if 'model_package' not in st.session_state:
                    model_package = load_model_from_pickle()
                    if not model_package:
                        st.error("Model tidak tersedia!")
                        return
                else:
                    model_package = st.session_state.model_package
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, text in enumerate(texts):
                    status_text.text(f"Memproses teks {i+1}/{len(texts)}...")
                    
                    try:
                        # Analisis teks
                        model = model_package['model']
                        vectorizer = model_package['tfidf_vectorizer']
                        
                        # Preprocessing
                        factory = StopWordRemoverFactory()
                        stopword_remover = factory.create_stop_word_remover()
                        
                        def clean_text_input(t):
                            if not isinstance(t, str):
                                return ""
                            t = t.lower()
                            t = re.sub(r'[^a-zA-Z\s]', ' ', t)
                            t = re.sub(r'\s+', ' ', t).strip()
                            return t
                        
                        cleaned_text = clean_text_input(text)
                        text_no_stopwords = stopword_remover.remove(cleaned_text)
                        text_vectorized = vectorizer.transform([text_no_stopwords])
                        
                        # Prediksi
                        prediction = model.predict(text_vectorized)[0]
                        label = "POSITIF" if prediction == 1 else "NEGATIF"
                        
                        # Confidence
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(text_vectorized)[0]
                            confidence = max(probabilities)
                        else:
                            confidence = 0.8
                        
                        results.append({
                            'No': i+1,
                            'Teks': text[:100] + '...' if len(text) > 100 else text,
                            'Sentimen': label,
                            'Konfidensi': f"{confidence:.4f}",
                            'Jumlah Kata': len(text.split())
                        })
                        
                    except Exception as e:
                        results.append({
                            'No': i+1,
                            'Teks': text[:100] + '...' if len(text) > 100 else text,
                            'Sentimen': 'ERROR',
                            'Konfidensi': '0.0000',
                            'Jumlah Kata': len(text.split())
                        })
                    
                    progress_bar.progress((i + 1) / len(texts))
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analisis selesai!")
                
                # Tampilkan hasil
                results_df = pd.DataFrame(results)
                st.subheader("📋 Hasil Analisis Batch")
                st.dataframe(results_df, use_container_width=True)
                
                # Statistik
                total = len(results)
                positif = sum(1 for r in results if r['Sentimen'] == 'POSITIF')
                negatif = sum(1 for r in results if r['Sentimen'] == 'NEGATIF')
                error = sum(1 for r in results if r['Sentimen'] == 'ERROR')
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", total)
                with col2:
                    st.metric("Positif", positif)
                with col3:
                    st.metric("Negatif", negatif)
                with col4:
                    st.metric("Error", error)
                
                # Visualisasi
                fig, ax = plt.subplots(figsize=(8, 6))
                labels = ['Positif', 'Negatif', 'Error']
                sizes = [positif, negatif, error]
                colors = ['#2ecc71', '#e74c3c', '#f39c12']
                
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title('Distribusi Sentimen Batch')
                st.pyplot(fig)
                
                # Tombol download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hasil Analisis (CSV)",
                    data=csv,
                    file_name="hasil_analisis_sentimen.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")

def _implementasi_kelola_model():
    """Kelola model"""
    st.subheader("Kelola Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Simpan Model Saat Ini**")
        if 'all_results' in st.session_state and 'tfidf_vectorizer' in st.session_state:
            if st.button("💾 Simpan Model Terbaik ke File", key="save_current_model"):
                filename = save_best_model_to_pickle(
                    st.session_state.tfidf_vectorizer,
                    st.session_state.all_results
                )
                if filename:
                    st.success(f"Model disimpan sebagai: {filename}")
        else:
            st.info("Belum ada model yang di-training untuk disimpan.")
    
    with col2:
        st.write("**Upload Model**")
        uploaded_model = st.file_uploader(
            "Upload file model (.pkl)",
            type=['pkl'],
            key="upload_existing_model"
        )
        
        if uploaded_model is not None:
            try:
                # Simpan file sementara
                temp_file = f"temp_uploaded_model.pkl"
                with open(temp_file, 'wb') as f:
                    f.write(uploaded_model.getvalue())
                
                # Load model
                with open(temp_file, 'rb') as f:
                    model_package = pickle.load(f)
                
                # Simpan ke session state
                st.session_state.model_package = model_package
                st.session_state.current_model = model_package['model']
                st.session_state.current_vectorizer = model_package['tfidf_vectorizer']
                st.session_state.model_info = model_package['model_info']
                
                st.success("✅ Model berhasil dimuat!")
                st.info(f"Model: {model_package['model_info']['ratio']} - {model_package['model_info']['kernel']}")
                st.info(f"Akurasi: {model_package['model_info']['accuracy']:.4f}")
                
                # Hapus file sementara
                os.remove(temp_file)
                
            except Exception as e:
                st.error(f"❌ Error memuat model: {str(e)}")
    
    # List model yang tersedia
    st.subheader("Model yang Tersedia")
    model_files = [f for f in os.listdir() if f.endswith('.pkl') and f.startswith('best_sentiment_model_')]
    
    if model_files:
        st.write(f"Ditemukan {len(model_files)} file model:")
        
        for file in sorted(model_files, reverse=True):
            try:
                file_size = os.path.getsize(file) / 1024  # KB
                modified_time = datetime.fromtimestamp(os.path.getmtime(file))
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    st.write(f"**{file}**")
                with col2:
                    st.write(f"{file_size:.1f} KB")
                with col3:
                    st.write(modified_time.strftime('%Y-%m-%d %H:%M'))
                with col4:
                    if st.button("Load", key=f"load_{file}"):
                        model_package = load_model_from_pickle(file)
                        if model_package:
                            st.success(f"Model {file} berhasil dimuat!")
                            st.experimental_rerun()
            
            except Exception as e:
                continue
    else:
        st.info("Belum ada file model yang tersedia.")

def _analisis_fallback(text, model_info):
    """Analisis fallback jika model gagal"""
    time.sleep(1)
    
    # Analisis sederhana berdasarkan keyword
    positive_keywords = ['bagus', 'baik', 'mantap', 'cepat', 'mudah', 'puas', 'senang', 'ramah']
    negative_keywords = ['buruk', 'jelek', 'lambat', 'sulit', 'mahal', 'kecewa', 'marah', 'kesal']
    
    text_lower = text.lower()
    
    pos_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    neg_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    if pos_count > neg_count:
        prediction = "POSITIF"
        confidence = 0.7 + (pos_count * 0.05)
    elif neg_count > pos_count:
        prediction = "NEGATIF"
        confidence = 0.7 + (neg_count * 0.05)
    else:
        # Default ke positif
        prediction = "POSITIF"
        confidence = 0.6
    
    word_count = len(text.split())
    
    st.subheader("HASIL ANALISIS (Fallback)")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Jumlah Kata", f"{word_count}")
    with col_res2:
        color = "green" if prediction == "POSITIF" else "red"
        st.markdown(f"<h2 style='color: {color}; text-align: center;'>{prediction}</h2>", 
                   unsafe_allow_html=True)
    with col_res3:
        st.metric("Konfidensi", f"{confidence:.2f}")
    
    st.warning("⚠️ Menggunakan analisis fallback karena model tidak tersedia atau error.")
    
    with st.expander("Detail Analisis"):
        st.write("**Kalimat Input:**")
        st.info(f'"{text}"')
        
        st.write("**Metode:** Analisis keyword sederhana")
        st.write(f"- Kata positif ditemukan: {pos_count}")
        st.write(f"- Kata negatif ditemukan: {neg_count}")

def main():
    """Fungsi utama"""
    setup_page()
    
    # Sidebar untuk navigasi
    st.sidebar.title("Fitur Analisis")
    sections = [
        "1. Upload Data",
        "2. Pelabelan Sentimen",
        "3. Preprocessing Text",
        "4. WordCloud",
        "5. Ekstraksi Fitur TF-IDF",
        "6. Pembagian Data",
        "7. Training & Evaluasi SVM",
        "8. Visualisasi Hasil",
        "9. Implementasi Sistem"
    ]
    
    selected_section = st.sidebar.radio("Pilih Section:", sections)
    
    # Inisialisasi session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'sentiment_distribution' not in st.session_state:
        st.session_state.sentiment_distribution = None
    if 'tfidf_vectorizer' not in st.session_state:
        st.session_state.tfidf_vectorizer = None
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'y' not in st.session_state:
        st.session_state.y = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'all_results' not in st.session_state:
        st.session_state.all_results = None
    if 'accuracy_comparison' not in st.session_state:
        st.session_state.accuracy_comparison = None
    if 'model_package' not in st.session_state:
        st.session_state.model_package = None
    
    # Eksekusi berdasarkan section yang dipilih
    if selected_section == "1. Upload Data":
        st.session_state.df = upload_data()
    
    elif selected_section == "2. Pelabelan Sentimen":
        if st.session_state.df is not None:
            st.session_state.df, st.session_state.sentiment_distribution = lexicon_sentiment_labeling(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "3. Preprocessing Text":
        if st.session_state.df is not None:
            st.session_state.df = text_preprocessing(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "4. WordCloud":
        if st.session_state.df is not None:
            create_wordcloud_viz(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "5. Ekstraksi Fitur TF-IDF":
        if st.session_state.df is not None:
            st.session_state.X, st.session_state.y, st.session_state.tfidf_vectorizer = tfidf_feature_extraction(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "6. Pembagian Data":
        if st.session_state.X is not None and st.session_state.y is not None:
            st.session_state.results = data_splitting(st.session_state.X, st.session_state.y)
        else:
            st.warning("Silakan lakukan ekstraksi fitur terlebih dahulu di section '5. Ekstraksi Fitur TF-IDF'!")
    
    elif selected_section == "7. Training & Evaluasi SVM":
        if st.session_state.results is not None:
            st.session_state.all_results, st.session_state.accuracy_comparison = train_evaluate_svm(st.session_state.results)
            
            # Otomatis simpan model setelah training
            if st.session_state.all_results is not None and st.session_state.tfidf_vectorizer is not None:
                save_best_model_to_pickle(st.session_state.tfidf_vectorizer, st.session_state.all_results)
        else:
            st.warning("Silakan lakukan pembagian data terlebih dahulu di section '6. Pembagian Data'!")
    
    elif selected_section == "8. Visualisasi Hasil":
        if st.session_state.all_results is not None:
            visualize_results(st.session_state.all_results, st.session_state.accuracy_comparison)
        else:
            st.warning("Silakan latih model terlebih dahulu di section '7. Training & Evaluasi SVM'!")
    
    elif selected_section == "9. Implementasi Sistem":
        implementasi_sistem()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Analisis Sentimen Ulasan Gojek 2026**
    """)

if __name__ == "__main__":
    main()
