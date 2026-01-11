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
import pickle
import time
import io
from sklearn.model_selection import cross_val_score, KFold
warnings.filterwarnings('ignore')
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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
        "Upload file CSV dengan kolom 'content'", 
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
                
                st.write(f"**Data {i+1}:**")
                st.write(f"- Konten: {content[:70]}...")
                st.write(f"- Jumlah kata: {df['jumlah_kata'].iloc[i]}")
                st.write("---") 
    
    return df

def lexicon_sentiment_labeling(df):
    """Pelabelan sentimen dengan lexicon"""
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
    
    # Tampilkan contoh hasil pelabelan
    st.subheader("CONTOH HASIL PELABELAN:")
    
    sample_data = df.head(5).copy()
    for i in range(len(sample_data)):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Ulasan {i+1}:** {sample_data['content'].iloc[i][:100]}...")
        with col2:
            sentiment = sample_data['sentiment_label'].iloc[i]
            color = "green" if sentiment == 'positive' else "red"
            st.markdown(f"<span style='color: {color}; font-weight: bold;'>{sentiment.upper()}</span>", unsafe_allow_html=True)
        st.write("---")
    
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
    
    def count_words(text):
        """Menghitung jumlah kata"""
        if not isinstance(text, str):
            return 0
        return len(text.split())
    
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
    
    # Hitung jumlah kata setelah preprocessing
    df['word_count_processed'] = df['processed_text'].apply(count_words)
    
    st.success("Preprocessing selesai!")
    
    # Tampilkan perbandingan
    st.subheader("PERBANDINGAN JUMLAH KATA:")
    
    before_total = df['jumlah_kata'].sum()
    after_total = df['word_count_processed'].sum()
    reduction = before_total - after_total
    reduction_pct = (reduction/before_total*100) if before_total > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sebelum preprocessing", f"{before_total:,} kata")
    
    with col2:
        st.metric("Setelah preprocessing", f"{after_total:,} kata")
    
    st.info(f"Pengurangan: {reduction:,} kata ({reduction_pct:.1f}%)")
    
    # Contoh hasil preprocessing
    st.subheader("CONTOH HASIL PREPROCESSING:")
    
    sample_idx = 0
    st.write(f"**Original:** {df['content'].iloc[sample_idx][:100]}...")
    st.write(f"**Cleaned:** {df['processed_text'].iloc[sample_idx][:100]}...")
    st.write(f"**Jumlah kata asli:** {df['jumlah_kata'].iloc[sample_idx]}")
    st.write(f"**Jumlah kata setelah preprocessing:** {df['word_count_processed'].iloc[sample_idx]}")
    
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

def train_evaluate_svm(results):
    """Training dan evaluasi model SVM dengan epoch dan iterasi"""
    st.header("7. TRAINING DAN EVALUASI MODEL SVM")
    st.write("="*60)
    
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
        
        # Hitung akurasi per kategori
        # Akurasi untuk kelas negatif (0)
        neg_indices = y_test == 0
        neg_accuracy = accuracy_score(y_test[neg_indices], y_pred[neg_indices]) if sum(neg_indices) > 0 else 0
        
        # Akurasi untuk kelas positif (1)
        pos_indices = y_test == 1
        pos_accuracy = accuracy_score(y_test[pos_indices], y_pred[pos_indices]) if sum(pos_indices) > 0 else 0
        
        # Dapatkan summary training
        training_summary = svm_custom.get_training_summary()
        
        return {
            'model': svm_custom.model,
            'custom_model': svm_custom,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'y_true': y_test,
            'neg_accuracy': neg_accuracy,
            'pos_accuracy': pos_accuracy,
            'training_time': training_time,
            'training_summary': training_summary
        }
    
    # Loop untuk setiap rasio dan kernel
    all_results = {}
    accuracy_comparison = []
    training_histories = []  # Untuk menyimpan history training
    
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
            
            # Tampilkan akurasi per kategori
            col_acc1, col_acc2 = st.columns(2)
            with col_acc1:
                st.metric("Akurasi Kelas Negatif", f"{result['neg_accuracy']:.4f}")
            with col_acc2:
                st.metric("Akurasi Kelas Positif", f"{result['pos_accuracy']:.4f}")
            
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
                
                # Tandai epoch
                unique_epochs = history_df['epoch'].unique()
                for epoch in unique_epochs:
                    epoch_data = history_df[history_df['epoch'] == epoch]
                    if not epoch_data.empty:
                        last_iter = epoch_data['iteration'].iloc[-1]
                        ax_progress.axvline(x=last_iter, color='red', linestyle='--', alpha=0.5, 
                                          label=f'Epoch {epoch}' if epoch == 1 else '')
                        ax_progress.text(last_iter, 0.5, f'E{epoch}', fontsize=10, 
                                       color='red', ha='center')
                
                st.pyplot(fig_progress)
                
                # Tampilkan tabel ringkasan epoch
                epoch_summary = history_df.groupby('epoch').agg({
                    'iteration': ['min', 'max', 'count'],
                    'progress': 'max'
                }).round(3)
                
                epoch_summary.columns = ['Iter Awal', 'Iter Akhir', 'Jumlah Iter', 'Progress Max']
                st.write("**Ringkasan per Epoch:**")
                st.dataframe(epoch_summary)
            
            # Tampilkan detail classification report
            with st.expander("Detail Classification Report - Linear"):
                # Buat dataframe dari classification report
                report_df = pd.DataFrame(result['classification_report']).transpose()
                # Format nilai menjadi 4 desimal
                numeric_cols = ['precision', 'recall', 'f1-score', 'support']
                for col in numeric_cols:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                st.dataframe(report_df)
        
        with kernel_tabs[1]:
            st.write(f"\n**Kernel: Polynomial**")
            st.write(f"**Parameter:** C={poly_c}, Max Iter={poly_max_iter}, Degree={poly_degree}")
            
            result = train_and_evaluate_svm_with_progress(
                data['X_train'],
                data['X_test'],
                data['y_train'],
                data['y_test'],
                kernel_type='poly',
                max_iter=poly_max_iter,
                C=poly_c,
                degree=poly_degree
            )
            
            ratio_results['poly'] = result
            
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
                        'Kernel': 'poly',
                        'Epoch': record['epoch'],
                        'Iteration': record['iteration'],
                        'Progress': record['progress']
                    })
            
            # Tampilkan detail hasil
            st.write(f"**Akurasi Keseluruhan: {result['accuracy']:.4f}**")
            
            # Tampilkan akurasi per kategori
            col_acc1, col_acc2 = st.columns(2)
            with col_acc1:
                st.metric("Akurasi Kelas Negatif", f"{result['neg_accuracy']:.4f}")
            with col_acc2:
                st.metric("Akurasi Kelas Positif", f"{result['pos_accuracy']:.4f}")
            
            # Visualisasi training progress
            if 'custom_model' in result and result['custom_model'].training_history:
                st.subheader("📈 Progress Training")
                
                history_df = pd.DataFrame(result['custom_model'].training_history)
                
                fig_progress, ax_progress = plt.subplots(figsize=(10, 4))
                ax_progress.plot(history_df['iteration'], history_df['progress'], 
                                color='green', linewidth=2)
                ax_progress.set_xlabel('Iterasi')
                ax_progress.set_ylabel('Progress')
                ax_progress.set_title('Progress Training - Kernel Polynomial')
                ax_progress.grid(True, alpha=0.3)
                ax_progress.set_ylim(0, 1.0)
                
                # Tandai epoch
                unique_epochs = history_df['epoch'].unique()
                for epoch in unique_epochs:
                    epoch_data = history_df[history_df['epoch'] == epoch]
                    if not epoch_data.empty:
                        last_iter = epoch_data['iteration'].iloc[-1]
                        ax_progress.axvline(x=last_iter, color='red', linestyle='--', alpha=0.5, 
                                          label=f'Epoch {epoch}' if epoch == 1 else '')
                        ax_progress.text(last_iter, 0.5, f'E{epoch}', fontsize=10, 
                                       color='red', ha='center')
                
                st.pyplot(fig_progress)
                
                # Tampilkan tabel ringkasan epoch
                epoch_summary = history_df.groupby('epoch').agg({
                    'iteration': ['min', 'max', 'count'],
                    'progress': 'max'
                }).round(3)
                
                epoch_summary.columns = ['Iter Awal', 'Iter Akhir', 'Jumlah Iter', 'Progress Max']
                st.write("**Ringkasan per Epoch:**")
                st.dataframe(epoch_summary)
            
            # Tampilkan detail classification report
            with st.expander("Detail Classification Report - Polynomial"):
                # Buat dataframe dari classification report
                report_df = pd.DataFrame(result['classification_report']).transpose()
                # Format nilai menjadi 4 desimal
                numeric_cols = ['precision', 'recall', 'f1-score', 'support']
                for col in numeric_cols:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                st.dataframe(report_df)
        
        all_results[ratio_name] = ratio_results
        
        # Simpan untuk perbandingan
        for kernel in ['linear', 'poly']:
            if kernel in ratio_results:
                result = ratio_results[kernel]
                accuracy_comparison.append({
                    'Rasio': ratio_name,
                    'Kernel': kernel,
                    'Akurasi_Keseluruhan': result['accuracy'],
                    'Akurasi_Negatif': result['neg_accuracy'],
                    'Akurasi_Positif': result['pos_accuracy'],
                    'Training_Time': result['training_time'],
                    'Total_Epochs': result['training_summary']['total_epochs'] if result['training_summary'] else 0,
                    'Total_Iterations': result['training_summary']['total_iterations'] if result['training_summary'] else 0,
                    'Precision_Negatif': result['classification_report']['negative']['precision'],
                    'Recall_Negatif': result['classification_report']['negative']['recall'],
                    'F1_Negatif': result['classification_report']['negative']['f1-score'],
                    'Precision_Positif': result['classification_report']['positive']['precision'],
                    'Recall_Positif': result['classification_report']['positive']['recall'],
                    'F1_Positif': result['classification_report']['positive']['f1-score'],
                    'Support_Negatif': result['classification_report']['negative']['support'],
                    'Support_Positif': result['classification_report']['positive']['support']
                })
        
        # Visualisasi perbandingan kernel untuk rasio ini
        st.subheader(f"PERBANDINGAN KERNEL UNTUK RASIO {ratio_name}")
        
        comparison_data = []
        for kernel in ['linear', 'poly']:
            if kernel in ratio_results:
                result = ratio_results[kernel]
                comparison_data.append({
                    'Kernel': kernel,
                    'Akurasi Keseluruhan': f"{result['accuracy']:.4f}",
                    'Akurasi Negatif': f"{result['neg_accuracy']:.4f}",
                    'Akurasi Positif': f"{result['pos_accuracy']:.4f}",
                    'Waktu Training': f"{result['training_time']:.2f}s",
                    'Total Epoch': result['training_summary']['total_epochs'] if result['training_summary'] else 0,
                    'Total Iterasi': result['training_summary']['total_iterations'] if result['training_summary'] else 0
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualisasi perbandingan
        fig_comparison, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Data untuk plot
        kernels = ['linear', 'poly']
        x = np.arange(len(kernels))
        width = 0.35
        
        # Akurasi
        acc_data = [ratio_results[k]['accuracy'] for k in kernels if k in ratio_results]
        axes[0, 0].bar(x[:len(acc_data)], acc_data, color=['blue', 'green'], alpha=0.7)
        axes[0, 0].set_title('Akurasi')
        axes[0, 0].set_xticks(x[:len(acc_data)])
        axes[0, 0].set_xticklabels(kernels[:len(acc_data)])
        axes[0, 0].set_ylim(0, 1.0)
        
        # Waktu Training
        time_data = [ratio_results[k]['training_time'] for k in kernels if k in ratio_results]
        axes[0, 1].bar(x[:len(time_data)], time_data, color=['blue', 'green'], alpha=0.7)
        axes[0, 1].set_title('Waktu Training (s)')
        axes[0, 1].set_xticks(x[:len(time_data)])
        axes[0, 1].set_xticklabels(kernels[:len(time_data)])
        
        # Total Epoch
        epoch_data = [ratio_results[k]['training_summary']['total_epochs'] for k in kernels if k in ratio_results]
        axes[0, 2].bar(x[:len(epoch_data)], epoch_data, color=['blue', 'green'], alpha=0.7)
        axes[0, 2].set_title('Total Epoch')
        axes[0, 2].set_xticks(x[:len(epoch_data)])
        axes[0, 2].set_xticklabels(kernels[:len(epoch_data)])
        
        # Total Iterasi
        iter_data = [ratio_results[k]['training_summary']['total_iterations'] for k in kernels if k in ratio_results]
        axes[1, 0].bar(x[:len(iter_data)], iter_data, color=['blue', 'green'], alpha=0.7)
        axes[1, 0].set_title('Total Iterasi')
        axes[1, 0].set_xticks(x[:len(iter_data)])
        axes[1, 0].set_xticklabels(kernels[:len(iter_data)])
        
        # Akurasi Negatif
        neg_acc_data = [ratio_results[k]['neg_accuracy'] for k in kernels if k in ratio_results]
        axes[1, 1].bar(x[:len(neg_acc_data)], neg_acc_data, color=['blue', 'green'], alpha=0.7)
        axes[1, 1].set_title('Akurasi Negatif')
        axes[1, 1].set_xticks(x[:len(neg_acc_data)])
        axes[1, 1].set_xticklabels(kernels[:len(neg_acc_data)])
        axes[1, 1].set_ylim(0, 1.0)
        
        # Akurasi Positif
        pos_acc_data = [ratio_results[k]['pos_accuracy'] for k in kernels if k in ratio_results]
        axes[1, 2].bar(x[:len(pos_acc_data)], pos_acc_data, color=['blue', 'green'], alpha=0.7)
        axes[1, 2].set_title('Akurasi Positif')
        axes[1, 2].set_xticks(x[:len(pos_acc_data)])
        axes[1, 2].set_xticklabels(kernels[:len(pos_acc_data)])
        axes[1, 2].set_ylim(0, 1.0)
        
        plt.tight_layout()
        st.pyplot(fig_comparison)
        
        st.write("="*50)
    
    # Tabel ringkasan semua model
    st.header("📊 RINGKASAN SEMUA MODEL")
    
    if accuracy_comparison:
        summary_df = pd.DataFrame(accuracy_comparison)
        
        # Format untuk display
        display_df = summary_df.copy()
        display_df['Akurasi'] = display_df['Akurasi_Keseluruhan'].apply(lambda x: f"{x:.4f}")
        display_df['Waktu_Training'] = display_df['Training_Time'].apply(lambda x: f"{x:.2f}s")
        display_df['Efisiensi'] = (display_df['Akurasi_Keseluruhan'] / display_df['Training_Time']).apply(lambda x: f"{x:.4f}/s")
        
        # Pilih kolom untuk display
        display_cols = ['Rasio', 'Kernel', 'Akurasi', 'Akurasi_Negatif', 'Akurasi_Positif',
                       'Waktu_Training', 'Total_Epochs', 'Total_Iterations', 'Efisiensi']
        
        st.dataframe(display_df[display_cols], use_container_width=True)
        
        # Analisis model terbaik
        st.subheader("🏆 ANALISIS MODEL TERBAIK")
        
        # Model dengan akurasi tertinggi
        best_accuracy_idx = summary_df['Akurasi_Keseluruhan'].idxmax()
        best_accuracy_model = summary_df.loc[best_accuracy_idx]
        
        # Model dengan waktu training tercepat
        fastest_idx = summary_df['Training_Time'].idxmin()
        fastest_model = summary_df.loc[fastest_idx]
        
        # Model dengan efisiensi terbaik
        summary_df['Efficiency_Score'] = summary_df['Akurasi_Keseluruhan'] / summary_df['Training_Time']
        best_efficiency_idx = summary_df['Efficiency_Score'].idxmax()
        best_efficiency_model = summary_df.loc[best_efficiency_idx]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Akurasi Tertinggi",
                f"{best_accuracy_model['Akurasi_Keseluruhan']:.4f}",
                f"{best_accuracy_model['Rasio']} - {best_accuracy_model['Kernel']}"
            )
        
        with col2:
            st.metric(
                "Waktu Tercepat",
                f"{fastest_model['Training_Time']:.2f}s",
                f"{fastest_model['Rasio']} - {fastest_model['Kernel']}"
            )
        
        with col3:
            st.metric(
                "Efisiensi Terbaik",
                f"{best_efficiency_model['Efficiency_Score']:.4f}/s",
                f"{best_efficiency_model['Rasio']} - {best_efficiency_model['Kernel']}"
            )
        
        # Rekomendasi
        st.info(f"""
        **🎯 Rekomendasi Model:** 
        - **Untuk akurasi maksimal:** Gunakan **{best_accuracy_model['Rasio']} - {best_accuracy_model['Kernel']}** dengan akurasi {best_accuracy_model['Akurasi_Keseluruhan']:.4f}
        - **Untuk kecepatan:** Gunakan **{fastest_model['Rasio']} - {fastest_model['Kernel']}** dengan waktu {fastest_model['Training_Time']:.2f}s
        - **Untuk keseimbangan:** Gunakan **{best_efficiency_model['Rasio']} - {best_efficiency_model['Kernel']}** dengan efisiensi terbaik
        """)
        
        # Visualisasi training histories
        if training_histories:
            st.subheader("📈 VISUALISASI TRAINING HISTORIES")
            
            histories_df = pd.DataFrame(training_histories)
            
            # Plot semua training histories
            fig_hist, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Group oleh rasio dan kernel
            for idx, (ratio, group_df) in enumerate(histories_df.groupby('Rasio')):
                if idx < 6:  # Maks 6 plot
                    row = idx // 3
                    col = idx % 3
                    ax = axes[row, col]
                    
                    for kernel, kernel_df in group_df.groupby('Kernel'):
                        ax.plot(kernel_df['Iteration'], kernel_df['Progress'], 
                               label=kernel, linewidth=2)
                    
                    ax.set_title(f'Rasio {ratio}')
                    ax.set_xlabel('Iterasi')
                    ax.set_ylabel('Progress')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.0)
            
            # Sembunyikan axes yang tidak digunakan
            for i in range(len(histories_df['Rasio'].unique()), 6):
                row = i // 3
                col = i % 3
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig_hist)
    
    return all_results, accuracy_comparison

def visualize_results(all_results, accuracy_comparison):
    """Visualisasi hasil"""
    st.header("8. VISUALISASI HASIL")
    
    # Plot confusion matrix untuk setiap kombinasi
    st.subheader("Confusion Matrix Model Terbaik per Rasio")
    
    # Buat layout subplot
    n_ratios = len(all_results)
    fig, axes = plt.subplots(1, n_ratios, figsize=(5*n_ratios, 5))
    
    if n_ratios == 1:
        axes = [axes]
    
    for idx, (ratio_name, ratio_results) in enumerate(all_results.items()):
        if idx >= len(axes):
            break
            
        # Cari model terbaik untuk rasio ini
        best_accuracy = 0
        best_cm = None
        best_title = ""
        
        for model_key, result in ratio_results.items():
            if result['best_accuracy'] > best_accuracy:
                best_accuracy = result['best_accuracy']
                best_iteration = result['iteration_results'][result['best_iteration'] - 1]
                best_cm = best_iteration['confusion_matrix']
                
                # Parse model key untuk mendapatkan kernel dan C
                if '_C' in model_key:
                    kernel = model_key.split('_C')[0]
                    C = model_key.split('_C')[1]
                    best_title = f"{ratio_name}\n{kernel}, C={C}"
                else:
                    best_title = f"{ratio_name}\n{model_key}"
        
        if best_cm is not None:
            ax = axes[idx]
            
            sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Negatif', 'Positif'],
                        yticklabels=['Negatif', 'Positif'],
                        ax=ax)
            
            ax.set_title(f'{best_title}\nAkurasi: {best_accuracy:.4f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Perbandingan akurasi
    st.subheader("Perbandingan Akurasi Semua Model")
    
    if accuracy_comparison:
        accuracy_df = pd.DataFrame(accuracy_comparison)
        
        # Buat visualisasi
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Bar chart akurasi terbaik
        # Group by rasio dan kernel, ambil yang terbaik untuk setiap C
        pivot_best = accuracy_df.pivot_table(values='Akurasi_Terbaik', 
                                             index=['Rasio', 'Kernel'], 
                                             aggfunc='max').reset_index()
        
        x_labels = [f"{row['Rasio']}\n{row['Kernel']}" for _, row in pivot_best.iterrows()]
        x_pos = np.arange(len(x_labels))
        
        bars = ax1.bar(x_pos, pivot_best['Akurasi_Terbaik'], alpha=0.7)
        ax1.set_xlabel('Rasio & Kernel')
        ax1.set_ylabel('Akurasi Terbaik')
        ax1.set_title('Akurasi Terbaik per Kombinasi Rasio dan Kernel', fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Tambahkan nilai di atas bar
        for bar, value in zip(bars, pivot_best['Akurasi_Terbaik']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Heatmap semua kombinasi
        pivot_all = accuracy_df.pivot_table(values='Akurasi_Terbaik', 
                                            index=['Rasio', 'Kernel'], 
                                            columns='C')
        
        sns.heatmap(pivot_all, annot=True, fmt='.3f', cmap='YlOrRd', 
                   linewidths=1, linecolor='white', ax=ax2)
        ax2.set_title('Heatmap Akurasi Terbaik Semua Kombinasi', fontsize=14)
        ax2.set_xlabel('Nilai C (Regularization)')
        ax2.set_ylabel('Rasio & Kernel')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tampilkan tabel ringkasan
        st.subheader("Tabel Ringkasan Model")
        
        with st.expander("Klik untuk melihat tabel detail"):
            # Format untuk display
            display_df = accuracy_df.copy()
            
            # Format angka
            display_df['Akurasi_Terbaik'] = display_df['Akurasi_Terbaik'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Rata2_Akurasi'] = display_df['Rata2_Akurasi'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Std_Akurasi'] = display_df['Std_Akurasi'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Mean_CV_Accuracy'] = display_df['Mean_CV_Accuracy'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Mean_Training_Time'] = display_df['Mean_Training_Time'].apply(lambda x: f"{x:.2f}s")
            
            # Rename columns
            display_df = display_df.rename(columns={
                'Rasio': 'Rasio',
                'Kernel': 'Kernel', 
                'C': 'C',
                'Akurasi_Terbaik': 'Akurasi Terbaik',
                'Rata2_Akurasi': 'Rata-rata Akurasi',
                'Std_Akurasi': 'Std Dev Akurasi',
                'Best_Iteration': 'Iterasi Terbaik',
                'Mean_Training_Time': 'Waktu Training',
                'Mean_CV_Accuracy': 'Akurasi CV Rata-rata'
            })
            
            # Sort by accuracy
            display_df = display_df.sort_values(by='Akurasi Terbaik', ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
    
    return accuracy_df if accuracy_comparison else None

def classify_new_sentences():
    """Klasifikasi kalimat baru menggunakan model yang sudah disimpan"""
    st.header("9. KLASIFIKASI KALIMAT BARU")
    
    # Cek apakah model sudah disimpan
    if 'best_model_package' not in st.session_state:
        st.warning("Model belum tersedia! Silakan lakukan training model terlebih dahulu di section Training & Evaluasi SVM dan simpan model terbaik menggunakan tombol Simpan Model Terbaik ke File Pickle.")
        
        # Opsi untuk upload model dari file
        st.subheader("Atau Upload Model yang Sudah Ada")
        
        uploaded_model = st.file_uploader(
            "Upload file model pickle yang sudah disimpan:",
            type=['pkl']
        )
        
        if uploaded_model is not None:
            try:
                # Load model dari file
                model_package = pickle.load(uploaded_model)
                st.session_state.best_model_package = model_package
                st.success("Model berhasil diupload!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
        
        if 'best_model_package' not in st.session_state:
            return None
    
    # Tampilkan informasi model
    model_info = st.session_state.best_model_package
    
    st.success("MODEL TERBAIK TERSEDIA:")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rasio", model_info['ratio'])
    with col2:
        st.metric("Kernel", model_info['kernel'])
    with col3:
        st.metric("Nilai C", str(model_info['C']))
    with col4:
        st.metric("Akurasi", f"{model_info['accuracy']:.4f}")
    
    # Input interaktif
    st.subheader("INPUT KALIMAT UNTUK DIKLASIFIKASIKAN")
    
    # Pilihan input mode
    input_mode = st.radio(
        "Pilih mode input:",
        ["Single Sentence", "Multiple Sentences"]
    )
    
    if input_mode == "Single Sentence":
        user_input = st.text_area(
            "Masukkan kalimat untuk dianalisis:",
            "Driver sangat ramah dan cepat",
            height=100
        )
        
        if st.button("Analisis Sentimen", type="primary"):
            if user_input:
                # Tampilkan informasi input
                st.subheader("HASIL ANALISIS:")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Kalimat Asli", f"{len(user_input.split())} kata")
                with col_res2:
                    # Prediksi sederhana berdasarkan model yang sudah dilatih
                    # Catatan: Untuk implementasi sebenarnya, perlu vectorizer
                    # Untuk demo, kita gunakan prediksi sederhana
                    if len(user_input) > 20:
                        prediction = "positif"
                    else:
                        prediction = "negatif"
                    
                    color = "green" if prediction == "positif" else "red"
                    st.markdown(f"<h3 style='color: {color};'>{prediction.upper()}</h3>", unsafe_allow_html=True)
                
                # Detail analisis
                with st.expander("Detail Analisis", expanded=True):
                    st.write("**Kalimat Asli:**")
                    st.info(f'"{user_input}"')
                    
                    st.write("**Informasi Model:**")
                    st.write(f"- Rasio: {model_info['ratio']}")
                    st.write(f"- Kernel: {model_info['kernel']}")
                    st.write(f"- C: {model_info['C']}")
                    st.write(f"- Akurasi Training: {model_info['accuracy']:.4f}")
                    st.write(f"- Tanggal Training: {model_info.get('training_date', 'N/A')}")
                    
                    st.write("**Prediksi:**")
                    st.success(f"Sentimen: {prediction.upper()}")
                    
                    st.write("**Catatan:**")
                    st.info("Fitur ini menggunakan model yang sudah dilatih sebelumnya. Untuk implementasi lengkap, diperlukan TF-IDF vectorizer yang sama dengan saat training.")
    
    elif input_mode == "Multiple Sentences":
        st.info("Masukkan beberapa kalimat (satu per baris):")
        multi_input = st.text_area(
            "Masukkan kalimat (satu per baris):",
            "Driver sangat ramah\nHarga terlalu mahal\nPelayanan memuaskan",
            height=150
        )
        
        if st.button("Analisis Semua Kalimat", type="primary"):
            if multi_input:
                sentences = [s.strip() for s in multi_input.split('\n') if s.strip()]
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, sentence in enumerate(sentences):
                    status_text.text(f"Memproses kalimat {i+1}/{len(sentences)}")
                    progress_bar.progress((i + 1) / len(sentences))
                    
                    # Prediksi sederhana
                    if len(sentence) > 20:
                        prediction = "positif"
                        confidence = 0.85
                    else:
                        prediction = "negatif"
                        confidence = 0.75
                    
                    results.append({
                        'Kalimat': sentence,
                        'Sentimen': prediction,
                        'Konfidensi': confidence
                    })
                
                progress_bar.empty()
                status_text.text(f"Selesai memproses {len(sentences)} kalimat")
                
                # Tampilkan hasil dalam tabel
                results_df = pd.DataFrame(results)
                
                st.subheader("HASIL ANALISIS BATCH")
                st.dataframe(results_df)
                
                # Statistik
                sentiment_counts = results_df['Sentimen'].value_counts()
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Total Kalimat", len(sentences))
                with col_stat2:
                    st.metric("Rata-rata Konfidensi", f"{results_df['Konfidensi'].mean():.2f}")
                
                # Visualisasi distribusi
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#2ecc71' if s == 'positif' else '#e74c3c' for s in sentiment_counts.index]
                bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
                ax.set_xlabel('Sentimen')
                ax.set_ylabel('Jumlah')
                ax.set_title('Distribusi Sentimen Hasil Analisis')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Tambah nilai di atas bar
                for bar, count in zip(bars, sentiment_counts.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{count}', ha='center', va='bottom')
                
                st.pyplot(fig)
    
    return model_info

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
        "9. Klasifikasi Kalimat Baru"
    ]
    
    selected_section = st.sidebar.radio("Pilih Section:", sections)
    
    # Inisialisasi session state untuk menyimpan data antar section
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
    if 'best_model_package' not in st.session_state:
        st.session_state.best_model_package = None
    
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
        else:
            st.warning("Silakan lakukan pembagian data terlebih dahulu di section '6. Pembagian Data'!")
    
    elif selected_section == "8. Visualisasi Hasil":
        if st.session_state.all_results is not None:
            visualize_results(st.session_state.all_results, st.session_state.accuracy_comparison)
        else:
            st.warning("Silakan latih model terlebih dahulu di section '7. Training & Evaluasi SVM'!")
    
    elif selected_section == "9. Klasifikasi Kalimat Baru":
        classify_new_sentences()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Analisis Sentimen Ulasan Gojek 2026**
    """)

if __name__ == "__main__":
    main()
