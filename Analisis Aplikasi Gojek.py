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
import pickle
import json
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
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Analisis Sentimen Ulasan Gojek")
    st.markdown("---")

def load_data():
    """Fungsi untuk memuat data"""
    st.header("1. LOAD DATA (8000 DATA)")
    
    # Buat data contoh 8000 baris
    np.random.seed(42)
    
    # Contoh kalimat positif
    positive_samples = [
        "aplikasi gojek sangat bagus dan mudah digunakan",
        "driver ramah dan cepat sampai tujuan",
        "pelayanan memuaskan harga terjangkau",
        "mantap banget recommended untuk semua",
        "proses pesan cepat tidak ada kendala",
        "aplikasi user friendly interface menarik",
        "driver sopan dan mengutamakan keselamatan",
        "fitur lengkap sangat membantu sehari-hari",
        "responsif dan mudah dioperasikan",
        "pengalaman menggunakan sangat menyenangkan"
    ]
    
    # Contoh kalimat negatif
    negative_samples = [
        "aplikasi sering error tidak stabil",
        "driver lambat dan tidak profesional",
        "pelayanan buruk sangat mengecewakan",
        "harga mahal tidak sesuai pelayanan",
        "sering terjadi masalah teknis",
        "customer service tidak responsif",
        "waiting time terlalu lama",
        "aplikasi lemot dan sering crash",
        "driver tidak tahu jalan tersesat",
        "pengalaman buruk tidak akan pakai lagi"
    ]
    
    # Generate 8000 data
    n_samples = 8000
    data_content = []
    data_sentiment = []
    
    for i in range(n_samples):
        if np.random.random() > 0.4:  # 60% positif, 40% negatif
            # Pilih random positive sample
            base_text = np.random.choice(positive_samples)
            sentiment = 'positif'
        else:
            # Pilih random negative sample
            base_text = np.random.choice(negative_samples)
            sentiment = 'negatif'
        
        # Tambah variasi teks
        variations = [
            "", "sangat", "sekali", "banget", "saya rasa", "menurut saya",
            "pengalaman pribadi", "baru saja mencoba", "setelah update"
        ]
        variation = np.random.choice(variations)
        
        if variation:
            content = f"{variation} {base_text}"
        else:
            content = base_text
            
        data_content.append(content)
        data_sentiment.append(sentiment)
    
    # Buat DataFrame
    df = pd.DataFrame({
        'content': data_content,
        'sentimen': data_sentiment
    })
    
    st.success(f"✅ Data berhasil dimuat: {len(df)} baris")
    
    # Hitung statistik dasar
    df['jumlah_kata'] = df['content'].apply(lambda x: len(str(x).split()))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", f"{len(df):,}")
    with col2:
        st.metric("Total Kata", f"{df['jumlah_kata'].sum():,}")
    with col3:
        st.metric("Rata-rata Kata", f"{df['jumlah_kata'].mean():.1f}")
    
    # Tampilkan distribusi sentimen awal
    if 'sentimen' in df.columns:
        sentiment_counts = df['sentimen'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Jumlah')
        ax.set_title('Distribusi Sentimen Awal')
        
        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + max(sentiment_counts.values)*0.01, str(v), ha='center')
        
        st.pyplot(fig)
    
    # Tampilkan preview data
    with st.expander("Preview Data (5 baris pertama)"):
        st.dataframe(df.head())
    
    return df

def analyze_word_count(df):
    """Analisis jumlah kata"""
    st.header("2. ANALISIS JUMLAH KATA DARI 8000 ULASAN")
    
    # Fungsi untuk menghitung jumlah kata
    def count_words(text):
        if not isinstance(text, str):
            return 0
        return len(text.split())
    
    # Hitung jumlah kata untuk semua 8000 ulasan
    df['word_count'] = df['content'].apply(count_words)
    
    # Tampilkan statistik
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Kata", f"{df['word_count'].sum():,}")
    with col2:
        st.metric("Rata-rata", f"{df['word_count'].mean():.1f}")
    with col3:
        st.metric("Median", f"{df['word_count'].median():.1f}")
    
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Minimal", f"{df['word_count'].min()}")
    with col5:
        st.metric("Maksimal", f"{df['word_count'].max()}")
    
    # Visualisasi
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(df['word_count'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['word_count'].mean(), color='red', linestyle='dashed', 
                    linewidth=2, label=f'Rata-rata: {df["word_count"].mean():.1f}')
    axes[0].set_xlabel('Jumlah Kata')
    axes[0].set_ylabel('Frekuensi')
    axes[0].set_title('Distribusi Jumlah Kata per Ulasan (8000 data)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df['word_count'])
    axes[1].set_ylabel('Jumlah Kata')
    axes[1].set_title('Box Plot Jumlah Kata')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return df

def lexicon_sentiment_labeling(df):
    """Pelabelan sentimen dengan lexicon"""
    st.header("3. PELABELAN SENTIMEN MENGGUNAKAN LEXICON")
    
    # Lexicon yang diperluas untuk Bahasa Indonesia
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
        'handal', 'andal', 'terpercaya', 'amanah', 'solutif', 'efektif'
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
        'mengecewa', 'menyusahkan', 'merepotkan', 'menghambat', 'menyakitkan'
    ]
    
    # Fungsi untuk pelabelan sentimen (TANPA KATEGORI NETRAL)
    def lexicon_sentiment_analysis_binary(text):
        if not isinstance(text, str):
            return 'neutral'

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Jika sama, gunakan heuristik tambahan
        if positive_count == negative_count:
            # Cek kata kunci yang sangat kuat
            strong_positive = any(word in text_lower for word in ['sangat baik', 'sangat bagus', 'luar biasa', 'terbaik'])
            strong_negative = any(word in text_lower for word in ['sangat buruk', 'sangat jelek', 'parah sekali', 'penipu'])

            if strong_positive:
                return 'positive'
            elif strong_negative:
                return 'negative'
            else:
                # Default ke positif jika netral
                return 'positive'

        return 'positive' if positive_count > negative_count else 'negative'
    
    # Terapkan pelabelan BINARY (hanya positif/negatif)
    df['sentiment_label'] = df['content'].apply(lexicon_sentiment_analysis_binary)
    
    # HAPUS jika ada yang masih netral (tidak seharusnya ada)
    df = df[df['sentiment_label'].isin(['positive', 'negative'])].copy()
    
    # Hitung distribusi sentimen
    sentiment_distribution = df['sentiment_label'].value_counts()
    total_data = len(df)
    
    # Tampilkan statistik
    st.success(f"✅ Pelabelan selesai: {total_data} ulasan")
    
    col1, col2 = st.columns(2)
    with col1:
        positif_count = sentiment_distribution.get('positive', 0)
        positif_pct = (positif_count/total_data*100) if total_data > 0 else 0
        st.metric("Positif", f"{positif_count:,}", f"{positif_pct:.1f}%")
    with col2:
        negatif_count = sentiment_distribution.get('negative', 0)
        negatif_pct = (negatif_count/total_data*100) if total_data > 0 else 0
        st.metric("Negatif", f"{negatif_count:,}", f"{negatif_pct:.1f}%")
    
    # Analisis jumlah kata per kategori
    positive_word_counts = df[df['sentiment_label'] == 'positive']['word_count']
    negative_word_counts = df[df['sentiment_label'] == 'negative']['word_count']
    
    # Visualisasi
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
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
    
    # Box plot jumlah kata per sentimen
    box_data = [positive_word_counts, negative_word_counts]
    axes[2].boxplot(box_data, labels=['Positif', 'Negatif'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'))
    axes[2].set_xlabel('Sentimen')
    axes[2].set_ylabel('Jumlah Kata')
    axes[2].set_title('Distribusi Jumlah Kata per Sentimen')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("Detail Analisis Kata per Kategori"):
        st.write("**POSITIF:**")
        st.write(f"- Total kata: {positive_word_counts.sum():,} kata")
        st.write(f"- Rata-rata: {positive_word_counts.mean():.1f} kata/ulasan")
        st.write(f"- Median: {positive_word_counts.median():.1f} kata")
        
        st.write("\n**NEGATIF:**")
        st.write(f"- Total kata: {negative_word_counts.sum():,} kata")
        st.write(f"- Rata-rata: {negative_word_counts.mean():.1f} kata/ulasan")
        st.write(f"- Median: {negative_word_counts.median():.1f} kata")
    
    return df, sentiment_distribution

def create_wordcloud_viz(df):
    """Visualisasi wordcloud"""
    st.header("4. WORDCLOUD VISUALIZATION")
    
    # Fungsi untuk membuat wordcloud
    def create_wordcloud(text, title, color):
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
        return len(words), len(unique_words)
    
    # Wordcloud untuk semua data
    st.subheader("Semua Ulasan")
    all_text = ' '.join(df['content'].astype(str).tolist())
    total_words, unique_words = create_wordcloud(all_text, 'WordCloud Semua Ulasan Gojek', 'steelblue')
    st.write(f"Total kata: {total_words:,} | Kata unik: {unique_words:,}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ulasan Positif")
        positive_text = ' '.join(df[df['sentiment_label'] == 'positive']['content'].astype(str).tolist())
        total_pos, unique_pos = create_wordcloud(positive_text, 'WordCloud - Ulasan Positif', 'green')
        st.write(f"Total kata: {total_pos:,} | Kata unik: {unique_pos:,}")
    
    with col2:
        st.subheader("Ulasan Negatif")
        negative_text = ' '.join(df[df['sentiment_label'] == 'negative']['content'].astype(str).tolist())
        total_neg, unique_neg = create_wordcloud(negative_text, 'WordCloud - Ulasan Negatif', 'darkred')
        st.write(f"Total kata: {total_neg:,} | Kata unik: {unique_neg:,}")

def text_preprocessing(df):
    """Preprocessing teks"""
    st.header("5. TEXT PREPROCESSING")
    
    # Inisialisasi tools
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    
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
    
    # Proses preprocessing dengan progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Memulai preprocessing...")
    # Cleaning
    df['cleaned_text'] = df['content'].apply(clean_text)
    progress_bar.progress(25)
    
    # Remove stopwords
    df['text_no_stopwords'] = df['cleaned_text'].apply(remove_stopwords)
    progress_bar.progress(50)
    
    # Tokenization
    df['tokens'] = df['text_no_stopwords'].apply(tokenize_text)
    progress_bar.progress(75)
    
    # Gabungkan token kembali menjadi string untuk TF-IDF
    df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    # Hitung jumlah kata setelah preprocessing
    df['word_count_processed'] = df['processed_text'].apply(count_words)
    progress_bar.progress(100)
    
    status_text.text("✓ Preprocessing selesai!")
    
    # Tampilkan perbandingan
    st.success("✅ Preprocessing berhasil!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sebelum Preprocessing", f"{df['word_count'].sum():,} kata")
    
    with col2:
        st.metric("Setelah Preprocessing", f"{df['word_count_processed'].sum():,} kata")
    
    reduction = df['word_count'].sum() - df['word_count_processed'].sum()
    reduction_pct = (reduction/df['word_count'].sum()*100) if df['word_count'].sum() > 0 else 0
    st.info(f"Pengurangan: {reduction:,} kata ({reduction_pct:.1f}%)")
    
    # Contoh hasil preprocessing
    with st.expander("Contoh Hasil Preprocessing"):
        sample_idx = 0
        st.write(f"**Original:** {df['content'].iloc[sample_idx][:100]}...")
        st.write(f"**Cleaned:** {df['processed_text'].iloc[sample_idx][:100]}...")
        st.write(f"**Jumlah kata asli:** {df['word_count'].iloc[sample_idx]}")
        st.write(f"**Jumlah kata setelah preprocessing:** {df['word_count_processed'].iloc[sample_idx]}")
    
    return df

def tfidf_feature_extraction(df):
    """Ekstraksi fitur TF-IDF"""
    st.header("6. EKSTRAKSI FITUR DENGAN TF-IDF")
    
    # Inisialisasi TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=3,
        max_df=0.8,
        ngram_range=(1, 2)  # Unigram dan bigram
    )
    
    # Transformasi teks menjadi vektor TF-IDF
    X = tfidf_vectorizer.fit_transform(df['processed_text'])
    y = df['sentiment_label'].map({'positive': 1, 'negative': 0})
    
    st.success(f"✓ Transformasi TF-IDF selesai!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dimensi Matriks", f"{X.shape}")
    with col2:
        st.metric("Jumlah Fitur", f"{len(tfidf_vectorizer.get_feature_names_out())}")
    
    # Tampilkan fitur teratas
    with st.expander("Top 20 Fitur berdasarkan IDF"):
        feature_names = tfidf_vectorizer.get_feature_names_out()
        idf_values = tfidf_vectorizer.idf_
        top_indices = np.argsort(idf_values)[:20]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                'Fitur': feature_names[idx],
                'IDF Score': f"{idf_values[idx]:.4f}"
            })
        
        st.dataframe(pd.DataFrame(top_features))
    
    return X, y, tfidf_vectorizer

def data_splitting(X, y):
    """Pembagian data training-testing"""
    st.header("7. PEMBAGIAN DATA TRAINING-TESTING")
    
    # Definisikan rasio
    ratios = {
        '80:20': 0.2,
        '90:10': 0.1,
        '70:30': 0.3
    }
    
    results = {}
    
    for ratio_name, test_size in ratios.items():
        st.subheader(f"Rasio: {ratio_name}")
        
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
            st.write("**Training Set:**")
            st.write(f"- Total: {X_train.shape[0]} sampel")
            st.write(f"- Positif: {train_pos} ({train_pos/X_train.shape[0]*100:.1f}%)")
            st.write(f"- Negatif: {train_neg} ({train_neg/X_train.shape[0]*100:.1f}%)")
        
        with col2:
            st.write("**Testing Set:**")
            st.write(f"- Total: {X_test.shape[0]} sampel")
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
    """Training dan evaluasi model SVM"""
    st.header("8. TRAINING DAN EVALUASI MODEL SVM")
    
    def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel_type='linear'):
        """Melatih dan mengevaluasi model SVM"""
        
        svm_model = SVC(
            kernel=kernel_type,
            random_state=42,
            C=1.0,
            probability=True if kernel_type == 'poly' else False
        )
        
        svm_model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = svm_model.predict(X_test)
        
        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['negative', 'positive'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'model': svm_model,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'y_true': y_test
        }
    
    # Loop untuk setiap rasio dan kernel
    all_results = {}
    accuracy_comparison = []
    
    for ratio_name, data in results.items():
        st.subheader(f"Evaluasi untuk Rasio {ratio_name}")
        
        ratio_results = {}
        
        for kernel in ['linear', 'poly']:
            with st.spinner(f"Training SVM dengan kernel {kernel}..."):
                result = train_and_evaluate_svm(
                    data['X_train'],
                    data['X_test'],
                    data['y_train'],
                    data['y_test'],
                    kernel_type=kernel
                )
            
            ratio_results[kernel] = result
            
            # Tampilkan hasil
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Akurasi", f"{result['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{result['classification_report']['positive']['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{result['classification_report']['positive']['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{result['classification_report']['positive']['f1-score']:.4f}")
            
            accuracy_comparison.append({
                'Rasio': ratio_name,
                'Kernel': kernel,
                'Akurasi': result['accuracy']
            })
        
        all_results[ratio_name] = ratio_results
        st.write("---")
    
    return all_results, accuracy_comparison

def visualize_results(all_results, accuracy_comparison):
    """Visualisasi hasil"""
    st.header("9. VISUALISASI HASIL")
    
    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    idx = 0
    for ratio_name, ratio_results in all_results.items():
        for kernel_name, result in ratio_results.items():
            cm = result['confusion_matrix']
            ax = axes[idx]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Negatif', 'Positif'],
                        yticklabels=['Negatif', 'Positif'],
                        ax=ax)
            
            ax.set_title(f'Rasio {ratio_name} - Kernel {kernel_name}\nAkurasi: {result["accuracy"]:.4f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            idx += 1
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Perbandingan akurasi
    st.subheader("Perbandingan Akurasi")
    accuracy_df = pd.DataFrame(accuracy_comparison)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=accuracy_df, x='Rasio', y='Akurasi', hue='Kernel', ax=ax)
    ax.set_title('Perbandingan Akurasi SVM Berbagai Rasio dan Kernel', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Kernel', loc='upper right')
    ax.grid(True, alpha=0.3)
    
    for i, row in accuracy_df.iterrows():
        ax.text(i % 3, row['Akurasi'] + 0.01, f"{row['Akurasi']:.4f}", 
                ha='center', fontsize=10)
    
    st.pyplot(fig)
    
    # Tabel perbandingan
    with st.expander("Tabel Perbandingan Detail"):
        st.dataframe(accuracy_df)
    
    return accuracy_df

def classify_new_sentences(all_results, tfidf_vectorizer):
    """Klasifikasi kalimat baru"""
    st.header("10. KLASIFIKASI KALIMAT BARU")
    
    # Fungsi preprocessing
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    
    def remove_stopwords(text):
        return stopword_remover.remove(text)
    
    def tokenize_text(text):
        return word_tokenize(text)
    
    def count_words(text):
        if not isinstance(text, str):
            return 0
        return len(text.split())
    
    # Pilih model terbaik
    best_accuracy = 0
    best_model_info = None
    
    for ratio_name, ratio_results in all_results.items():
        for kernel_name, result in ratio_results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model_info = {
                    'model': result['model'],
                    'ratio': ratio_name,
                    'kernel': kernel_name,
                    'accuracy': result['accuracy']
                }
    
    st.success(f"✨ Model Terbaik: Rasio {best_model_info['ratio']} - Kernel {best_model_info['kernel']}")
    st.metric("Akurasi Terbaik", f"{best_model_info['accuracy']:.4f}")
    
    # Fungsi prediksi
    def predict_sentiment(text, model, vectorizer):
        """Fungsi untuk memprediksi sentimen dari kalimat baru"""
        
        # Preprocessing
        cleaned_text = clean_text(text)
        text_no_stopwords = remove_stopwords(cleaned_text)
        tokens = tokenize_text(text_no_stopwords)
        processed_text = ' '.join(tokens)
        
        # Transformasi TF-IDF
        text_vectorized = vectorizer.transform([processed_text])
        
        # Prediksi
        prediction = model.predict(text_vectorized)[0]
        sentiment = 'POSITIF' if prediction == 1 else 'NEGATIF'
        
        # Hitung jumlah kata
        word_count = count_words(text)
        word_count_processed = count_words(processed_text)
        
        return sentiment, processed_text, word_count, word_count_processed
    
    # Contoh kalimat
    st.subheader("Contoh Kalimat")
    test_sentences = [
        "aplikasi gojek sangat bagus dan membantu sekali",
        "pelayanan buruk, driver tidak profesional",
        "sangat cepat dan murah, saya suka",
        "aplikasi sering error dan sulit digunakan",
        "driver ramah dan perjalanan nyaman",
        "waktu tunggu terlalu lama dan mahal",
        "gojek adalah aplikasi terbaik untuk transportasi",
        "saya kecewa dengan pelayanan yang diberikan",
        "mantap banget nih aplikasi, recommended!",
        "parah banget servicenya, ga mau lagi pake gojek"
    ]
    
    results_list = []
    for i, sentence in enumerate(test_sentences, 1):
        sentiment, processed_text, wc_original, wc_processed = predict_sentiment(
            sentence, 
            best_model_info['model'], 
            tfidf_vectorizer
        )
        
        results_list.append({
            'No': i,
            'Kalimat': sentence,
            'Jml Kata': wc_original,
            'Hasil': sentiment,
            'Warna': '🟢' if sentiment == 'POSITIF' else '🔴'
        })
    
    # Tampilkan hasil dalam tabel
    results_df = pd.DataFrame(results_list)
    st.table(results_df[['No', 'Kalimat', 'Jml Kata', 'Hasil', 'Warna']])
    
    # Input interaktif
    st.subheader("Input Kalimat Anda")
    user_input = st.text_area("Masukkan kalimat untuk dianalisis:", 
                             "Aplikasi Gojek sangat membantu dalam kehidupan sehari-hari")
    
    if st.button("Analisis Sentimen"):
        if user_input:
            sentiment, processed_text, wc_original, wc_processed = predict_sentiment(
                user_input, 
                best_model_info['model'], 
                tfidf_vectorizer
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jumlah Kata", wc_original)
            with col2:
                st.metric("Kata setelah Preprocessing", wc_processed)
            with col3:
                color = "green" if sentiment == 'POSITIF' else "red"
                st.markdown(f"<h3 style='color: {color};'>{sentiment}</h3>", unsafe_allow_html=True)
            
            with st.expander("Detail Analisis"):
                st.write(f"**Kalimat asli:** {user_input}")
                st.write(f"**Setelah preprocessing:** {processed_text}")
                st.write(f"**Model yang digunakan:** {best_model_info['ratio']} ({best_model_info['kernel']})")
                st.write(f"**Akurasi model:** {best_model_info['accuracy']:.4f}")
    
    return best_model_info

def final_statistics(df, sentiment_distribution, tfidf_vectorizer, best_model_info, all_results):
    """Statistik final dan penyimpanan data"""
    st.header("11. STATISTIK FINAL")
    
    st.subheader("Rekapitulasi Proyek")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Data Awal", f"{len(df):,}")
        st.metric("Total Kata Semua Ulasan", f"{df['word_count'].sum():,}")
        st.metric("Rata-rata Kata per Ulasan", f"{df['word_count'].mean():.1f}")
    
    with col2:
        positif_count = sentiment_distribution.get('positive', 0)
        negatif_count = sentiment_distribution.get('negative', 0)
        st.metric("Ulasan Positif", f"{positif_count:,}")
        st.metric("Ulasan Negatif", f"{negatif_count:,}")
        st.metric("Jumlah Fitur TF-IDF", f"{len(tfidf_vectorizer.get_feature_names_out())}")
    
    st.metric("Akurasi Terbaik", f"{best_model_info['accuracy']:.4f}", 
              f"{best_model_info['ratio']} ({best_model_info['kernel']})")
    
    # Ringkasan semua model
    st.subheader("Ringkasan Semua Model")
    summary_data = []
    for ratio_name, ratio_results in all_results.items():
        for kernel_name, result in ratio_results.items():
            summary_data.append({
                'Rasio': ratio_name,
                'Kernel': kernel_name,
                'Akurasi': f"{result['accuracy']:.4f}",
                'Precision': f"{result['classification_report']['positive']['precision']:.4f}",
                'Recall': f"{result['classification_report']['positive']['recall']:.4f}",
                'F1-Score': f"{result['classification_report']['positive']['f1-score']:.4f}"
            })
    
    st.dataframe(pd.DataFrame(summary_data))
    
    # Opsi untuk menyimpan data
    st.subheader("Simpan Hasil Analisis")
    
    if st.button("💾 Simpan Model dan Hasil Analisis"):
        # Buat timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Simpan model terbaik
        model_filename = f'best_svm_model_{timestamp}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump({
                'model': best_model_info['model'],
                'vectorizer': tfidf_vectorizer,
                'accuracy': best_model_info['accuracy'],
                'ratio': best_model_info['ratio'],
                'kernel': best_model_info['kernel'],
                'feature_names': tfidf_vectorizer.get_feature_names_out().tolist()
            }, f)
        
        # Simpan semua hasil
        results_summary = []
        for ratio_name, ratio_results in all_results.items():
            for kernel_name, result in ratio_results.items():
                results_summary.append({
                    'ratio': ratio_name,
                    'kernel': kernel_name,
                    'accuracy': float(result['accuracy']),
                    'precision_positive': float(result['classification_report']['positive']['precision']),
                    'recall_positive': float(result['classification_report']['positive']['recall']),
                    'f1_positive': float(result['classification_report']['positive']['f1-score']),
                    'confusion_matrix': result['confusion_matrix'].tolist()
                })
        
        results_filename = f'model_results_{timestamp}.json'
        with open(results_filename, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Simpan data yang telah diproses
        data_filename = f'processed_gojek_reviews_{timestamp}.csv'
        df_save = df[['content', 'sentiment_label', 'word_count', 'processed_text', 'word_count_processed']].copy()
        df_save.to_csv(data_filename, index=False, encoding='utf-8')
        
        st.success(f"✅ Data dan model berhasil disimpan:")
        st.write(f"- Model terbaik: `{model_filename}`")
        st.write(f"- Hasil evaluasi: `{results_filename}`")
        st.write(f"- Data proses: `{data_filename}`")

def main():
    """Fungsi utama"""
    setup_page()
    
    # Sidebar untuk navigasi
    st.sidebar.title("📊 Navigasi Analisis")
    sections = [
        "1. Load Data",
        "2. Analisis Jumlah Kata", 
        "3. Pelabelan Sentimen",
        "4. WordCloud",
        "5. Preprocessing Text",
        "6. Ekstraksi Fitur TF-IDF",
        "7. Pembagian Data",
        "8. Training & Evaluasi SVM",
        "9. Visualisasi Hasil",
        "10. Klasifikasi Kalimat Baru",
        "11. Statistik Final"
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
    if 'best_model_info' not in st.session_state:
        st.session_state.best_model_info = None
    
    # Eksekusi berdasarkan section yang dipilih
    if selected_section == "1. Load Data":
        st.session_state.df = load_data()
    
    elif selected_section == "2. Analisis Jumlah Kata":
        if st.session_state.df is not None:
            st.session_state.df = analyze_word_count(st.session_state.df)
        else:
            st.warning("⚠️ Silakan jalankan section '1. Load Data' terlebih dahulu!")
    
    elif selected_section == "3. Pelabelan Sentimen":
        if st.session_state.df is not None:
            st.session_state.df, st.session_state.sentiment_distribution = lexicon_sentiment_labeling(st.session_state.df)
        else:
            st.warning("⚠️ Silakan jalankan section '1. Load Data' terlebih dahulu!")
    
    elif selected_section == "4. WordCloud":
        if st.session_state.df is not None:
            create_wordcloud_viz(st.session_state.df)
        else:
            st.warning("⚠️ Silakan jalankan section '1. Load Data' terlebih dahulu!")
    
    elif selected_section == "5. Preprocessing Text":
        if st.session_state.df is not None:
            st.session_state.df = text_preprocessing(st.session_state.df)
        else:
            st.warning("⚠️ Silakan jalankan section '1. Load Data' terlebih dahulu!")
    
    elif selected_section == "6. Ekstraksi Fitur TF-IDF":
        if st.session_state.df is not None:
            st.session_state.X, st.session_state.y, st.session_state.tfidf_vectorizer = tfidf_feature_extraction(st.session_state.df)
        else:
            st.warning("⚠️ Silakan jalankan section '1. Load Data' terlebih dahulu!")
    
    elif selected_section == "7. Pembagian Data":
        if st.session_state.X is not None and st.session_state.y is not None:
            st.session_state.results = data_splitting(st.session_state.X, st.session_state.y)
        else:
            st.warning("⚠️ Silakan jalankan section '6. Ekstraksi Fitur TF-IDF' terlebih dahulu!")
    
    elif selected_section == "8. Training & Evaluasi SVM":
        if st.session_state.results is not None:
            st.session_state.all_results, st.session_state.accuracy_comparison = train_evaluate_svm(st.session_state.results)
        else:
            st.warning("⚠️ Silakan jalankan section '7. Pembagian Data' terlebih dahulu!")
    
    elif selected_section == "9. Visualisasi Hasil":
        if st.session_state.all_results is not None:
            visualize_results(st.session_state.all_results, st.session_state.accuracy_comparison)
        else:
            st.warning("⚠️ Silakan jalankan section '8. Training & Evaluasi SVM' terlebih dahulu!")
    
    elif selected_section == "10. Klasifikasi Kalimat Baru":
        if st.session_state.all_results is not None and st.session_state.tfidf_vectorizer is not None:
            st.session_state.best_model_info = classify_new_sentences(st.session_state.all_results, st.session_state.tfidf_vectorizer)
        else:
            st.warning("⚠️ Silakan jalankan section '8. Training & Evaluasi SVM' terlebih dahulu!")
    
    elif selected_section == "11. Statistik Final":
        if (st.session_state.df is not None and 
            st.session_state.sentiment_distribution is not None and
            st.session_state.tfidf_vectorizer is not None and
            st.session_state.best_model_info is not None and
            st.session_state.all_results is not None):
            
            final_statistics(
                st.session_state.df,
                st.session_state.sentiment_distribution,
                st.session_state.tfidf_vectorizer,
                st.session_state.best_model_info,
                st.session_state.all_results
            )
        else:
            st.warning("⚠️ Silakan jalankan semua section sebelumnya terlebih dahulu!")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("**Analisis Sentimen Ulasan Gojek**\n\nMenggunakan SVM dan Lexicon Bahasa Indonesia")
    
    # Tombol untuk menjalankan semua
    if st.sidebar.button("🚀 Jalankan Semua Analisis", type="primary"):
        st.info("Proses sedang berjalan... Harap tunggu!")
        
        # Reset dan jalankan semua
        st.session_state.df = load_data()
        st.session_state.df = analyze_word_count(st.session_state.df)
        st.session_state.df, st.session_state.sentiment_distribution = lexicon_sentiment_labeling(st.session_state.df)
        st.session_state.df = text_preprocessing(st.session_state.df)
        st.session_state.X, st.session_state.y, st.session_state.tfidf_vectorizer = tfidf_feature_extraction(st.session_state.df)
        st.session_state.results = data_splitting(st.session_state.X, st.session_state.y)
        st.session_state.all_results, st.session_state.accuracy_comparison = train_evaluate_svm(st.session_state.results)
        st.session_state.best_model_info = classify_new_sentences(st.session_state.all_results, st.session_state.tfidf_vectorizer)
        
        st.success("✅ Semua analisis selesai! Lihat hasil di section 11.")

if __name__ == "__main__":
    main()
