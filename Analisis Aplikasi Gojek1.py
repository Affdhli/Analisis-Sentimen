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
import io

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

def upload_data():
    """Fungsi untuk upload data"""
    st.header("1. UPLOAD DATA")
    
    # Pilihan upload atau gunakan data contoh
    option = st.radio(
        "Pilih sumber data:",
        ["📤 Upload file CSV", "📊 Gunakan data contoh"]
    )
    
    df = None
    
    if option == "📤 Upload file CSV":
        uploaded_file = st.file_uploader(
            "Upload file CSV dengan kolom 'content' (dan 'sentimen' jika ada)", 
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                # Baca file
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File berhasil diupload: {uploaded_file.name}")
                
                # Validasi kolom
                if 'content' not in df.columns:
                    st.error("❌ File harus memiliki kolom 'content'")
                    return None
                
                # Ambil 8000 data pertama jika lebih
                max_data = 8000
                original_count = len(df)
                
                if len(df) > max_data:
                    df = df.head(max_data)
                    st.info(f"📊 Mengambil {max_data} data pertama dari total {original_count} data")
                else:
                    st.info(f"📊 Menggunakan semua data yang tersedia: {len(df)} data")
                
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
                st.error(f"❌ Error membaca file: {str(e)}")
                return None
    
    else:  # Gunakan data contoh
        st.info("📊 Membuat data contoh 8000 baris...")
        
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
        
        st.success(f"✅ Data contoh berhasil dibuat: {len(df)} baris")
        
        # Tampilkan preview
        with st.expander("Preview Data Contoh"):
            st.dataframe(df.head())
    
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
        
        # Tampilkan distribusi sentimen awal jika ada
        if 'sentimen' in df.columns:
            st.subheader("Distribusi Sentimen Awal")
            sentiment_counts = df['sentimen'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#2ecc71', '#e74c3c']
            
            # Pastikan warna sesuai dengan jumlah kategori
            if len(sentiment_counts) == 2:
                bar_colors = colors
            else:
                bar_colors = plt.cm.Set3(range(len(sentiment_counts)))
            
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, alpha=0.7)
            ax.set_xlabel('Sentimen')
            ax.set_ylabel('Jumlah')
            ax.set_title('Distribusi Sentimen Awal')
            
            for i, v in enumerate(sentiment_counts.values):
                ax.text(i, v + max(sentiment_counts.values)*0.01, str(v), ha='center')
            
            st.pyplot(fig)
            
            # Pie chart
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                    autopct='%1.1f%%', startangle=90)
            ax2.set_title('Persentase Sentimen')
            st.pyplot(fig2)
        
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
    st.subheader("📊 STATISTIK JUMLAH KATA (8000 ULASAN):")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Kata Semua Ulasan", f"{df['word_count'].sum():,} kata")
    with col2:
        st.metric("Rata-rata Kata per Ulasan", f"{df['word_count'].mean():.1f} kata")
    with col3:
        st.metric("Median Kata per Ulasan", f"{df['word_count'].median():.1f} kata")
    
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Ulasan Terpendek", f"{df['word_count'].min()} kata")
    with col5:
        st.metric("Ulasan Terpanjang", f"{df['word_count'].max()} kata")
    
    # Visualisasi
    st.subheader("📈 Visualisasi Distribusi")
    
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
    with st.spinner("🔄 Melabeli sentimen..."):
        df['sentiment_label'] = df['content'].apply(lexicon_sentiment_analysis_binary)
    
    # HAPUS jika ada yang masih netral (tidak seharusnya ada)
    df = df[df['sentiment_label'].isin(['positive', 'negative'])].copy()
    
    # Hitung distribusi sentimen
    sentiment_distribution = df['sentiment_label'].value_counts()
    total_data = len(df)
    
    # Tampilkan statistik
    st.success(f"✅ Pelabelan selesai: {total_data} ulasan")
    
    st.subheader("📊 DISTRIBUSI SENTIMEN (BINARY - HANYA POSITIF/NEGATIF):")
    
    col1, col2 = st.columns(2)
    with col1:
        positif_count = sentiment_distribution.get('positive', 0)
        positif_pct = (positif_count/total_data*100) if total_data > 0 else 0
        st.metric("Positif", f"{positif_count:,}", f"({positif_pct:.1f}%)")
    with col2:
        negatif_count = sentiment_distribution.get('negative', 0)
        negatif_pct = (negatif_count/total_data*100) if total_data > 0 else 0
        st.metric("Negatif", f"{negatif_count:,}", f"({negatif_pct:.1f}%)")
    
    # Analisis jumlah kata per kategori
    positive_word_counts = df[df['sentiment_label'] == 'positive']['word_count']
    negative_word_counts = df[df['sentiment_label'] == 'negative']['word_count']
    
    st.subheader("📊 ANALISIS JUMLAH KATA PER KATEGORI:")
    
    col_pos, col_neg = st.columns(2)
    
    with col_pos:
        st.write("**POSITIF:**")
        st.write(f"Total kata: {positive_word_counts.sum():,} kata")
        st.write(f"Rata-rata: {positive_word_counts.mean():.1f} kata/ulasan")
        st.write(f"Median: {positive_word_counts.median():.1f} kata")
    
    with col_neg:
        st.write("**NEGATIF:**")
        st.write(f"Total kata: {negative_word_counts.sum():,} kata")
        st.write(f"Rata-rata: {negative_word_counts.mean():.1f} kata/ulasan")
        st.write(f"Median: {negative_word_counts.median():.1f} kata")
    
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
        st.write(f"**{title}:**")
        st.write(f"- Total kata: {len(words):,}")
        st.write(f"- Kata unik: {len(unique_words):,}")
        st.write("---")
    
    # Wordcloud untuk semua data
    all_text = ' '.join(df['content'].astype(str).tolist())
    create_wordcloud(all_text, 'WordCloud Semua Ulasan Gojek', 'steelblue')
    
    # Wordcloud untuk positif
    positive_text = ' '.join(df[df['sentiment_label'] == 'positive']['content'].astype(str).tolist())
    create_wordcloud(positive_text, 'WordCloud - Ulasan Positif', 'green')
    
    # Wordcloud untuk negatif
    negative_text = ' '.join(df[df['sentiment_label'] == 'negative']['content'].astype(str).tolist())
    create_wordcloud(negative_text, 'WordCloud - Ulasan Negatif', 'darkred')

def text_preprocessing(df):
    """Preprocessing teks"""
    st.header("5. TEXT PREPROCESSING")
    
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
    
    st.success("✓ Preprocessing selesai!")
    
    # Tampilkan perbandingan
    st.subheader("📊 PERBANDINGAN JUMLAH KATA:")
    
    before_total = df['word_count'].sum()
    after_total = df['word_count_processed'].sum()
    reduction = before_total - after_total
    reduction_pct = (reduction/before_total*100) if before_total > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sebelum preprocessing", f"{before_total:,} kata")
    
    with col2:
        st.metric("Setelah preprocessing", f"{after_total:,} kata")
    
    st.info(f"📉 Pengurangan: {reduction:,} kata ({reduction_pct:.1f}%)")
    
    # Contoh hasil preprocessing
    st.subheader("📝 CONTOH HASIL PREPROCESSING:")
    
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
    with st.spinner("🔄 Melakukan transformasi TF-IDF..."):
        X = tfidf_vectorizer.fit_transform(df['processed_text'])
        y = df['sentiment_label'].map({'positive': 1, 'negative': 0})
    
    st.success(f"✓ Transformasi TF-IDF selesai!")
    
    st.subheader("📊 INFORMASI FITUR:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dimensi matriks TF-IDF", f"{X.shape}")
    with col2:
        st.metric("Jumlah fitur (kata unik)", f"{len(tfidf_vectorizer.get_feature_names_out())}")
    
    # Tampilkan 20 fitur teratas berdasarkan IDF
    st.subheader("🏆 Top 20 Fitur berdasarkan IDF (kata paling khas):")
    
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
    st.header("7. PEMBAGIAN DATA TRAINING-TESTING")
    
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
    """Training dan evaluasi model SVM"""
    st.header("8. TRAINING DAN EVALUASI MODEL SVM")
    st.write("="*60)
    
    def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel_type='linear'):
        """Melatih dan mengevaluasi model SVM"""

        svm_model = SVC(
            kernel=kernel_type,
            random_state=42,
            C=1.0,
            probability=True if kernel_type == 'poly' else False
        )

        with st.spinner(f"Training SVM dengan kernel {kernel_type}..."):
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
        st.subheader(f"EVALUASI UNTUK RASIO {ratio_name}")
        st.write('='*40)
        
        ratio_results = {}
        
        for kernel in ['linear', 'poly']:
            st.write(f"\n**Kernel: {kernel}**")
            
            result = train_and_evaluate_svm(
                data['X_train'],
                data['X_test'],
                data['y_train'],
                data['y_test'],
                kernel_type=kernel
            )
            
            ratio_results[kernel] = result
            
            # Tampilkan akurasi
            st.write(f"**Akurasi: {result['accuracy']:.4f}**")
            
            # Buat tabel evaluasi untuk kernel ini
            eval_data = {
                'Metric': ['Akurasi', 'Precision (Negatif)', 'Recall (Negatif)', 'F1-Score (Negatif)', 
                          'Precision (Positif)', 'Recall (Positif)', 'F1-Score (Positif)'],
                'Nilai': [
                    result['accuracy'],
                    result['classification_report']['negative']['precision'],
                    result['classification_report']['negative']['recall'],
                    result['classification_report']['negative']['f1-score'],
                    result['classification_report']['positive']['precision'],
                    result['classification_report']['positive']['recall'],
                    result['classification_report']['positive']['f1-score']
                ]
            }
            
            eval_df = pd.DataFrame(eval_data)
            eval_df['Nilai'] = eval_df['Nilai'].apply(lambda x: f"{x:.4f}")
            
            # Tampilkan tabel
            st.table(eval_df)
            
            # Tampilkan detail classification report
            with st.expander(f"📋 Detail Classification Report - {kernel}"):
                # Buat dataframe dari classification report
                report_df = pd.DataFrame(result['classification_report']).transpose()
                # Format nilai menjadi 4 desimal
                numeric_cols = ['precision', 'recall', 'f1-score', 'support']
                for col in numeric_cols:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                st.dataframe(report_df)
            
            # Confusion Matrix dalam bentuk tabel
            st.write("**Confusion Matrix:**")
            cm_df = pd.DataFrame(
                result['confusion_matrix'],
                index=['Actual Negatif', 'Actual Positif'],
                columns=['Predicted Negatif', 'Predicted Positif']
            )
            st.table(cm_df)
            
            # Simpan untuk perbandingan
            accuracy_comparison.append({
                'Rasio': ratio_name,
                'Kernel': kernel,
                'Akurasi': result['accuracy'],
                'Precision_Negatif': result['classification_report']['negative']['precision'],
                'Recall_Negatif': result['classification_report']['negative']['recall'],
                'F1_Negatif': result['classification_report']['negative']['f1-score'],
                'Precision_Positif': result['classification_report']['positive']['precision'],
                'Recall_Positif': result['classification_report']['positive']['recall'],
                'F1_Positif': result['classification_report']['positive']['f1-score']
            })
            
            st.write("---")
        
        all_results[ratio_name] = ratio_results
        
        # Tampilkan tabel perbandingan untuk rasio ini
        st.subheader(f"📊 PERBANDINGAN KERNEL UNTUK RASIO {ratio_name}")
        comparison_data = []
        for kernel in ['linear', 'poly']:
            if kernel in ratio_results:
                result = ratio_results[kernel]
                comparison_data.append({
                    'Kernel': kernel,
                    'Akurasi': f"{result['accuracy']:.4f}",
                    'Precision (Negatif)': f"{result['classification_report']['negative']['precision']:.4f}",
                    'Recall (Negatif)': f"{result['classification_report']['negative']['recall']:.4f}",
                    'F1-Score (Negatif)': f"{result['classification_report']['negative']['f1-score']:.4f}",
                    'Precision (Positif)': f"{result['classification_report']['positive']['precision']:.4f}",
                    'Recall (Positif)': f"{result['classification_report']['positive']['recall']:.4f}",
                    'F1-Score (Positif)': f"{result['classification_report']['positive']['f1-score']:.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        st.write("="*50)
    
    # Tabel ringkasan semua model
    st.header("📈 RINGKASAN SEMUA MODEL")
    
    summary_data = []
    for item in accuracy_comparison:
        summary_data.append({
            'Rasio': item['Rasio'],
            'Kernel': item['Kernel'],
            'Akurasi': f"{item['Akurasi']:.4f}",
            'P_Neg': f"{item['Precision_Negatif']:.4f}",
            'R_Neg': f"{item['Recall_Negatif']:.4f}",
            'F1_Neg': f"{item['F1_Negatif']:.4f}",
            'P_Pos': f"{item['Precision_Positif']:.4f}",
            'R_Pos': f"{item['Recall_Positif']:.4f}",
            'F1_Pos': f"{item['F1_Positif']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    return all_results, accuracy_comparison

def visualize_results(all_results, accuracy_comparison):
    """Visualisasi hasil"""
    st.header("9. VISUALISASI HASIL")
    
    # Plot confusion matrix untuk setiap kombinasi
    st.subheader("📊 Confusion Matrix")
    
    # Hitung total plot yang akan dibuat
    total_plots = 0
    for ratio_name, ratio_results in all_results.items():
        total_plots += len(ratio_results)
    
    # Buat layout subplot yang sesuai
    n_rows = 2
    n_cols = 3  # Untuk 6 plot (3 rasio × 2 kernel)
    
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
                    
                    ax.set_title(f'Rasio {ratio_name} - Kernel {kernel_name}\nAkurasi: {result["accuracy"]:.4f}')
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
                
                ax.set_title(f'Rasio {ratio_name} - Kernel {kernel_name}\nAkurasi: {result["accuracy"]:.4f}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Perbandingan akurasi
    st.subheader("📈 Perbandingan Akurasi")
    
    if accuracy_comparison:
        accuracy_df = pd.DataFrame(accuracy_comparison)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Pastikan ada data untuk plot
        if not accuracy_df.empty:
            # Buat plot bar
            if 'Kernel' in accuracy_df.columns:
                # Plot dengan hue berdasarkan Kernel
                if len(accuracy_df['Kernel'].unique()) > 1:
                    sns.barplot(data=accuracy_df, x='Rasio', y='Akurasi', hue='Kernel', ax=ax)
                else:
                    # Jika hanya satu kernel, plot tanpa hue
                    sns.barplot(data=accuracy_df, x='Rasio', y='Akurasi', ax=ax)
            else:
                sns.barplot(data=accuracy_df, x='Rasio', y='Akurasi', ax=ax)
            
            ax.set_title('Perbandingan Akurasi SVM Berbagai Rasio dan Kernel', fontsize=14)
            ax.set_ylim(0, 1.0)
            
            if 'Kernel' in accuracy_df.columns and len(accuracy_df['Kernel'].unique()) > 1:
                ax.legend(title='Kernel', loc='upper right')
            
            ax.grid(True, alpha=0.3)
            
            # Tambahkan nilai di atas bar
            for i, (_, row) in enumerate(accuracy_df.iterrows()):
                if 'Kernel' in accuracy_df.columns and len(accuracy_df['Kernel'].unique()) > 1:
                    # Untuk grouped bar plot, posisi x lebih kompleks
                    pass
                else:
                    ax.text(i, row['Akurasi'] + 0.01, f"{row['Akurasi']:.4f}", 
                           ha='center', fontsize=10)
        
        st.pyplot(fig)
        
        # Tampilkan tabel perbandingan
        with st.expander("📋 Tabel Perbandingan Akurasi"):
            st.dataframe(accuracy_df)
    else:
        st.warning("Tidak ada data akurasi untuk divisualisasikan.")
    
    return accuracy_df if accuracy_comparison else None
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
    
    st.success(f"✨ MODEL TERBAIK:")
    st.write(f"   Rasio: {best_model_info['ratio']}")
    st.write(f"   Kernel: {best_model_info['kernel']}")
    st.write(f"   Akurasi: {best_model_info['accuracy']:.4f}")
    
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
    
    # Contoh kalimat untuk diklasifikasikan
    st.subheader("📝 KLASIFIKASI KALIMAT CONTOH")
    
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
            'Warna': '✅' if sentiment == 'POSITIF' else '❌'
        })
    
    # Tampilkan hasil dalam tabel
    results_df = pd.DataFrame(results_list)
    st.table(results_df[['No', 'Kalimat', 'Jml Kata', 'Hasil', 'Warna']])
    
    # Input interaktif
    st.subheader("🔍 INPUT INTERAKTIF DARI PENGGUNA")
    
    st.info("📝 MASUKKAN KALIMAT UNTUK DIKLASIFIKASIKAN")
    
    # Input text
    user_input = st.text_area(
        "Masukkan kalimat untuk dianalisis:",
        "Aplikasi Gojek sangat membantu dalam kehidupan sehari-hari"
    )
    
    if st.button("🎯 Analisis Sentimen"):
        if user_input:
            sentiment, processed_text, wc_original, wc_processed = predict_sentiment(
                user_input, 
                best_model_info['model'], 
                tfidf_vectorizer
            )
            
            # Tampilkan hasil
            st.subheader("🔍 HASIL ANALISIS:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Kalimat", f"{wc_original} kata")
            with col2:
                st.metric("Setelah Preprocessing", f"{wc_processed} kata")
            with col3:
                color = "green" if sentiment == 'POSITIF' else "red"
                st.markdown(f"<h3 style='color: {color};'>{sentiment}</h3>", unsafe_allow_html=True)
            
            with st.expander("📋 Detail Analisis"):
                st.write(f"**Kalimat:** '{user_input}'")
                st.write(f"**Setelah preprocessing:** '{processed_text}'")
                st.write(f"**Model:** {best_model_info['ratio']} ({best_model_info['kernel']})")
                st.write(f"**Akurasi model:** {best_model_info['accuracy']:.4f}")
                
                # Cek kata kunci
                st.write("**Kata kunci terdeteksi:**")
                
                # Lexicon untuk pengecekan
                positive_words = ['bagus', 'baik', 'mantap', 'cepat', 'mudah', 'suka', 'puas', 'ramah', 'nyaman']
                negative_words = ['buruk', 'jelek', 'lambat', 'mahal', 'error', 'sulit', 'kecewa', 'lama']
                
                user_lower = user_input.lower()
                pos_words_found = [word for word in positive_words if word in user_lower]
                neg_words_found = [word for word in negative_words if word in user_lower]
                
                if pos_words_found:
                    st.write(f"✅ Positif: {', '.join(pos_words_found[:5])}")
                if neg_words_found:
                    st.write(f"❌ Negatif: {', '.join(neg_words_found[:5])}")
    
    return best_model_info

def final_statistics(df, sentiment_distribution, tfidf_vectorizer, best_model_info, all_results):
    """Statistik final dan penyimpanan data"""
    st.header("12. STATISTIK FINAL DAN SIMPAN DATA")
    
    st.subheader("📊 REKAPITULASI PROYEK:")
    
    # Tampilkan ringkasan
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"1. Total data awal: {len(df)} ulasan")
        st.write(f"2. Total kata semua ulasan: {df['word_count'].sum():,} kata")
        st.write(f"3. Rata-rata kata per ulasan: {df['word_count'].mean():.1f} kata")
        st.write(f"4. Distribusi sentimen:")
        positif_count = sentiment_distribution.get('positive', 0)
        negatif_count = sentiment_distribution.get('negative', 0)
        st.write(f"   - Positif: {positif_count} ulasan")
        st.write(f"   - Negatif: {negatif_count} ulasan")
    
    with col2:
        st.write(f"5. Setelah preprocessing: {df['word_count_processed'].sum():,} kata")
        st.write(f"6. Jumlah fitur TF-IDF: {len(tfidf_vectorizer.get_feature_names_out())}")
        if best_model_info:
            st.write(f"7. Akurasi terbaik: {best_model_info['accuracy']:.4f} ({best_model_info['ratio']}, {best_model_info['kernel']})")
    
    # Simpan model dan data
    st.subheader("💾 SIMPAN HASIL ANALISIS")
    
    if st.button("💾 Simpan Model dan Hasil Analisis"):
        # Buat timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
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
            
            st.success(f"✅ DATA DAN MODEL DISIMPAN:")
            st.write(f"   Model terbaik: {model_filename}")
            st.write(f"   Hasil evaluasi: {results_filename}")
            st.write(f"   Data proses: {data_filename}")
            
            # Tampilkan pesan akhir
            st.success("""
            ✨ PROYEK ANALISIS SENTIMEN ULASAN GOJEK SELESAI ✨
            """)
            
        except Exception as e:
            st.error(f"❌ Error menyimpan file: {str(e)}")

def main():
    """Fungsi utama"""
    setup_page()
    
    # Sidebar untuk navigasi
    st.sidebar.title("📊 Navigasi Analisis")
    sections = [
        "1. Upload Data",
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
    
    # Koreksi: Section 11 harus "Statistik Final" bukan "12. Statistik Final"
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
    if selected_section == "1. Upload Data":
        st.session_state.df = upload_data()
    
    elif selected_section == "2. Analisis Jumlah Kata":
        if st.session_state.df is not None:
            st.session_state.df = analyze_word_count(st.session_state.df)
        else:
            st.warning("⚠️ Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "3. Pelabelan Sentimen":
        if st.session_state.df is not None:
            st.session_state.df, st.session_state.sentiment_distribution = lexicon_sentiment_labeling(st.session_state.df)
        else:
            st.warning("⚠️ Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "4. WordCloud":
        if st.session_state.df is not None:
            create_wordcloud_viz(st.session_state.df)
        else:
            st.warning("⚠️ Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "5. Preprocessing Text":
        if st.session_state.df is not None:
            st.session_state.df = text_preprocessing(st.session_state.df)
        else:
            st.warning("⚠️ Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "6. Ekstraksi Fitur TF-IDF":
        if st.session_state.df is not None:
            st.session_state.X, st.session_state.y, st.session_state.tfidf_vectorizer = tfidf_feature_extraction(st.session_state.df)
        else:
            st.warning("⚠️ Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "7. Pembagian Data":
        if st.session_state.X is not None and st.session_state.y is not None:
            st.session_state.results = data_splitting(st.session_state.X, st.session_state.y)
        else:
            st.warning("⚠️ Silakan lakukan ekstraksi fitur terlebih dahulu di section '6. Ekstraksi Fitur TF-IDF'!")
    
    elif selected_section == "8. Training & Evaluasi SVM":
        if st.session_state.results is not None:
            st.session_state.all_results, st.session_state.accuracy_comparison = train_evaluate_svm(st.session_state.results)
        else:
            st.warning("⚠️ Silakan lakukan pembagian data terlebih dahulu di section '7. Pembagian Data'!")
    
    elif selected_section == "9. Visualisasi Hasil":
        if st.session_state.all_results is not None:
            visualize_results(st.session_state.all_results, st.session_state.accuracy_comparison)
        else:
            st.warning("⚠️ Silakan latih model terlebih dahulu di section '8. Training & Evaluasi SVM'!")
    
    elif selected_section == "10. Klasifikasi Kalimat Baru":
        if st.session_state.all_results is not None and st.session_state.tfidf_vectorizer is not None:
            st.session_state.best_model_info = classify_new_sentences(st.session_state.all_results, st.session_state.tfidf_vectorizer)
        else:
            st.warning("⚠️ Silakan latih model terlebih dahulu di section '8. Training & Evaluasi SVM'!")
    
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
            st.warning("⚠️ Silakan selesaikan semua section sebelumnya terlebih dahulu!")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("**Analisis Sentimen Ulasan Gojek**\n\nMenggunakan SVM dengan kernel Linear dan Polynomial")

if __name__ == "__main__":
    main()
