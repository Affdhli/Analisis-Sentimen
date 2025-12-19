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
        layout="wide"
    )
    
    st.title("Analisis Sentimen Ulasan Gojek")
    st.markdown("---")

def upload_data():
    """Fungsi untuk upload data"""
    st.header("1. UPLOAD DATA")
    
    # Pilihan upload atau gunakan data contoh
    option = st.radio(
        "Pilih sumber data:",
        ["Upload file CSV", "Gunakan data contoh"]
    )
    
    df = None
    
    if option == "Upload file CSV":
        uploaded_file = st.file_uploader(
            "Upload file CSV dengan kolom 'content' (dan 'sentimen' jika ada)", 
            type=['csv']
        )
        
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
                st.error(f"❌ Error membaca file: {str(e)}")
                return None
    
    else:  # Gunakan data contoh
        st.info("Membuat data contoh 8000 baris...")
        
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
            "pengalaman menggunakan sangat menyenangkan",
            "tidak terlalu mahal",
            "kurang begitu mahal",
            "harga lumayan murah",
            "cukup terjangkau",
            "tidak buruk juga"
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
            "pengalaman buruk tidak akan pakai lagi",
            "tidak terlalu bagus",
            "kurang memuaskan",
            "cukup mengecewakan",
            "tidak sebaik yang diharapkan"
        ]
        
        # Contoh kalimat netral/ambigu
        ambiguous_samples = [
            "harga standar seperti biasa",
            "lumayan lah untuk transportasi",
            "biasa saja tidak istimewa",
            "cukup untuk kebutuhan sehari-hari",
            "tidak ada yang spesial"
        ]
        
        # Generate 8000 data
        n_samples = 8000
        data_content = []
        data_sentiment = []
        
        for i in range(n_samples):
            rand_val = np.random.random()
            
            if rand_val > 0.45:  # 55% positif
                base_text = np.random.choice(positive_samples)
                sentiment = 'positif'
            elif rand_val > 0.15:  # 30% negatif
                base_text = np.random.choice(negative_samples)
                sentiment = 'negatif'
            else:  # 15% ambigu
                base_text = np.random.choice(ambiguous_samples)
                # Untuk data ambigu, random pilih positif atau negatif
                sentiment = 'positif' if np.random.random() > 0.5 else 'negatif'
            
            # Tambah variasi teks
            variations = [
                "", "sangat", "sekali", "banget", "saya rasa", "menurut saya",
                "pengalaman pribadi", "baru saja mencoba", "setelah update",
                "sebenarnya", "mungkin", "agak", "sedikit"
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
        
        st.success(f"Data contoh berhasil dibuat: {len(df)} baris")
        
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
    st.subheader("STATISTIK JUMLAH KATA (8000 ULASAN):")
    
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
    st.subheader("Visualisasi Distribusi")
    
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
    
    # Tampilkan statistik
    st.success(f"Pelabelan selesai: {total_data} ulasan")
    
    st.subheader("DISTRIBUSI SENTIMEN (BINARY - HANYA POSITIF/NEGATIF):")
    
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
    
    st.subheader("ANALISIS JUMLAH KATA PER KATEGORI:")
    
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
    st.subheader("PERBANDINGAN JUMLAH KATA:")
    
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
        
        # Hitung akurasi per kategori
        # Akurasi untuk kelas negatif (0)
        neg_indices = y_test == 0
        neg_accuracy = accuracy_score(y_test[neg_indices], y_pred[neg_indices]) if sum(neg_indices) > 0 else 0
        
        # Akurasi untuk kelas positif (1)
        pos_indices = y_test == 1
        pos_accuracy = accuracy_score(y_test[pos_indices], y_pred[pos_indices]) if sum(pos_indices) > 0 else 0

        return {
            'model': svm_model,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'y_true': y_test,
            'neg_accuracy': neg_accuracy,
            'pos_accuracy': pos_accuracy
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
            
            # Tampilkan akurasi umum
            st.write(f"**Akurasi Keseluruhan: {result['accuracy']:.4f}**")
            
            # Tampilkan akurasi per kategori
            col_acc1, col_acc2 = st.columns(2)
            with col_acc1:
                st.metric("Akurasi Kelas Negatif", f"{result['neg_accuracy']:.4f}")
            with col_acc2:
                st.metric("Akurasi Kelas Positif", f"{result['pos_accuracy']:.4f}")
            
            # Buat tabel evaluasi lengkap
            eval_data = {
                'Metric': [
                    'Akurasi Keseluruhan',
                    'Akurasi Kelas Negatif',
                    'Akurasi Kelas Positif',
                    'Precision (Negatif)',
                    'Recall (Negatif)', 
                    'F1-Score (Negatif)',
                    'Precision (Positif)',
                    'Recall (Positif)',
                    'F1-Score (Positif)',
                    'Support (Negatif)',
                    'Support (Positif)'
                ],
                'Nilai': [
                    result['accuracy'],
                    result['neg_accuracy'],
                    result['pos_accuracy'],
                    result['classification_report']['negative']['precision'],
                    result['classification_report']['negative']['recall'],
                    result['classification_report']['negative']['f1-score'],
                    result['classification_report']['positive']['precision'],
                    result['classification_report']['positive']['recall'],
                    result['classification_report']['positive']['f1-score'],
                    result['classification_report']['negative']['support'],
                    result['classification_report']['positive']['support']
                ]
            }
            
            eval_df = pd.DataFrame(eval_data)
            eval_df['Nilai'] = eval_df['Nilai'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
            
            # Tampilkan tabel
            st.table(eval_df)
            
            # Visualisasi perbandingan akurasi
            fig_acc, ax_acc = plt.subplots(figsize=(8, 4))
            categories = ['Keseluruhan', 'Negatif', 'Positif']
            acc_values = [result['accuracy'], result['neg_accuracy'], result['pos_accuracy']]
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            
            bars = ax_acc.bar(categories, acc_values, color=colors, alpha=0.7)
            ax_acc.set_ylabel('Akurasi')
            ax_acc.set_title(f'Perbandingan Akurasi - Kernel {kernel}')
            ax_acc.set_ylim(0, 1.0)
            ax_acc.grid(True, alpha=0.3, axis='y')
            
            # Tambahkan nilai di atas bar
            for bar, value in zip(bars, acc_values):
                height = bar.get_height()
                ax_acc.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            st.pyplot(fig_acc)
            
            # Tampilkan detail classification report
            with st.expander(f"Detail Classification Report - {kernel}"):
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
            
            # Hitung akurasi dari confusion matrix
            tn, fp, fn, tp = result['confusion_matrix'].ravel()
            total = tn + fp + fn + tp
            
            st.write("**Perhitungan Akurasi dari Confusion Matrix:**")
            st.write(f"- True Negative (TN): {tn}")
            st.write(f"- False Positive (FP): {fp}")
            st.write(f"- False Negative (FN): {fn}")
            st.write(f"- True Positive (TP): {tp}")
            st.write(f"- Total: {total}")
            st.write(f"- Akurasi Keseluruhan: (TN+TP)/Total = ({tn}+{tp})/{total} = {(tn+tp)/total:.4f}")
            st.write(f"- Akurasi Kelas Negatif: TN/(TN+FP) = {tn}/({tn}+{fp}) = {tn/(tn+fp) if (tn+fp)>0 else 0:.4f}")
            st.write(f"- Akurasi Kelas Positif: TP/(TP+FN) = {tp}/({tp}+{fn}) = {tp/(tp+fn) if (tp+fn)>0 else 0:.4f}")
            
            # Simpan untuk perbandingan
            accuracy_comparison.append({
                'Rasio': ratio_name,
                'Kernel': kernel,
                'Akurasi_Keseluruhan': result['accuracy'],
                'Akurasi_Negatif': result['neg_accuracy'],
                'Akurasi_Positif': result['pos_accuracy'],
                'Precision_Negatif': result['classification_report']['negative']['precision'],
                'Recall_Negatif': result['classification_report']['negative']['recall'],
                'F1_Negatif': result['classification_report']['negative']['f1-score'],
                'Precision_Positif': result['classification_report']['positive']['precision'],
                'Recall_Positif': result['classification_report']['positive']['recall'],
                'F1_Positif': result['classification_report']['positive']['f1-score'],
                'Support_Negatif': result['classification_report']['negative']['support'],
                'Support_Positif': result['classification_report']['positive']['support']
            })
            
            st.write("---")
        
        all_results[ratio_name] = ratio_results
        
        # Tampilkan tabel perbandingan untuk rasio ini
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
                    'Precision (Negatif)': f"{result['classification_report']['negative']['precision']:.4f}",
                    'Recall (Negatif)': f"{result['classification_report']['negative']['recall']:.4f}",
                    'F1-Score (Negatif)': f"{result['classification_report']['negative']['f1-score']:.4f}",
                    'Precision (Positif)': f"{result['classification_report']['positive']['precision']:.4f}",
                    'Recall (Positif)': f"{result['classification_report']['positive']['recall']:.4f}",
                    'F1-Score (Positif)': f"{result['classification_report']['positive']['f1-score']:.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualisasi perbandingan kernel untuk rasio ini
        fig_kernel, ax_kernel = plt.subplots(figsize=(10, 6))
        
        kernels = ['linear', 'poly']
        x = np.arange(len(kernels))
        width = 0.25
        
        # Data untuk plot
        overall_acc = [ratio_results[k]['accuracy'] for k in kernels]
        neg_acc = [ratio_results[k]['neg_accuracy'] for k in kernels]
        pos_acc = [ratio_results[k]['pos_accuracy'] for k in kernels]
        
        ax_kernel.bar(x - width, overall_acc, width, label='Akurasi Keseluruhan', color='#3498db')
        ax_kernel.bar(x, neg_acc, width, label='Akurasi Negatif', color='#e74c3c')
        ax_kernel.bar(x + width, pos_acc, width, label='Akurasi Positif', color='#2ecc71')
        
        ax_kernel.set_xlabel('Kernel')
        ax_kernel.set_ylabel('Akurasi')
        ax_kernel.set_title(f'Perbandingan Akurasi Berbagai Kernel - Rasio {ratio_name}')
        ax_kernel.set_xticks(x)
        ax_kernel.set_xticklabels(kernels)
        ax_kernel.set_ylim(0, 1.0)
        ax_kernel.legend()
        ax_kernel.grid(True, alpha=0.3, axis='y')
        
        # Tambahkan nilai di atas bar
        for i, (overall, neg, pos) in enumerate(zip(overall_acc, neg_acc, pos_acc)):
            ax_kernel.text(i - width, overall + 0.01, f'{overall:.3f}', ha='center', fontsize=9)
            ax_kernel.text(i, neg + 0.01, f'{neg:.3f}', ha='center', fontsize=9)
            ax_kernel.text(i + width, pos + 0.01, f'{pos:.3f}', ha='center', fontsize=9)
        
        st.pyplot(fig_kernel)
        
        st.write("="*50)
    
    # Tabel ringkasan semua model
    st.header("RINGKASAN SEMUA MODEL")
    
    summary_data = []
    for item in accuracy_comparison:
        summary_data.append({
            'Rasio': item['Rasio'],
            'Kernel': item['Kernel'],
            'Akurasi': f"{item['Akurasi_Keseluruhan']:.4f}",
            'Akurasi_Neg': f"{item['Akurasi_Negatif']:.4f}",
            'Akurasi_Pos': f"{item['Akurasi_Positif']:.4f}",
            'P_Neg': f"{item['Precision_Negatif']:.4f}",
            'R_Neg': f"{item['Recall_Negatif']:.4f}",
            'F1_Neg': f"{item['F1_Negatif']:.4f}",
            'P_Pos': f"{item['Precision_Positif']:.4f}",
            'R_Pos': f"{item['Recall_Positif']:.4f}",
            'F1_Pos': f"{item['F1_Positif']:.4f}",
            'Support_Neg': int(item['Support_Negatif']),
            'Support_Pos': int(item['Support_Positif'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Visualisasi perbandingan semua model
    st.subheader("VISUALISASI PERBANDINGAN SEMUA MODEL")
    
    # Persiapkan data untuk visualisasi
    if accuracy_comparison:
        vis_df = pd.DataFrame(accuracy_comparison)
        
        # Buat multi-index untuk plotting
        fig_all, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Perbandingan akurasi keseluruhan
        ax1 = axes[0, 0]
        sns.barplot(data=vis_df, x='Rasio', y='Akurasi_Keseluruhan', hue='Kernel', ax=ax1)
        ax1.set_title('Akurasi Keseluruhan per Rasio dan Kernel')
        ax1.set_ylabel('Akurasi')
        ax1.set_ylim(0, 1.0)
        ax1.legend(title='Kernel')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Tambahkan nilai di atas bar
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.3f', fontsize=9)
        
        # Plot 2: Perbandingan akurasi negatif
        ax2 = axes[0, 1]
        sns.barplot(data=vis_df, x='Rasio', y='Akurasi_Negatif', hue='Kernel', ax=ax2)
        ax2.set_title('Akurasi Kelas Negatif per Rasio dan Kernel')
        ax2.set_ylabel('Akurasi')
        ax2.set_ylim(0, 1.0)
        ax2.legend(title='Kernel')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Tambahkan nilai di atas bar
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.3f', fontsize=9)
        
        # Plot 3: Perbandingan akurasi positif
        ax3 = axes[1, 0]
        sns.barplot(data=vis_df, x='Rasio', y='Akurasi_Positif', hue='Kernel', ax=ax3)
        ax3.set_title('Akurasi Kelas Positif per Rasio dan Kernel')
        ax3.set_ylabel('Akurasi')
        ax3.set_ylim(0, 1.0)
        ax3.legend(title='Kernel')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Tambahkan nilai di atas bar
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%.3f', fontsize=9)
        
        # Plot 4: Perbandingan selisih akurasi positif-negatif
        ax4 = axes[1, 1]
        vis_df['Selisih_Akurasi'] = vis_df['Akurasi_Positif'] - vis_df['Akurasi_Negatif']
        sns.barplot(data=vis_df, x='Rasio', y='Selisih_Akurasi', hue='Kernel', ax=ax4)
        ax4.set_title('Selisih Akurasi (Positif - Negatif) per Rasio dan Kernel')
        ax4.set_ylabel('Selisih Akurasi')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.legend(title='Kernel')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Tambahkan nilai di atas bar
        for container in ax4.containers:
            ax4.bar_label(container, fmt='%.3f', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_all)
        
        # Analisis performa per kelas
        st.subheader("ANALISIS PERFORMA PER KELAS")
        
        # Hitung rata-rata akurasi per kelas
        avg_neg_acc = vis_df['Akurasi_Negatif'].mean()
        avg_pos_acc = vis_df['Akurasi_Positif'].mean()
        
        col_avg1, col_avg2 = st.columns(2)
        with col_avg1:
            st.metric("Rata-rata Akurasi Kelas Negatif", f"{avg_neg_acc:.4f}")
        with col_avg2:
            st.metric("Rata-rata Akurasi Kelas Positif", f"{avg_pos_acc:.4f}")
        
        # Identifikasi model terbaik per kelas
        best_neg_idx = vis_df['Akurasi_Negatif'].idxmax()
        best_pos_idx = vis_df['Akurasi_Positif'].idxmax()
        
        st.write("**Model Terbaik untuk Kelas Negatif:**")
        st.write(f"- Rasio: {vis_df.loc[best_neg_idx, 'Rasio']}")
        st.write(f"- Kernel: {vis_df.loc[best_neg_idx, 'Kernel']}")
        st.write(f"- Akurasi: {vis_df.loc[best_neg_idx, 'Akurasi_Negatif']:.4f}")
        
        st.write("**Model Terbaik untuk Kelas Positif:**")
        st.write(f"- Rasio: {vis_df.loc[best_pos_idx, 'Rasio']}")
        st.write(f"- Kernel: {vis_df.loc[best_pos_idx, 'Kernel']}")
        st.write(f"- Akurasi: {vis_df.loc[best_pos_idx, 'Akurasi_Positif']:.4f}")
        
        # Rekomendasi model berdasarkan keseimbangan akurasi
        st.write("**Rekomendasi Model Berdasarkan Keseimbangan Akurasi:**")
        
        # Hitung selisih absolut antara akurasi positif dan negatif
        vis_df['Selisih_Absolut'] = abs(vis_df['Akurasi_Positif'] - vis_df['Akurasi_Negatif'])
        
        # Cari model dengan selisih terkecil (paling seimbang)
        most_balanced_idx = vis_df['Selisih_Absolut'].idxmin()
        
        st.write(f"**Model Paling Seimbang:**")
        st.write(f"- Rasio: {vis_df.loc[most_balanced_idx, 'Rasio']}")
        st.write(f"- Kernel: {vis_df.loc[most_balanced_idx, 'Kernel']}")
        st.write(f"- Akurasi Negatif: {vis_df.loc[most_balanced_idx, 'Akurasi_Negatif']:.4f}")
        st.write(f"- Akurasi Positif: {vis_df.loc[most_balanced_idx, 'Akurasi_Positif']:.4f}")
        st.write(f"- Selisih: {vis_df.loc[most_balanced_idx, 'Selisih_Absolut']:.4f}")
    
    return all_results, accuracy_comparison

def visualize_results(all_results, accuracy_comparison):
    """Visualisasi hasil"""
    st.header("9. VISUALISASI HASIL")
    
    # Plot confusion matrix untuk setiap kombinasi
    st.subheader("Confusion Matrix")
    
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
    
    # Perbandingan akurasi - DIPERBAIKI
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
                # Buat pivot table untuk plotting yang lebih baik
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
                # Buat data untuk stacked bar
                ratios = accuracy_df['Rasio'].unique()
                kernel_types = accuracy_df['Kernel'].unique()
                
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
                    ax2.set_ylim(0, 2.0)  # Maksimum 2.0 karena stacked
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

def classify_new_sentences(all_results, tfidf_vectorizer):
    """Klasifikasi kalimat baru dengan penanganan kalimat rancu"""
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
    
    st.success(f"MODEL TERBAIK:")
    st.write(f"   Rasio: {best_model_info['ratio']}")
    st.write(f"   Kernel: {best_model_info['kernel']}")
    st.write(f"   Akurasi: {best_model_info['accuracy']:.4f}")
    
    # Fungsi prediksi dengan analisis konteks yang ditingkatkan
    def predict_sentiment_with_context(text, model, vectorizer):
        """Fungsi untuk memprediksi sentimen dengan analisis konteks yang ditingkatkan"""
        
        # Preprocessing
        cleaned_text = clean_text(text)
        text_no_stopwords = remove_stopwords(cleaned_text)
        tokens = tokenize_text(text_no_stopwords)
        processed_text = ' '.join(tokens)
        
        # Transformasi TF-IDF
        text_vectorized = vectorizer.transform([processed_text])
        
        # Prediksi dari model
        prediction = model.predict(text_vectorized)[0]
        
        # Analisis manual untuk kalimat rancu dengan metode yang lebih akurat
        manual_sentiment, manual_score = analyze_sentiment_manual_improved(text)
        
        # Gabungkan hasil dari model dan analisis manual dengan strategi baru
        final_sentiment = combine_predictions_improved(prediction, manual_sentiment, manual_score, text)
        
        # Hitung jumlah kata
        word_count = count_words(text)
        word_count_processed = count_words(processed_text)
        
        return final_sentiment, processed_text, word_count, word_count_processed, manual_score, manual_sentiment
    
    def analyze_sentiment_manual_improved(text):
        """Analisis sentimen manual yang lebih akurat untuk kalimat negasi"""
        text_lower = text.lower()
        
        # Kata-kata kunci untuk analisis yang diperluas
        negation_words = ['tidak', 'bukan', 'belum', 'jangan', 'kurang', 'sedikit', 'agak', 'cukup', 'lumayan']
        
        # Kata positif dengan bobot berbeda
        positive_words = {
            'bagus': 1.0, 'baik': 1.0, 'mantap': 1.0, 'cepat': 0.8, 'mudah': 0.8,
            'puas': 1.0, 'ramah': 0.7, 'nyaman': 0.8, 'murah': 1.0, 'terjangkau': 1.0,
            'hemat': 0.8, 'efisien': 0.7, 'profesional': 0.7, 'aman': 0.6
        }
        
        # Kata negatif dengan bobot berbeda
        negative_words = {
            'buruk': 1.0, 'jelek': 1.0, 'lambat': 0.8, 'mahal': 1.0, 'error': 0.8,
            'sulit': 0.7, 'kecewa': 1.0, 'lama': 0.6, 'rumit': 0.6, 'mengecewakan': 1.2,
            'menyedihkan': 1.2, 'parah': 1.0
        }
        
        # Kata intensifier
        intensifier_words = ['sangat', 'sekali', 'banget', 'amat', 'terlalu']
        
        words = text_lower.split()
        score = 0
        sentiment_words_analysis = []
        
        # Analisis kalimat per kata dengan window konteks
        for i, word in enumerate(words):
            word_score = 0
            has_negation = False
            has_intensifier = False
            
            # Cek kata positif
            if word in positive_words:
                word_score = positive_words[word]
                
                # Cek negasi dalam window 2 kata sebelumnya
                for j in range(max(0, i-2), i):
                    if words[j] in negation_words:
                        has_negation = True
                        # Kata positif dengan negasi menjadi negatif dengan bobot penuh
                        word_score = -word_score * 1.0
                        break
                
                # Cek intensifier dalam window 2 kata sebelumnya
                for j in range(max(0, i-2), i):
                    if words[j] in intensifier_words:
                        has_intensifier = True
                        word_score *= 1.3  # Tingkatkan bobot jika ada intensifier
                        break
                        
                score += word_score
                
                # Simpan analisis untuk debugging
                if has_negation:
                    sentiment_words_analysis.append(f"'{word}' dengan negasi → negatif ({word_score:.1f})")
                elif has_intensifier:
                    sentiment_words_analysis.append(f"'{word}' dengan intensifier → positif kuat ({word_score:.1f})")
                else:
                    sentiment_words_analysis.append(f"'{word}' → positif ({word_score:.1f})")
                    
            # Cek kata negatif
            elif word in negative_words:
                word_score = negative_words[word]
                
                # Cek negasi dalam window 2 kata sebelumnya
                for j in range(max(0, i-2), i):
                    if words[j] in negation_words:
                        has_negation = True
                        # Kata negatif dengan negasi menjadi positif dengan bobot penuh
                        word_score = abs(word_score) * 1.0  # Konversi ke positif
                        break
                
                # Cek intensifier dalam window 2 kata sebelumnya
                for j in range(max(0, i-2), i):
                    if words[j] in intensifier_words:
                        has_intensifier = True
                        word_score *= 1.3  # Tingkatkan bobot jika ada intensifier
                        break
                        
                # Jika bukan negasi, maka tetap negatif
                if not has_negation:
                    word_score = -word_score
                    
                score += word_score
                
                # Simpan analisis untuk debugging
                if has_negation:
                    sentiment_words_analysis.append(f"'{word}' dengan negasi → positif ({word_score:.1f})")
                elif has_intensifier:
                    sentiment_words_analysis.append(f"'{word}' dengan intensifier → negatif kuat ({word_score:.1f})")
                else:
                    sentiment_words_analysis.append(f"'{word}' → negatif ({word_score:.1f})")
        
        # Analisis pola khusus untuk kalimat negasi
        special_patterns = {
            # Pola negasi + kata negatif → positif
            'tidak begitu mahal': 0.8,
            'tidak terlalu mahal': 0.8,
            'kurang begitu mahal': 0.7,
            'tidak buruk': 0.6,
            'tidak jelek': 0.6,
            'tidak lambat': 0.5,
            'tidak sulit': 0.5,
            'belum pernah kecewa': 0.7,
            
            # Pola negasi + kata positif → negatif
            'tidak terlalu bagus': -0.6,
            'tidak begitu baik': -0.6,
            'kurang memuaskan': -0.7,
            'tidak puas': -0.8,
            'belum senang': -0.5,
            
            # Pola moderasi
            'lumayan murah': 0.5,
            'cukup terjangkau': 0.4,
            'agak mahal': -0.4,
            'sedikit lambat': -0.3,
        }
        
        # Cek pola khusus
        pattern_bonus = 0
        matched_pattern = None
        for pattern, pattern_score in special_patterns.items():
            if pattern in text_lower:
                pattern_bonus = pattern_score
                matched_pattern = pattern
                score += pattern_bonus
                sentiment_words_analysis.append(f"Pola '{pattern}' → {pattern_score:.1f}")
                break
        
        # Tentukan sentimen manual berdasarkan skor
        if score > 0.2:
            manual_sentiment = 'POSITIF'
        elif score < -0.2:
            manual_sentiment = 'NEGATIF'
        else:
            manual_sentiment = 'AMBIGU'
        
        return manual_sentiment, score
    
    def combine_predictions_improved(model_prediction, manual_sentiment, manual_score, text):
        """Gabungkan prediksi model dengan analisis manual dengan strategi yang lebih cerdas"""
        text_lower = text.lower()
        
        # Cek apakah ada pola negasi yang jelas
        negation_patterns = [
            'tidak begitu', 'tidak terlalu', 'kurang begitu', 
            'tidak', 'bukan', 'belum', 'kurang'
        ]
        
        has_negation = any(pattern in text_lower for pattern in negation_patterns)
        
        # Kata-kata target yang sering dinegasikan
        target_words_in_text = []
        common_targets = ['mahal', 'buruk', 'jelek', 'lambat', 'sulit', 'bagus', 'baik', 'puas']
        for word in common_targets:
            if word in text_lower:
                target_words_in_text.append(word)
        
        # Jika ada negasi dan ada kata target, prioritaskan analisis manual
        if has_negation and target_words_in_text:
            st.info(f"🔍 Ditemukan negasi pada kata: {', '.join(target_words_in_text)}")
            
            # Untuk pola seperti "tidak mahal", "kurang begitu mahal" → POSITIF
            if any(word in ['mahal', 'buruk', 'jelek'] for word in target_words_in_text):
                return 'POSITIF'
            
            # Untuk pola seperti "tidak bagus", "kurang baik" → NEGATIF
            elif any(word in ['bagus', 'baik', 'puas'] for word in target_words_in_text):
                return 'NEGATIF'
        
        # Untuk kasus lain, gunakan analisis manual jika skornya cukup kuat
        if abs(manual_score) > 0.3:
            return manual_sentiment
        else:
            # Jika analisis manual lemah, gunakan model
            return 'POSITIF' if model_prediction == 1 else 'NEGATIF'
    
    # Contoh kalimat untuk diklasifikasikan (dengan fokus pada kalimat negasi)
    st.subheader("KLASIFIKASI KALIMAT CONTOH")
    
    test_sentences = [
        "aplikasi gojek sangat bagus dan membantu sekali",
        "pelayanan buruk, driver tidak profesional",
        "sangat cepat dan murah, saya suka",
        "aplikasi sering error dan sulit digunakan",
        "driver ramah dan perjalanan nyaman",
        "waktu tunggu terlalu lama dan mahal",
        # Kalimat negasi yang harus menjadi positif
        "tidak begitu mahal",           # Harus: POSITIF
        "kurang begitu mahal",          # Harus: POSITIF  
        "tidak terlalu mahal",          # Harus: POSITIF
        "tidak buruk juga",             # Harus: POSITIF
        "tidak jelek",                  # Harus: POSITIF
        "tidak lambat",                 # Harus: POSITIF
        # Kalimat negasi yang harus menjadi negatif
        "tidak terlalu bagus",          # Harus: NEGATIF
        "kurang memuaskan",             # Harus: NEGATIF
        "tidak begitu baik",            # Harus: NEGATIF
        "belum puas",                   # Harus: NEGATIF
        # Kalimat moderasi
        "lumayan murah",                # Harus: POSITIF
        "cukup terjangkau",             # Harus: POSITIF
        "agak mahal",                   # Harus: NEGATIF
        "sedikit lambat"                # Harus: NEGATIF
    ]
    
    results_list = []
    for i, sentence in enumerate(test_sentences, 1):
        sentiment, processed_text, wc_original, wc_processed, manual_score, manual_sentiment = predict_sentiment_with_context(
            sentence, 
            best_model_info['model'], 
            tfidf_vectorizer
        )
        
        # Tentukan kategori kalimat
        if any(word in sentence.lower() for word in ['tidak', 'kurang', 'bukan', 'belum']):
            category = 'Negasi'
        elif any(word in sentence.lower() for word in ['lumayan', 'cukup', 'agak', 'sedikit']):
            category = 'Moderasi'
        else:
            category = 'Biasa'
        
        results_list.append({
            'No': i,
            'Kalimat': sentence,
            'Kategori': category,
            'Jml Kata': wc_original,
            'Hasil': sentiment,
            'Skor Manual': f"{manual_score:.2f}",
            'Analisis Manual': manual_sentiment,
            'Warna': '✅' if sentiment == 'POSITIF' else '❌'
        })
    
    # Tampilkan hasil dalam tabel
    results_df = pd.DataFrame(results_list)
    st.table(results_df[['No', 'Kalimat', 'Kategori', 'Jml Kata', 'Hasil', 'Skor Manual', 'Analisis Manual', 'Warna']])
    
    # Analisis khusus untuk kalimat negasi
    st.subheader("🔍 ANALISIS DETAIL KALIMAT NEGASI")
    
    negation_sentences = [s for s in test_sentences if any(word in s.lower() for word in ['tidak', 'kurang', 'bukan', 'belum'])]
    
    for sentence in negation_sentences:
        sentiment, processed_text, wc_original, wc_processed, manual_score, manual_sentiment = predict_sentiment_with_context(
            sentence, 
            best_model_info['model'], 
            tfidf_vectorizer
        )
        
        st.write(f"**Kalimat:** '{sentence}'")
        st.write(f"**Hasil Akhir:** {sentiment}")
        st.write(f"**Skor Manual:** {manual_score:.2f}")
        st.write(f"**Analisis Manual:** {manual_sentiment}")
        
        # Analisis kata per kata
        words = sentence.lower().split()
        analysis = []
        
        negation_words = ['tidak', 'bukan', 'belum', 'kurang']
        positive_words = ['bagus', 'baik', 'mantap', 'murah', 'terjangkau', 'puas']
        negative_words = ['buruk', 'jelek', 'mahal', 'lambat', 'sulit', 'kecewa', 'memuaskan']
        
        for i, word in enumerate(words):
            if word in negation_words:
                analysis.append(f"**'{word}'** → kata negasi/pembalik")
            elif word in positive_words:
                # Cek apakah ada negasi sebelumnya
                neg_before = any(words[j] in negation_words for j in range(max(0, i-2), i))
                if neg_before:
                    analysis.append(f"**'{word}'** (dengan negasi) → **NEGATIF** (kata positif dibalik)")
                else:
                    analysis.append(f"**'{word}'** → **POSITIF**")
            elif word in negative_words:
                # Cek apakah ada negasi sebelumnya
                neg_before = any(words[j] in negation_words for j in range(max(0, i-2), i))
                if neg_before:
                    analysis.append(f"**'{word}'** (dengan negasi) → **POSITIF** (kata negatif dibalik)")
                else:
                    analysis.append(f"**'{word}'** → **NEGATIF**")
        
        if analysis:
            st.write("**Analisis Kata Per Kata:**")
            for a in analysis:
                st.write(f"- {a}")
        
        st.write("---")
    
    # Input interaktif
    st.subheader("INPUT INTERAKTIF DARI PENGGUNA")
    
    st.info("MASUKKAN KALIMAT UNTUK DIKLASIFIKASIKAN")
    
    # Contoh kalimat negasi untuk placeholder
    example_negation = ""
    
    # Input text dengan contoh kalimat negasi
    user_input = st.text_area(
        "Masukkan kalimat untuk dianalisis:",
        example_negation,
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("Analisis Sentimen", type="primary")
    with col2:
        if st.button("Reset"):
            user_input = ""
    
    if analyze_btn and user_input:
        sentiment, processed_text, wc_original, wc_processed, manual_score, manual_sentiment = predict_sentiment_with_context(
            user_input, 
            best_model_info['model'], 
            tfidf_vectorizer
        )
        
        # Tampilkan hasil
        st.subheader("HASIL ANALISIS:")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Kalimat Asli", f"{wc_original} kata")
        with col2:
            st.metric("Setelah Preprocessing", f"{wc_processed} kata")
        with col3:
            color = "green" if sentiment == 'POSITIF' else "red"
            st.markdown(f"<h3 style='color: {color};'>{sentiment}</h3>", unsafe_allow_html=True)
        with col4:
            st.metric("Skor Analisis Manual", f"{manual_score:.2f}")
        
        with st.expander("Detail Analisis Lengkap", expanded=True):
            st.write(f"**Kalimat Asli:** '{user_input}'")
            st.write(f"**Setelah preprocessing:** '{processed_text}'")
            st.write(f"**Model:** {best_model_info['ratio']} ({best_model_info['kernel']})")
            st.write(f"**Akurasi model:** {best_model_info['accuracy']:.4f}")
            st.write(f"**Analisis Manual:** {manual_sentiment} (skor: {manual_score:.2f})")
            
            # Deteksi pola negasi
            negation_detected = any(word in user_input.lower() for word in ['tidak', 'bukan', 'belum', 'kurang'])
            if negation_detected:
                st.write("**DETEKSI NEGASI:** Terdeteksi kata pembalik dalam kalimat")
                
                # Analisis spesifik untuk negasi
                words = user_input.lower().split()
                negation_words_found = [w for w in words if w in ['tidak', 'bukan', 'belum', 'kurang']]
                st.write(f"**Kata negasi ditemukan:** {', '.join(negation_words_found)}")
                
                # Cek kata yang mungkin dibalik
                potential_targets = []
                target_words = ['mahal', 'buruk', 'jelek', 'lambat', 'sulit', 'bagus', 'baik', 'puas', 'cepat', 'murah']
                for word in target_words:
                    if word in user_input.lower():
                        potential_targets.append(word)
                
                if potential_targets:
                    st.write(f"**Kata target yang mungkin dibalik:** {', '.join(potential_targets)}")
                    
                    # Berikan penjelasan untuk setiap kata target
                    for target in potential_targets:
                        if target in ['mahal', 'buruk', 'jelek', 'lambat', 'sulit']:
                            st.write(f"  - '{target}' dengan negasi → **POSITIF** (misal: 'tidak {target}' = baik)")
                        elif target in ['bagus', 'baik', 'puas', 'cepat', 'murah']:
                            st.write(f"  - '{target}' dengan negasi → **NEGATIF** (misal: 'tidak {target}' = kurang baik)")
            
            # Analisis kata kunci
            st.write("**Analisis Kata Kunci:**")
            
            positive_words = ['bagus', 'baik', 'mantap', 'cepat', 'mudah', 'suka', 'puas', 'ramah', 'nyaman', 'murah', 'terjangkau']
            negative_words = ['buruk', 'jelek', 'lambat', 'mahal', 'error', 'sulit', 'kecewa', 'lama']
            negation_words = ['tidak', 'bukan', 'belum', 'jangan', 'kurang', 'sedikit', 'agak']
            
            user_lower = user_input.lower()
            words = user_lower.split()
            
            found_keywords = False
            for i, word in enumerate(words):
                if word in positive_words:
                    found_keywords = True
                    # Cek negasi sebelumnya
                    neg_before = any(words[j] in negation_words for j in range(max(0, i-3), i))
                    if neg_before:
                        st.write(f"❓ **'{word}'** dengan negasi → mengurangi sentimen positif")
                    else:
                        st.write(f"**'{word}'** → meningkatkan sentimen positif")
                
                elif word in negative_words:
                    found_keywords = True
                    # Cek negasi sebelumnya
                    neg_before = any(words[j] in negation_words for j in range(max(0, i-3), i))
                    if neg_before:
                        st.write(f"**'{word}'** dengan negasi → meningkatkan sentimen positif")
                    else:
                        st.write(f"**'{word}'** → meningkatkan sentimen negatif")
                
                elif word in negation_words:
                    found_keywords = True
                    st.write(f"**'{word}'** → kata pembalik/negasi")
            
            if not found_keywords:
                st.write("Tidak ditemukan kata kunci sentimen yang jelas.")
    
    return best_model_info

def final_statistics(df, sentiment_distribution, tfidf_vectorizer, best_model_info, all_results):
    """Statistik final dan penyimpanan data"""
    st.header("12. STATISTIK FINAL DAN SIMPAN DATA")
    
    st.subheader("REKAPITULASI PROYEK:")
    
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
    
    # Fitur baru: analisis kalimat rancu
    st.subheader("🔍 FITUR ANALISIS KALIMAT RANCU")
    
    st.write("""
    **Fitur baru yang telah ditambahkan:**
    1. **Deteksi kata negasi**: dapat mendeteksi kata seperti "tidak", "kurang", "bukan"
    2. **Analisis konteks**: menganalisis kata dalam konteks kalimat
    3. **Penanganan pola khusus**: seperti "kurang begitu mahal", "tidak terlalu bagus"
    4. **Skoring manual**: memberikan skor berdasarkan analisis kata per kata
    5. **Gabungan prediksi**: menggabungkan hasil model dengan analisis manual
    """)
    
    # Contoh analisis kalimat rancu
    st.write("**Contoh kalimat rancu yang dapat dianalisis:**")
    ambiguous_examples = [
        "kurang begitu mahal → POSITIF (karena 'mahal' dinegasikan oleh 'kurang begitu')",
        "tidak terlalu bagus → NEGATIF (karena 'bagus' dinegasikan oleh 'tidak terlalu')",
        "harga lumayan murah → POSITIF (karena 'murah' dengan intensitas sedang)",
        "cukup terjangkau → POSITIF (karena 'terjangkau' dengan intensitas cukup)",
        "tidak buruk juga → POSITIF (karena 'buruk' dinegasikan oleh 'tidak')"
    ]
    
    for example in ambiguous_examples:
        st.write(f"- {example}")
    
    # Simpan model dan data
    st.subheader("SIMPAN HASIL ANALISIS")
    
    if st.button("Simpan Model dan Hasil Analisis"):
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
            
            st.success(f"DATA DAN MODEL DISIMPAN:")
            st.write(f"   Model terbaik: {model_filename}")
            st.write(f"   Hasil evaluasi: {results_filename}")
            st.write(f"   Data proses: {data_filename}")
        except Exception as e:
            st.error(f"Error menyimpan file: {str(e)}")

def main():
    """Fungsi utama"""
    setup_page()
    
    # Sidebar untuk navigasi
    st.sidebar.title("Navigasi Analisis")
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
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "3. Pelabelan Sentimen":
        if st.session_state.df is not None:
            st.session_state.df, st.session_state.sentiment_distribution = lexicon_sentiment_labeling(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "4. WordCloud":
        if st.session_state.df is not None:
            create_wordcloud_viz(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "5. Preprocessing Text":
        if st.session_state.df is not None:
            st.session_state.df = text_preprocessing(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "6. Ekstraksi Fitur TF-IDF":
        if st.session_state.df is not None:
            st.session_state.X, st.session_state.y, st.session_state.tfidf_vectorizer = tfidf_feature_extraction(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "7. Pembagian Data":
        if st.session_state.X is not None and st.session_state.y is not None:
            st.session_state.results = data_splitting(st.session_state.X, st.session_state.y)
        else:
            st.warning("Silakan lakukan ekstraksi fitur terlebih dahulu di section '6. Ekstraksi Fitur TF-IDF'!")
    
    elif selected_section == "8. Training & Evaluasi SVM":
        if st.session_state.results is not None:
            st.session_state.all_results, st.session_state.accuracy_comparison = train_evaluate_svm(st.session_state.results)
        else:
            st.warning("Silakan lakukan pembagian data terlebih dahulu di section '7. Pembagian Data'!")
    
    elif selected_section == "9. Visualisasi Hasil":
        if st.session_state.all_results is not None:
            visualize_results(st.session_state.all_results, st.session_state.accuracy_comparison)
        else:
            st.warning("Silakan latih model terlebih dahulu di section '8. Training & Evaluasi SVM'!")
    
    elif selected_section == "10. Klasifikasi Kalimat Baru":
        if st.session_state.all_results is not None and st.session_state.tfidf_vectorizer is not None:
            st.session_state.best_model_info = classify_new_sentences(st.session_state.all_results, st.session_state.tfidf_vectorizer)
        else:
            st.warning("Silakan latih model terlebih dahulu di section '8. Training & Evaluasi SVM'!")
    
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
            st.warning("Silakan selesaikan semua section sebelumnya terlebih dahulu!")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Analisis Sentimen Ulasan Gojek 2025**
    """)

if __name__ == "__main__":
    main()
