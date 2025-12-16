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
    st.header("2. ANALISIS JUMLAH KATA")
    
    # Fungsi untuk menghitung jumlah kata
    def count_words(text):
        if not isinstance(text, str):
            return 0
        return len(text.split())
    
    # Hitung jumlah kata untuk semua ulasan
    df['word_count'] = df['content'].apply(count_words)
    
    # Tampilkan statistik
    st.subheader("📊 Statistik Jumlah Kata")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Kata", f"{df['word_count'].sum():,}")
    with col2:
        st.metric("Rata-rata", f"{df['word_count'].mean():.1f}")
    with col3:
        st.metric("Median", f"{df['word_count'].median():.1f}")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Minimal", f"{df['word_count'].min()}")
    with col5:
        st.metric("Maksimal", f"{df['word_count'].max()}")
    with col6:
        std_dev = df['word_count'].std()
        st.metric("Standar Deviasi", f"{std_dev:.1f}")
    
    # Visualisasi
    st.subheader("📈 Visualisasi Distribusi")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(df['word_count'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(df['word_count'].mean(), color='red', linestyle='dashed', 
                    linewidth=2, label=f'Rata-rata: {df["word_count"].mean():.1f}')
    axes[0].set_xlabel('Jumlah Kata')
    axes[0].set_ylabel('Frekuensi')
    axes[0].set_title('Distribusi Jumlah Kata per Ulasan')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df['word_count'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'))
    axes[1].set_ylabel('Jumlah Kata')
    axes[1].set_title('Box Plot Jumlah Kata')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Distribusi detail
    with st.expander("📋 Detail Distribusi Jumlah Kata"):
        # Hitung persentil
        percentiles = [25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(df['word_count'], percentiles)
        
        st.write("**Persentil:**")
        for p, val in zip(percentiles, percentile_values):
            st.write(f"- Persentil {p}%: {val:.1f} kata")
        
        # Hitung frekuensi rentang kata
        st.write("\n**Distribusi Rentang Kata:**")
        bins = [0, 5, 10, 15, 20, 30, 50, 100, df['word_count'].max()]
        labels = ['0-5', '6-10', '11-15', '16-20', '21-30', '31-50', '51-100', '>100']
        
        df['word_range'] = pd.cut(df['word_count'], bins=bins, labels=labels, include_lowest=True)
        range_counts = df['word_range'].value_counts().sort_index()
        
        for range_label, count in range_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"- {range_label} kata: {count} ulasan ({percentage:.1f}%)")
    
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
    
    # Tampilkan info lexicon
    with st.expander("ℹ️ Informasi Lexicon"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Kata Positif:** {len(positive_words)} kata")
            st.write("Contoh: " + ", ".join(positive_words[:10]) + "...")
        with col2:
            st.write(f"**Kata Negatif:** {len(negative_words)} kata")
            st.write("Contoh: " + ", ".join(negative_words[:10]) + "...")
    
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
    
    st.subheader("📊 Distribusi Sentimen Hasil Pelabelan")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        positif_count = sentiment_distribution.get('positive', 0)
        positif_pct = (positif_count/total_data*100) if total_data > 0 else 0
        st.metric("Positif", f"{positif_count:,}", f"{positif_pct:.1f}%")
    with col2:
        negatif_count = sentiment_distribution.get('negative', 0)
        negatif_pct = (negatif_count/total_data*100) if total_data > 0 else 0
        st.metric("Negatif", f"{negatif_count:,}", f"{negatif_pct:.1f}%")
    with col3:
        st.metric("Total Data", f"{total_data:,}")
    
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
    positive_word_counts = df[df['sentiment_label'] == 'positive']['word_count']
    negative_word_counts = df[df['sentiment_label'] == 'negative']['word_count']
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
    
    # Analisis detail
    with st.expander("📋 Detail Analisis per Kategori"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 POSITIF:**")
            st.write(f"- Total ulasan: {len(positive_word_counts):,}")
            st.write(f"- Total kata: {positive_word_counts.sum():,} kata")
            st.write(f"- Rata-rata: {positive_word_counts.mean():.1f} kata/ulasan")
            st.write(f"- Median: {positive_word_counts.median():.1f} kata")
            st.write(f"- Minimal: {positive_word_counts.min()} kata")
            st.write(f"- Maksimal: {positive_word_counts.max()} kata")
        
        with col2:
            st.write("**📉 NEGATIF:**")
            st.write(f"- Total ulasan: {len(negative_word_counts):,}")
            st.write(f"- Total kata: {negative_word_counts.sum():,} kata")
            st.write(f"- Rata-rata: {negative_word_counts.mean():.1f} kata/ulasan")
            st.write(f"- Median: {negative_word_counts.median():.1f} kata")
            st.write(f"- Minimal: {negative_word_counts.min()} kata")
            st.write(f"- Maksimal: {negative_word_counts.max()} kata")
    
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
        
        # Hitung jumlah kata unik
        words = text.split()
        unique_words = set(words)
        
        return fig, len(words), len(unique_words)
    
    # Wordcloud untuk semua data
    st.subheader("🌐 Semua Ulasan")
    all_text = ' '.join(df['content'].astype(str).tolist())
    fig_all, total_words, unique_words = create_wordcloud(all_text, 'WordCloud Semua Ulasan Gojek', 'steelblue')
    st.pyplot(fig_all)
    
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.metric("Total Kata", f"{total_words:,}")
    with col_stats2:
        st.metric("Kata Unik", f"{unique_words:,}")
    
    # Wordcloud untuk positif dan negatif
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Ulasan Positif")
        positive_text = ' '.join(df[df['sentiment_label'] == 'positive']['content'].astype(str).tolist())
        if positive_text.strip():
            fig_pos, total_pos, unique_pos = create_wordcloud(positive_text, 'WordCloud - Ulasan Positif', 'green')
            st.pyplot(fig_pos)
            
            col_stats_pos1, col_stats_pos2 = st.columns(2)
            with col_stats_pos1:
                st.metric("Total Kata", f"{total_pos:,}")
            with col_stats_pos2:
                st.metric("Kata Unik", f"{unique_pos:,}")
        else:
            st.info("Tidak ada data positif")
    
    with col2:
        st.subheader("❌ Ulasan Negatif")
        negative_text = ' '.join(df[df['sentiment_label'] == 'negative']['content'].astype(str).tolist())
        if negative_text.strip():
            fig_neg, total_neg, unique_neg = create_wordcloud(negative_text, 'WordCloud - Ulasan Negatif', 'darkred')
            st.pyplot(fig_neg)
            
            col_stats_neg1, col_stats_neg2 = st.columns(2)
            with col_stats_neg1:
                st.metric("Total Kata", f"{total_neg:,}")
            with col_stats_neg2:
                st.metric("Kata Unik", f"{unique_neg:,}")
        else:
            st.info("Tidak ada data negatif")

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
    st.subheader("🔄 Proses Preprocessing")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Memulai preprocessing...")
    
    # Cleaning
    df['cleaned_text'] = df['content'].apply(clean_text)
    progress_bar.progress(25)
    status_text.text("Cleaning teks...")
    
    # Remove stopwords
    df['text_no_stopwords'] = df['cleaned_text'].apply(remove_stopwords)
    progress_bar.progress(50)
    status_text.text("Menghapus stopwords...")
    
    # Tokenization
    df['tokens'] = df['text_no_stopwords'].apply(tokenize_text)
    progress_bar.progress(75)
    status_text.text("Tokenisasi teks...")
    
    # Gabungkan token kembali menjadi string untuk TF-IDF
    df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    # Hitung jumlah kata setelah preprocessing
    df['word_count_processed'] = df['processed_text'].apply(count_words)
    progress_bar.progress(100)
    
    status_text.text("✓ Preprocessing selesai!")
    
    # Tampilkan perbandingan
    st.success("✅ Preprocessing berhasil!")
    
    st.subheader("📊 Perbandingan Sebelum dan Sesudah Preprocessing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        before_total = df['word_count'].sum()
        st.metric("Sebelum Preprocessing", f"{before_total:,} kata")
    
    with col2:
        after_total = df['word_count_processed'].sum()
        st.metric("Setelah Preprocessing", f"{after_total:,} kata")
    
    with col3:
        reduction = before_total - after_total
        reduction_pct = (reduction/before_total*100) if before_total > 0 else 0
        st.metric("Pengurangan", f"{reduction:,} kata", f"{reduction_pct:.1f}%")
    
    # Contoh hasil preprocessing
    st.subheader("📝 Contoh Hasil Preprocessing")
    
    sample_options = st.selectbox(
        "Pilih contoh data:",
        list(range(1, 6)),
        format_func=lambda x: f"Data ke-{x}"
    )
    
    sample_idx = sample_options - 1
    
    if sample_idx < len(df):
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            st.write("**Sebelum Preprocessing:**")
            st.info(f"{df['content'].iloc[sample_idx]}")
            st.write(f"Jumlah kata: {df['word_count'].iloc[sample_idx]}")
        
        with col_ex2:
            st.write("**Setelah Preprocessing:**")
            st.success(f"{df['processed_text'].iloc[sample_idx]}")
            st.write(f"Jumlah kata: {df['word_count_processed'].iloc[sample_idx]}")
        
        # Tampilkan token
        with st.expander("🔍 Detail Token"):
            st.write("**Tokens:**", df['tokens'].iloc[sample_idx])
            st.write(f"Jumlah token: {len(df['tokens'].iloc[sample_idx])}")
    
    # Statistik setelah preprocessing
    with st.expander("📈 Statistik Setelah Preprocessing"):
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("Rata-rata Kata", f"{df['word_count_processed'].mean():.1f}")
        
        with col_stat2:
            st.metric("Median Kata", f"{df['word_count_processed'].median():.1f}")
        
        with col_stat3:
            st.metric("Kata Terpendek", f"{df['word_count_processed'].min()}")
        
        col_stat4, col_stat5 = st.columns(2)
        
        with col_stat4:
            st.metric("Kata Terpanjang", f"{df['word_count_processed'].max()}")
        
        with col_stat5:
            empty_texts = df[df['word_count_processed'] == 0].shape[0]
            st.metric("Teks Kosong", f"{empty_texts}")
    
    return df

def tfidf_feature_extraction(df):
    """Ekstraksi fitur TF-IDF"""
    st.header("6. EKSTRAKSI FITUR DENGAN TF-IDF")
    
    # Konfigurasi TF-IDF
    st.subheader("⚙️ Konfigurasi TF-IDF")
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        max_features = st.slider("Max Features", 1000, 5000, 3000, 500)
    
    with col_config2:
        min_df = st.slider("Min DF", 1, 10, 3, 1)
    
    with col_config3:
        max_df = st.slider("Max DF", 0.5, 1.0, 0.8, 0.05)
    
    # Inisialisasi TF-IDF Vectorizer
    with st.spinner("🔄 Melakukan ekstraksi fitur TF-IDF..."):
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2)  # Unigram dan bigram
        )
        
        # Transformasi teks menjadi vektor TF-IDF
        X = tfidf_vectorizer.fit_transform(df['processed_text'])
        y = df['sentiment_label'].map({'positive': 1, 'negative': 0})
    
    st.success(f"✅ Transformasi TF-IDF selesai!")
    
    # Tampilkan informasi
    st.subheader("📊 Informasi Fitur")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Dimensi Matriks", f"{X.shape}")
    with col_info2:
        st.metric("Jumlah Fitur", f"{len(tfidf_vectorizer.get_feature_names_out())}")
    with col_info3:
        st.metric("Jumlah Sampel", f"{X.shape[0]}")
    
    # Tampilkan fitur teratas
    st.subheader("🏆 Top 20 Fitur Berdasarkan IDF")
    
    with st.spinner("🔄 Menghitung fitur teratas..."):
        feature_names = tfidf_vectorizer.get_feature_names_out()
        idf_values = tfidf_vectorizer.idf_
        
        # Ambil 20 fitur dengan IDF terendah (paling khas)
        top_indices = np.argsort(idf_values)[:20]
        
        top_features_data = []
        for idx in top_indices:
            top_features_data.append({
                'Fitur': feature_names[idx],
                'IDF Score': f"{idf_values[idx]:.4f}"
            })
        
        top_features_df = pd.DataFrame(top_features_data)
        st.dataframe(top_features_df, use_container_width=True)
    
    # Tampilkan beberapa contoh fitur
    with st.expander("🔍 Contoh Fitur Lainnya"):
        st.write("**Contoh fitur unigram:**")
        unigram_features = [f for f in feature_names if ' ' not in f]
        st.write(", ".join(unigram_features[:50]) + "...")
        
        st.write("\n**Contoh fitur bigram:**")
        bigram_features = [f for f in feature_names if ' ' in f]
        st.write(", ".join(bigram_features[:20]) + "...")
    
    return X, y, tfidf_vectorizer

def data_splitting(X, y):
    """Pembagian data training-testing"""
    st.header("7. PEMBAGIAN DATA TRAINING-TESTING")
    
    # Pilihan rasio
    st.subheader("📐 Pilih Rasio Pembagian")
    
    ratio_option = st.selectbox(
        "Pilih rasio pembagian data:",
        ['80:20 (Training:Testing)', '90:10 (Training:Testing)', '70:30 (Training:Testing)', 'Custom']
    )
    
    if ratio_option == '80:20 (Training:Testing)':
        test_size = 0.2
        ratio_name = '80:20'
    elif ratio_option == '90:10 (Training:Testing)':
        test_size = 0.1
        ratio_name = '90:10'
    elif ratio_option == '70:30 (Training:Testing)':
        test_size = 0.3
        ratio_name = '70:30'
    else:  # Custom
        test_size = st.slider("Ukuran testing set (%)", 10, 40, 20) / 100
        train_size = 1 - test_size
        ratio_name = f"{int(train_size*100)}:{int(test_size*100)}"
    
    # Random seed
    random_seed = st.number_input("Random Seed", min_value=0, max_value=100, value=42)
    
    # Split data
    if st.button("🔀 Bagikan Data"):
        with st.spinner(f"🔄 Membagi data dengan rasio {ratio_name}..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_seed,
                stratify=y
            )
            
            # Hitung distribusi sentimen di training dan testing
            train_pos = sum(y_train == 1)
            train_neg = sum(y_train == 0)
            test_pos = sum(y_test == 1)
            test_neg = sum(y_test == 0)
            
            st.success(f"✅ Pembagian data selesai!")
            
            # Tampilkan hasil
            st.subheader("📊 Hasil Pembagian Data")
            
            col_train, col_test = st.columns(2)
            
            with col_train:
                st.metric("Training Set", f"{X_train.shape[0]:,} sampel")
                st.write(f"- Positif: {train_pos:,} ({train_pos/X_train.shape[0]*100:.1f}%)")
                st.write(f"- Negatif: {train_neg:,} ({train_neg/X_train.shape[0]*100:.1f}%)")
                
                # Pie chart training
                fig_train, ax_train = plt.subplots(figsize=(4, 4))
                ax_train.pie([train_pos, train_neg], labels=['Positif', 'Negatif'], 
                            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
                ax_train.set_title('Training Set')
                st.pyplot(fig_train)
            
            with col_test:
                st.metric("Testing Set", f"{X_test.shape[0]:,} sampel")
                st.write(f"- Positif: {test_pos:,} ({test_pos/X_test.shape[0]*100:.1f}%)")
                st.write(f"- Negatif: {test_neg:,} ({test_neg/X_test.shape[0]*100:.1f}%)")
                
                # Pie chart testing
                fig_test, ax_test = plt.subplots(figsize=(4, 4))
                ax_test.pie([test_pos, test_neg], labels=['Positif', 'Negatif'], 
                           autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
                ax_test.set_title('Testing Set')
                st.pyplot(fig_test)
            
            # Bar chart perbandingan
            fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
            categories = ['Training', 'Testing']
            positif_counts = [train_pos, test_pos]
            negatif_counts = [train_neg, test_neg]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax_comp.bar(x - width/2, positif_counts, width, label='Positif', color='#2ecc71')
            ax_comp.bar(x + width/2, negatif_counts, width, label='Negatif', color='#e74c3c')
            
            ax_comp.set_xlabel('Dataset')
            ax_comp.set_ylabel('Jumlah Sampel')
            ax_comp.set_title('Perbandingan Distribusi Sentimen')
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(categories)
            ax_comp.legend()
            
            # Tambah label di setiap bar
            for i, v in enumerate(positif_counts):
                ax_comp.text(i - width/2, v + max(max(positif_counts), max(negatif_counts))*0.01, 
                           str(v), ha='center')
            for i, v in enumerate(negatif_counts):
                ax_comp.text(i + width/2, v + max(max(positif_counts), max(negatif_counts))*0.01, 
                           str(v), ha='center')
            
            st.pyplot(fig_comp)
            
            # Simpan hasil
            results = {
                ratio_name: {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'test_size': test_size,
                    'random_seed': random_seed
                }
            }
            
            return results
    
    return None

def train_evaluate_svm(results):
    """Training dan evaluasi model SVM"""
    st.header("8. TRAINING DAN EVALUASI MODEL SVM")
    
    if results is None:
        st.warning("⚠️ Silakan lakukan pembagian data terlebih dahulu di section 7!")
        return None, None
    
    # Ambil rasio yang tersedia
    ratio_name = list(results.keys())[0]
    data = results[ratio_name]
    
    st.subheader(f"Rasio: {ratio_name}")
    
    # Pilihan kernel
    st.subheader("⚙️ Konfigurasi Model SVM")
    
    kernel_option = st.selectbox(
        "Pilih kernel SVM:",
        ['linear', 'poly']
    )
    
    # Parameter tambahan berdasarkan kernel
    if kernel_option == 'poly':
        degree = st.slider("Degree", 2, 5, 3)
    else:
        degree = 3
    
    C_value = st.slider("Parameter C", 0.1, 10.0, 1.0, 0.1)
    
    if st.button("🚀 Latih Model SVM"):
        with st.spinner(f"🔄 Training SVM dengan kernel {kernel_option}..."):
            
            # Konfigurasi model
            if kernel_option == 'poly':
                svm_model = SVC(
                    kernel=kernel_option,
                    degree=degree,
                    C=C_value,
                    random_state=42,
                    probability=True
                )
            else:
                svm_model = SVC(
                    kernel=kernel_option,
                    C=C_value,
                    random_state=42,
                    probability=kernel_option == 'rbf'
                )
            
            # Training
            svm_model.fit(data['X_train'], data['y_train'])
            
            # Prediksi
            y_pred = svm_model.predict(data['X_test'])
            
            # Evaluasi
            accuracy = accuracy_score(data['y_test'], y_pred)
            report = classification_report(data['y_test'], y_pred, 
                                          target_names=['negative', 'positive'], 
                                          output_dict=True)
            cm = confusion_matrix(data['y_test'], y_pred)
            
            # Hasil
            result = {
                'model': svm_model,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'y_true': data['y_test'],
                'kernel': kernel_option,
                'C': C_value,
                'degree': degree if kernel_option == 'poly' else None
            }
            
            st.success(f"✅ Training selesai!")
            
            # Tampilkan metrik
            st.subheader("📊 Hasil Evaluasi")
            
            col_acc, col_prec, col_rec, col_f1 = st.columns(4)
            
            with col_acc:
                st.metric("Akurasi", f"{accuracy:.4f}")
            
            with col_prec:
                precision = report['positive']['precision']
                st.metric("Precision (Positif)", f"{precision:.4f}")
            
            with col_rec:
                recall = report['positive']['recall']
                st.metric("Recall (Positif)", f"{recall:.4f}")
            
            with col_f1:
                f1 = report['positive']['f1-score']
                st.metric("F1-Score (Positif)", f"{f1:.4f}")
            
            # Confusion Matrix
            st.subheader("📈 Confusion Matrix")
            
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Negatif', 'Positif'],
                        yticklabels=['Negatif', 'Positif'],
                        ax=ax_cm)
            
            ax_cm.set_title(f'Confusion Matrix\nAkurasi: {accuracy:.4f}')
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            
            st.pyplot(fig_cm)
            
            # Classification Report Detail
            with st.expander("📋 Detail Classification Report"):
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            
            # Simpan hasil
            all_results = {
                ratio_name: {
                    kernel_option: result
                }
            }
            
            accuracy_comparison = [{
                'Rasio': ratio_name,
                'Kernel': kernel_option,
                'Akurasi': accuracy,
                'C': C_value,
                'Degree': degree if kernel_option == 'poly' else 'N/A'
            }]
            
            return all_results, accuracy_comparison
    
    return None, None

def visualize_results(all_results, accuracy_comparison):
    """Visualisasi hasil"""
    st.header("9. VISUALISASI HASIL")
    
    if all_results is None:
        st.warning("⚠️ Silakan latih model terlebih dahulu di section 8!")
        return None
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    
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
            
            st.pyplot(fig)
    
    # Jika ada multiple results untuk perbandingan
    if accuracy_comparison and len(accuracy_comparison) > 1:
        st.subheader("📈 Perbandingan Akurasi")
        
        accuracy_df = pd.DataFrame(accuracy_comparison)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar
        x = np.arange(len(accuracy_df))
        bars = ax.bar(x, accuracy_df['Akurasi'], color='skyblue', alpha=0.7)
        
        ax.set_xlabel('Kernel')
        ax.set_ylabel('Akurasi')
        ax.set_title('Perbandingan Akurasi Model SVM')
        ax.set_xticks(x)
        
        # Buat label yang informatif
        labels = [f"{row['Kernel']}\nC={row['C']}" + 
                 (f"\nDeg={row['Degree']}" if row['Degree'] != 'N/A' else '') 
                 for _, row in accuracy_df.iterrows()]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Tambah nilai di atas bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Tabel perbandingan
        with st.expander("📋 Tabel Perbandingan Detail"):
            st.dataframe(accuracy_df)
    
    return accuracy_df

def classify_new_sentences(all_results, tfidf_vectorizer):
    """Klasifikasi kalimat baru"""
    st.header("10. KLASIFIKASI KALIMAT BARU")
    
    if all_results is None or tfidf_vectorizer is None:
        st.warning("⚠️ Silakan selesaikan training model terlebih dahulu!")
        return None
    
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
                    'accuracy': result['accuracy'],
                    'C': result.get('C', 1.0),
                    'degree': result.get('degree', None)
                }
    
    st.success(f"✨ Model Terbaik: Rasio {best_model_info['ratio']} - Kernel {best_model_info['kernel']}")
    
    col_best1, col_best2, col_best3 = st.columns(3)
    with col_best1:
        st.metric("Akurasi", f"{best_model_info['accuracy']:.4f}")
    with col_best2:
        st.metric("Parameter C", f"{best_model_info['C']}")
    with col_best3:
        degree_display = best_model_info['degree'] if best_model_info['degree'] else 'N/A'
        st.metric("Degree", degree_display)
    
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
        
        # Probabilitas (jika tersedia)
        try:
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities)
        except:
            confidence = None
        
        # Hitung jumlah kata
        word_count = count_words(text)
        word_count_processed = count_words(processed_text)
        
        return sentiment, processed_text, word_count, word_count_processed, confidence
    
    # Contoh kalimat
    st.subheader("📝 Contoh Klasifikasi")
    
    example_sentences = [
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
    
    # Tampilkan contoh dalam selectbox
    example_choice = st.selectbox(
        "Pilih contoh kalimat atau buat sendiri:",
        ["-- Pilih contoh --"] + example_sentences + ["-- Input manual --"]
    )
    
    if example_choice == "-- Input manual --":
        user_input = st.text_area("Masukkan kalimat Anda:", 
                                 "Aplikasi Gojek sangat membantu dalam kehidupan sehari-hari")
    elif example_choice != "-- Pilih contoh --":
        user_input = example_choice
    else:
        user_input = ""
    
    # Input interaktif
    st.subheader("🔍 Analisis Kalimat")
    
    if user_input:
        if st.button("🎯 Analisis Sentimen"):
            sentiment, processed_text, wc_original, wc_processed, confidence = predict_sentiment(
                user_input, 
                best_model_info['model'], 
                tfidf_vectorizer
            )
            
            # Tampilkan hasil
            col_result1, col_result2, col_result3, col_result4 = st.columns(4)
            
            with col_result1:
                st.metric("Jumlah Kata", wc_original)
            
            with col_result2:
                st.metric("Kata Setelah Preprocessing", wc_processed)
            
            with col_result3:
                # Warna berdasarkan sentimen
                color = "🟢" if sentiment == 'POSITIF' else "🔴"
                st.markdown(f"### {color} {sentiment}")
            
            with col_result4:
                if confidence:
                    st.metric("Confidence", f"{confidence:.2%}")
            
            # Detail analisis
            with st.expander("📋 Detail Analisis"):
                st.write("**Kalimat Asli:**")
                st.info(user_input)
                
                st.write("**Setelah Preprocessing:**")
                st.success(processed_text)
                
                st.write("**Model yang Digunakan:**")
                st.write(f"- Rasio: {best_model_info['ratio']}")
                st.write(f"- Kernel: {best_model_info['kernel']}")
                st.write(f"- Parameter C: {best_model_info['C']}")
                if best_model_info['degree']:
                    st.write(f"- Degree: {best_model_info['degree']}")
                st.write(f"- Akurasi: {best_model_info['accuracy']:.4f}")
                
                # Cek kata kunci
                st.write("**Analisis Kata Kunci:**")
                
                # Lexicon untuk pengecekan
                positive_words = ['bagus', 'baik', 'mantap', 'cepat', 'mudah', 'suka', 'puas', 'ramah', 'nyaman']
                negative_words = ['buruk', 'jelek', 'lambat', 'mahal', 'error', 'sulit', 'kecewa', 'lama']
                
                user_lower = user_input.lower()
                pos_found = [word for word in positive_words if word in user_lower]
                neg_found = [word for word in negative_words if word in user_lower]
                
                if pos_found:
                    st.write(f"✅ Kata positif ditemukan: {', '.join(pos_found)}")
                if neg_found:
                    st.write(f"❌ Kata negatif ditemukan: {', '.join(neg_found)}")
    
    # Batch analysis
    st.subheader("📁 Analisis Batch")
    
    batch_input = st.text_area(
        "Masukkan beberapa kalimat (satu per baris):",
        "Aplikasi Gojek sangat membantu\nDriver ramah dan sopan\nPelayanan buruk sekali\nHarga terlalu mahal"
    )
    
    if batch_input and st.button("📊 Analisis Batch"):
        sentences = [s.strip() for s in batch_input.split('\n') if s.strip()]
        
        results = []
        for sentence in sentences:
            sentiment, processed_text, wc_original, wc_processed, confidence = predict_sentiment(
                sentence, 
                best_model_info['model'], 
                tfidf_vectorizer
            )
            
            results.append({
                'Kalimat': sentence,
                'Jml Kata': wc_original,
                'Hasil': sentiment,
                'Confidence': f"{confidence:.2%}" if confidence else "N/A",
                'Warna': '🟢' if sentiment == 'POSITIF' else '🔴'
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Statistik batch
        positif_count = sum(1 for r in results if r['Hasil'] == 'POSITIF')
        negatif_count = len(results) - positif_count
        
        col_batch1, col_batch2 = st.columns(2)
        with col_batch1:
            st.metric("Positif", positif_count)
        with col_batch2:
            st.metric("Negatif", negatif_count)
    
    return best_model_info

def final_statistics(df, sentiment_distribution, tfidf_vectorizer, best_model_info, all_results):
    """Statistik final dan penyimpanan data"""
    st.header("11. STATISTIK FINAL")
    
    if df is None:
        st.warning("⚠️ Tidak ada data yang tersedia!")
        return
    
    st.subheader("📊 Rekapitulasi Proyek")
    
    # Ringkasan utama
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    
    with col_sum1:
        st.metric("Total Data", f"{len(df):,}")
        st.metric("Total Kata Awal", f"{df['word_count'].sum():,}")
        st.metric("Kata Setelah Preprocessing", f"{df['word_count_processed'].sum():,}")
    
    with col_sum2:
        positif_count = sentiment_distribution.get('positive', 0)
        negatif_count = sentiment_distribution.get('negative', 0)
        st.metric("Ulasan Positif", f"{positif_count:,}")
        st.metric("Ulasan Negatif", f"{negatif_count:,}")
        
        if tfidf_vectorizer:
            st.metric("Jumlah Fitur TF-IDF", f"{len(tfidf_vectorizer.get_feature_names_out())}")
    
    with col_sum3:
        if best_model_info:
            st.metric("Akurasi Terbaik", f"{best_model_info['accuracy']:.4f}")
            st.metric("Model Terbaik", f"{best_model_info['ratio']}")
            st.metric("Kernel", best_model_info['kernel'])
    
    # Detail lengkap
    with st.expander("📋 Detail Lengkap"):
        st.write("**1. Statistik Data:**")
        st.write(f"- Total data: {len(df):,}")
        st.write(f"- Data dengan sentimen positif: {positif_count:,} ({positif_count/len(df)*100:.1f}%)")
        st.write(f"- Data dengan sentimen negatif: {negatif_count:,} ({negatif_count/len(df)*100:.1f}%)")
        
        st.write("\n**2. Statistik Jumlah Kata:**")
        st.write(f"- Total kata awal: {df['word_count'].sum():,}")
        st.write(f"- Rata-rata kata per ulasan: {df['word_count'].mean():.1f}")
        st.write(f"- Total kata setelah preprocessing: {df['word_count_processed'].sum():,}")
        st.write(f"- Rata-rata kata setelah preprocessing: {df['word_count_processed'].mean():.1f}")
        
        if tfidf_vectorizer:
            st.write("\n**3. Ekstraksi Fitur:**")
            st.write(f"- Jumlah fitur TF-IDF: {len(tfidf_vectorizer.get_feature_names_out())}")
            st.write(f"- N-gram range: (1, 2) - unigram dan bigram")
        
        if best_model_info:
            st.write("\n**4. Model Terbaik:**")
            st.write(f"- Rasio: {best_model_info['ratio']}")
            st.write(f"- Kernel: {best_model_info['kernel']}")
            st.write(f"- Akurasi: {best_model_info['accuracy']:.4f}")
            st.write(f"- Parameter C: {best_model_info['C']}")
            if best_model_info['degree']:
                st.write(f"- Degree: {best_model_info['degree']}")
    
    # Ringkasan semua model
    if all_results:
        st.subheader("🏆 Ringkasan Semua Model")
        
        summary_data = []
        for ratio_name, ratio_results in all_results.items():
            for kernel_name, result in ratio_results.items():
                summary_data.append({
                    'Rasio': ratio_name,
                    'Kernel': kernel_name,
                    'Akurasi': f"{result['accuracy']:.4f}",
                    'Precision': f"{result['classification_report']['positive']['precision']:.4f}",
                    'Recall': f"{result['classification_report']['positive']['recall']:.4f}",
                    'F1-Score': f"{result['classification_report']['positive']['f1-score']:.4f}",
                    'C': result.get('C', 'N/A'),
                    'Degree': result.get('degree', 'N/A')
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # Opsi untuk menyimpan data
    st.subheader("💾 Simpan Hasil Analisis")
    
    save_option = st.checkbox("Simpan hasil analisis")
    
    if save_option:
        # Nama file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        col_save1, col_save2 = st.columns(2)
        
        with col_save1:
            model_filename = st.text_input("Nama file model:", f"best_svm_model_{timestamp}.pkl")
            results_filename = st.text_input("Nama file hasil:", f"model_results_{timestamp}.json")
        
        with col_save2:
            data_filename = st.text_input("Nama file data:", f"processed_gojek_reviews_{timestamp}.csv")
        
        if st.button("💾 Simpan Semua File"):
            try:
                # Simpan model terbaik
                if best_model_info:
                    with open(model_filename, 'wb') as f:
                        pickle.dump({
                            'model': best_model_info['model'],
                            'vectorizer': tfidf_vectorizer,
                            'accuracy': best_model_info['accuracy'],
                            'ratio': best_model_info['ratio'],
                            'kernel': best_model_info['kernel'],
                            'feature_names': tfidf_vectorizer.get_feature_names_out().tolist() if tfidf_vectorizer else []
                        }, f)
                
                # Simpan semua hasil
                if all_results:
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
                    
                    with open(results_filename, 'w') as f:
                        json.dump(results_summary, f, indent=2)
                
                # Simpan data yang telah diproses
                df_save = df[['content', 'sentiment_label', 'word_count', 'processed_text', 'word_count_processed']].copy()
                df_save.to_csv(data_filename, index=False, encoding='utf-8')
                
                st.success(f"✅ Data dan model berhasil disimpan!")
                st.write(f"📁 **File yang disimpan:**")
                st.write(f"- Model terbaik: `{model_filename}`")
                st.write(f"- Hasil evaluasi: `{results_filename}`")
                st.write(f"- Data proses: `{data_filename}`")
                
                # Download links
                st.download_button(
                    label="📥 Download Model",
                    data=open(model_filename, 'rb').read(),
                    file_name=model_filename,
                    mime="application/octet-stream"
                )
                
                st.download_button(
                    label="📥 Download Hasil",
                    data=open(results_filename, 'rb').read(),
                    file_name=results_filename,
                    mime="application/json"
                )
                
                st.download_button(
                    label="📥 Download Data",
                    data=open(data_filename, 'rb').read(),
                    file_name=data_filename,
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error menyimpan file: {str(e)}")
    
    # Konklusi
    st.subheader("✨ Konklusi")
    
    if best_model_info:
        st.success(
            f"Analisis sentimen berhasil dilakukan dengan akurasi terbaik **{best_model_info['accuracy']:.4f}** "
            f"menggunakan model SVM dengan kernel **{best_model_info['kernel']}** "
            f"pada rasio pembagian data **{best_model_info['ratio']}**."
        )
    else:
        st.info("Proses analisis sentimen telah selesai. Hasil dapat dilihat pada section-section sebelumnya.")

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
    
    # Footer dan info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **📝 Panduan Penggunaan:**
    1. Upload data CSV atau gunakan data contoh
    2. Ikuti alur analisis dari atas ke bawah
    3. Hasil tiap section akan disimpan untuk section berikutnya
    4. Gunakan tombol "Jalankan Semua" untuk otomatisasi
    """)
    
    # Tombol untuk menjalankan semua
    if st.sidebar.button("🚀 Jalankan Semua Analisis", type="primary"):
        with st.spinner("Menjalankan semua analisis..."):
            
            # Section 1: Upload Data
            st.session_state.df = upload_data()
            if st.session_state.df is not None:
                st.success("✅ Section 1: Upload Data selesai!")
                
                # Section 2: Analisis Jumlah Kata
                st.session_state.df = analyze_word_count(st.session_state.df)
                st.success("✅ Section 2: Analisis Jumlah Kata selesai!")
                
                # Section 3: Pelabelan Sentimen
                st.session_state.df, st.session_state.sentiment_distribution = lexicon_sentiment_labeling(st.session_state.df)
                st.success("✅ Section 3: Pelabelan Sentimen selesai!")
                
                # Section 4: WordCloud
                create_wordcloud_viz(st.session_state.df)
                st.success("✅ Section 4: WordCloud selesai!")
                
                # Section 5: Preprocessing Text
                st.session_state.df = text_preprocessing(st.session_state.df)
                st.success("✅ Section 5: Preprocessing Text selesai!")
                
                # Section 6: Ekstraksi Fitur TF-IDF
                st.session_state.X, st.session_state.y, st.session_state.tfidf_vectorizer = tfidf_feature_extraction(st.session_state.df)
                st.success("✅ Section 6: Ekstraksi Fitur TF-IDF selesai!")
                
                # Section 7: Pembagian Data
                st.session_state.results = data_splitting(st.session_state.X, st.session_state.y)
                st.success("✅ Section 7: Pembagian Data selesai!")
                
                # Section 8: Training & Evaluasi SVM
                st.session_state.all_results, st.session_state.accuracy_comparison = train_evaluate_svm(st.session_state.results)
                st.success("✅ Section 8: Training & Evaluasi SVM selesai!")
                
                # Section 10: Klasifikasi Kalimat Baru
                st.session_state.best_model_info = classify_new_sentences(st.session_state.all_results, st.session_state.tfidf_vectorizer)
                st.success("✅ Section 10: Klasifikasi Kalimat Baru selesai!")
                
                st.balloons()
                st.success("✨ Semua analisis berhasil diselesaikan! Lihat hasil di section 11.")
            else:
                st.error("❌ Gagal memuat data. Silakan coba lagi.")

if __name__ == "__main__":
    main()
