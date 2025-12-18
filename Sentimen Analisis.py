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
        
        # Buat data contoh 8000 baris dengan kalimat rancu
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
            "tidak terlalu mahal untuk pelayanannya",
            "kurang begitu mahal dibanding yang lain",
            "tidak jelek untuk harga segitu",
            "lumayan bagus untuk aplikasi transportasi",
            "cukup memuaskan untuk harga murah",
            "tidak buruk sama sekali",
            "cukup membantu walaupun sederhana",
            "tidak terlalu sulit digunakan",
            "kurang lambat dari yang saya kira",
            "tidak terlalu buruk untuk pemula"
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
            "tidak begitu bagus seperti yang diharapkan",
            "kurang memuaskan untuk harga mahal",
            "tidak cepat seperti iklannya",
            "cukup mahal untuk kualitas biasa",
            "lumayan buruk untuk aplikasi populer",
            "tidak membantu sama sekali",
            "kurang ramah dalam pelayanan",
            "tidak praktis seperti yang dikatakan",
            "cukup mengecewakan untuk harga segitu",
            "tidak nyaman untuk perjalanan jauh"
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
                "pengalaman pribadi", "baru saja mencoba", "setelah update",
                "sebenarnya", "mungkin", "kurang lebih", "agak", "sedikit"
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
        
        # Tampilkan contoh data termasuk yang rancu
        with st.expander("Contoh Data (10 baris pertama)"):
            for i in range(min(10, len(df))):
                content = str(df['content'].iloc[i])
                sentiment = df['sentimen'].iloc[i] if 'sentimen' in df.columns else 'N/A'
                
                # Cek apakah kalimat rancu
                is_ambiguous = any(word in content.lower() for word in ['kurang', 'tidak', 'cukup', 'lumayan', 'agak', 'sedikit'])
                
                st.write(f"**Data {i+1}:**")
                st.write(f"- Konten: {content}")
                if is_ambiguous:
                    st.write(f"- ⚠️ **Kalimat Rancu**: Ya")
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
    
    # Analisis kalimat rancu
    st.subheader("🔍 ANALISIS KALIMAT RANCU")
    
    # Kata-kata yang menandakan kalimat rancu
    ambiguous_words = ['kurang', 'tidak', 'cukup', 'lumayan', 'agak', 'sedikit', 
                      'mungkin', 'sebenarnya', 'kurang lebih', 'agaknya']
    
    # Identifikasi kalimat rancu
    df['is_ambiguous'] = df['content'].apply(
        lambda x: any(word in x.lower() for word in ambiguous_words)
    )
    
    ambiguous_count = df['is_ambiguous'].sum()
    ambiguous_percentage = (ambiguous_count / len(df)) * 100
    
    col_amb1, col_amb2 = st.columns(2)
    with col_amb1:
        st.metric("Kalimat Rancu", f"{ambiguous_count:,}")
    with col_amb2:
        st.metric("Persentase Rancu", f"{ambiguous_percentage:.1f}%")
    
    # Distribusi sentimen pada kalimat rancu
    if 'sentimen' in df.columns:
        ambiguous_by_sentiment = df[df['is_ambiguous']].groupby('sentimen').size()
        
        if not ambiguous_by_sentiment.empty:
            st.write("**Distribusi Sentimen pada Kalimat Rancu:**")
            for sentiment, count in ambiguous_by_sentiment.items():
                percentage = (count / ambiguous_count) * 100
                st.write(f"- {sentiment}: {count} ({percentage:.1f}%)")
    
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
    
    # Contoh kalimat rancu
    with st.expander("📝 Contoh Kalimat Rancu"):
        ambiguous_samples = df[df['is_ambiguous']].head(5)
        for i, (_, row) in enumerate(ambiguous_samples.iterrows()):
            st.write(f"**Contoh {i+1}:**")
            st.write(f"- Kalimat: {row['content']}")
            if 'sentimen' in df.columns:
                st.write(f"- Sentimen: {row['sentimen']}")
            st.write(f"- Jumlah kata: {row['word_count']}")
            st.write("---")
    
    return df

def lexicon_sentiment_labeling(df):
    """Pelabelan sentimen dengan lexicon yang diperbaiki untuk kalimat rancu"""
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
    
    # Kata negasi yang membalikkan makna
    negation_words = ['tidak', 'kurang', 'bukan', 'jangan', 'tanpa', 'belum']
    
    # Kata pengubah intensitas
    intensity_modifiers = {
        'sangat': 2.0,
        'sekali': 2.0,
        'banget': 2.0,
        'agak': 0.5,
        'sedikit': 0.5,
        'cukup': 0.7,
        'lumayan': 0.7,
        'kurang': 0.3,
        'tidak': 0.0,
        'amat': 2.0,
        'terlalu': 1.5,
        'hampir': 0.8
    }
    
    # Fungsi untuk pelabelan sentimen dengan handling kalimat rancu
    def lexicon_sentiment_analysis_advanced(text):
        if not isinstance(text, str):
            return 'neutral'
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Inisialisasi skor
        positive_score = 0
        negative_score = 0
        
        # Analisis setiap kata dengan konteks
        for i, word in enumerate(words):
            intensity = 1.0
            is_negated = False
            
            # Cek kata sebelumnya untuk negasi
            if i > 0:
                prev_word = words[i-1]
                if prev_word in negation_words:
                    is_negated = True
                
                # Cek intensitas dari kata sebelumnya
                if prev_word in intensity_modifiers:
                    intensity = intensity_modifiers[prev_word]
            
            # Cek jika kata ini adalah kata positif
            if word in positive_words:
                if is_negated:
                    negative_score += intensity  # Negasi membalikkan ke negatif
                else:
                    positive_score += intensity
            
            # Cek jika kata ini adalah kata negatif
            elif word in negative_words:
                if is_negated:
                    positive_score += intensity  # Negasi membalikkan ke positif
                else:
                    negative_score += intensity
        
        # Hitung kata kunci yang sangat kuat
        strong_positive = any(word in text_lower for word in ['sangat baik', 'sangat bagus', 'luar biasa', 'terbaik'])
        strong_negative = any(word in text_lower for word in ['sangat buruk', 'sangat jelek', 'parah sekali', 'penipu'])
        
        # Tambah bobot untuk kata kunci kuat
        if strong_positive:
            positive_score += 2.0
        if strong_negative:
            negative_score += 2.0
        
        # Decision logic dengan threshold
        if positive_score == negative_score:
            # Jika sama, cek konteks keseluruhan
            if 'tidak' in text_lower and 'mahal' in text_lower:
                return 'positive'  # "tidak mahal" biasanya positif
            elif 'kurang' in text_lower and 'mahal' in text_lower:
                return 'positive'  # "kurang mahal" biasanya positif
            elif 'cukup' in text_lower or 'lumayan' in text_lower:
                return 'positive'  # Kata moderat cenderung positif
            else:
                return 'positive'  # Default ke positif
        
        return 'positive' if positive_score > negative_score else 'negative'
    
    # Terapkan pelabelan advanced
    with st.spinner("🔄 Melabeli sentimen (dengan handling kalimat rancu)..."):
        df['sentiment_label'] = df['content'].apply(lexicon_sentiment_analysis_advanced)
    
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
    
    # Analisis khusus untuk kalimat rancu
    st.subheader("🔍 ANALISIS KALIMAT RANCU SETELAH PELABELAN")
    
    if 'is_ambiguous' in df.columns:
        ambiguous_sentiment = df[df['is_ambiguous']]['sentiment_label'].value_counts()
        ambiguous_total = df['is_ambiguous'].sum()
        
        if ambiguous_total > 0:
            col_amb1, col_amb2 = st.columns(2)
            with col_amb1:
                amb_positif = ambiguous_sentiment.get('positive', 0)
                amb_pos_pct = (amb_positif/ambiguous_total*100) if ambiguous_total > 0 else 0
                st.metric("Rancu → Positif", f"{amb_positif:,}", f"({amb_pos_pct:.1f}%)")
            with col_amb2:
                amb_negatif = ambiguous_sentiment.get('negative', 0)
                amb_neg_pct = (amb_negatif/ambiguous_total*100) if ambiguous_total > 0 else 0
                st.metric("Rancu → Negatif", f"{amb_negatif:,}", f"({amb_neg_pct:.1f}%)")
    
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
    
    # Contoh hasil pelabelan untuk kalimat rancu
    with st.expander("📝 Contoh Pelabelan Kalimat Rancu"):
        if 'is_ambiguous' in df.columns:
            ambiguous_examples = df[df['is_ambiguous']].head(5)
            for i, (_, row) in enumerate(ambiguous_examples.iterrows()):
                st.write(f"**Contoh {i+1}:**")
                st.write(f"- Kalimat: `{row['content']}`")
                st.write(f"- Hasil Pelabelan: **{row['sentiment_label']}**")
                
                # Analisis detail
                text_lower = row['content'].lower()
                has_negation = any(neg in text_lower for neg in negation_words)
                has_intensity = any(int_word in text_lower for int_word in intensity_modifiers.keys())
                
                if has_negation:
                    st.write(f"  ⚠️ Mengandung negasi: Ya")
                if has_intensity:
                    st.write(f"  📊 Mengandung pengubah intensitas: Ya")
                
                st.write("---")
    
    return df, sentiment_distribution

# Untuk menghemat ruang, saya akan menunjukkan hanya bagian yang diubah...
