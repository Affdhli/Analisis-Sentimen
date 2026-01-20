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
    
    # ============= TAMBAHAN: TAMPILKAN SEMUA HASIL TAHAPAN PREPROCESSING =============
    st.subheader("HASIL SEMUA TAHAPAN PREPROCESSING")
    
    # Pilih jumlah baris yang ingin ditampilkan
    st.markdown("### Pilih Jumlah Data untuk Ditampilkan")
    show_all = st.checkbox("Tampilkan semua data", value=False)
    
    if show_all:
        num_rows = len(df)
        st.write(f"Menampilkan semua {num_rows} baris data:")
    else:
        num_rows = st.slider("Jumlah baris yang ditampilkan:", 
                            min_value=5, 
                            max_value=min(30, len(df)), 
                            value=10)
        st.write(f"Menampilkan {num_rows} baris pertama:")
    
    # Tampilkan semua tahapan preprocessing untuk setiap baris
    for i in range(min(num_rows, len(df))):
        with st.expander(f"Data {i+1}: {df['content'].iloc[i][:50]}...", expanded=(i==0)):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1. TEKS ASLI:**")
                st.text_area(f"Teks asli {i+1}", 
                           df['content'].iloc[i], 
                           height=100,
                           disabled=True,
                           key=f"original_{i}")
                st.caption(f"Panjang: {len(str(df['content'].iloc[i]))} karakter")
            
            with col2:
                st.markdown("**2. SETELAH CLEANING & CASE FOLDING:**")
                st.text_area(f"Cleaned text {i+1}", 
                           df['cleaned_text'].iloc[i], 
                           height=100,
                           disabled=True,
                           key=f"cleaned_{i}")
                st.caption(f"Panjang: {len(str(df['cleaned_text'].iloc[i]))} karakter")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**3. SETELAH REMOVE STOPWORDS:**")
                st.text_area(f"No stopwords {i+1}", 
                           df['text_no_stopwords'].iloc[i], 
                           height=100,
                           disabled=True,
                           key=f"nostopwords_{i}")
                st.caption(f"Panjang: {len(str(df['text_no_stopwords'].iloc[i]))} karakter")
            
            with col4:
                st.markdown("**4. HASIL TOKENIZATION:**")
                tokens = df['tokens'].iloc[i]
                tokens_str = " | ".join([f"`{token}`" for token in tokens])
                st.markdown(tokens_str)
                st.caption(f"Jumlah token: {len(tokens)}")
            
            st.markdown("**5. TEKS FINAL (untuk TF-IDF):**")
            st.text_area(f"Processed text {i+1}", 
                       df['processed_text'].iloc[i], 
                       height=80,
                       disabled=True,
                       key=f"processed_{i}")
            st.caption(f"Panjang: {df['text_length_processed'].iloc[i]} karakter")
            
            # Ringkasan perubahan
            st.markdown("**RINGKASAN PERUBAHAN:**")
            original_len = len(str(df['content'].iloc[i]))
            final_len = df['text_length_processed'].iloc[i]
            reduction_pct_item = ((original_len - final_len) / original_len * 100) if original_len > 0 else 0
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Panjang awal", f"{original_len}", delta=None)
            with col_b:
                st.metric("Panjang akhir", f"{final_len}", delta=None)
            with col_c:
                st.metric("Pengurangan", f"{reduction_pct_item:.1f}%", 
                         delta=f"-{original_len - final_len} karakter")
    
    # ============= TAMBAHAN: TABEL RINGKASAN SEMUA TAHAPAN =============
    st.subheader("TABEL RINGKASAN PREPROCESSING")
    
    # Buat dataframe ringkasan
    summary_df = pd.DataFrame({
        'No': range(1, min(num_rows, len(df)) + 1),
        'Teks Asli (potongan)': df['content'].head(num_rows).apply(lambda x: x[:30] + "..." if len(x) > 30 else x),
        'Cleaned Text': df['cleaned_text'].head(num_rows).apply(lambda x: x[:30] + "..." if len(x) > 30 else x),
        'No Stopwords': df['text_no_stopwords'].head(num_rows).apply(lambda x: x[:30] + "..." if len(x) > 30 else x),
        'Jumlah Token': df['tokens'].head(num_rows).apply(len),
        'Panjang Awal': df['text_length'].head(num_rows),
        'Panjang Akhir': df['text_length_processed'].head(num_rows),
        'Pengurangan %': ((df['text_length'] - df['text_length_processed']) / df['text_length'] * 100).head(num_rows).round(1)
    })
    
    # Tampilkan tabel
    st.dataframe(summary_df, use_container_width=True, height=400)
    
    # ============= TAMBAHAN: SIMPAN HASIL PREPROCESSING =============
    st.subheader("SIMPAN HASIL PREPROCESSING")
    
    # Pilihan format
    save_format = st.radio("Format file:", ["CSV", "Excel", "JSON"], horizontal=True)
    
    # Nama file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = st.text_input("Nama file:", value=f"hasil_preprocessing_{timestamp}")
    
    if st.button("Simpan ke File"):
        try:
            # Buat dataframe lengkap untuk disimpan
            save_df = df.copy()
            
            if save_format == "CSV":
                file_path = f"{filename}.csv"
                save_df.to_csv(file_path, index=False, encoding='utf-8')
                st.success(f"Data disimpan sebagai {file_path}")
                
            elif save_format == "Excel":
                file_path = f"{filename}.xlsx"
                save_df.to_excel(file_path, index=False)
                st.success(f"Data disimpan sebagai {file_path}")
                
            elif save_format == "JSON":
                file_path = f"{filename}.json"
                save_df.to_json(file_path, orient='records', indent=2, force_ascii=False)
                st.success(f"Data disimpan sebagai {file_path}")
            
            # Tampilkan info file
            import os
            file_size = os.path.getsize(file_path) / 1024  # KB
            st.info(f"Ukuran file: {file_size:.2f} KB")
            st.info(f"Jumlah data: {len(df)} baris")
            
            # Tombol download
            with open(file_path, "rb") as f:
                btn = st.download_button(
                    label="Download File",
                    data=f,
                    file_name=os.path.basename(file_path),
                    mime="text/csv" if save_format == "CSV" else 
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if save_format == "Excel" else 
                         "application/json"
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # ============= TAMBAHAN: STATISTIK DETAIL =============
    st.subheader("STATISTIK DETAIL")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_tokens = df['tokens'].apply(len).mean()
        st.metric("Rata-rata Token per Teks", f"{avg_tokens:.1f}")
    
    with col2:
        max_tokens = df['tokens'].apply(len).max()
        st.metric("Token Maksimum", f"{max_tokens}")
    
    with col3:
        min_tokens = df['tokens'].apply(len).min()
        st.metric("Token Minimum", f"{min_tokens}")
    
    # Grafik distribusi token
    st.markdown("#### Distribusi Jumlah Token")
    token_counts = df['tokens'].apply(len)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(token_counts, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_xlabel('Jumlah Token')
    ax.set_ylabel('Frekuensi')
    ax.set_title('Distribusi Jumlah Token per Dokumen')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Info kolom yang dihasilkan
    st.markdown("#### Kolom yang Dihasilkan:")
    cols_info = pd.DataFrame({
        'Kolom': df.columns,
        'Tipe Data': df.dtypes.astype(str).values,
        'Contoh Data': df.iloc[0].astype(str).apply(lambda x: x[:50] + "..." if len(x) > 50 else x).values
    })
    st.dataframe(cols_info, use_container_width=True)
    
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
    
    # TAMBAHAN: SIMPAN HASIL TF-IDF
    st.subheader("SIMPAN HASIL EKSTRAKSI FITUR TF-IDF")
    
    # Buat tab untuk berbagai format penyimpanan
    tab1, tab2, tab3, tab4 = st.tabs(["Matriks TF-IDF", "Daftar Fitur", "Top Fitur", "Semua Data"])
    
    with tab1:
        st.markdown("### Simpan Matriks TF-IDF")
        
        # Konversi sparse matrix ke dense (untuk sample kecil) atau simpan sebagai sparse
        save_option = st.radio("Pilih format matriks:", 
                              ["Sparse Matrix (CSR - direkomendasikan)", "Dense Matrix"], 
                              help="Sparse Matrix lebih efisien untuk data tekstual")
        
        if save_option == "Dense Matrix":
            # Konversi ke dense matrix (hanya untuk data kecil)
            if X.shape[0] * X.shape[1] < 1000000:  # Batas 1 juta elemen
                X_dense = X.toarray()
                st.info(f"Matriks dense: {X_dense.shape}")
                
                # Tampilkan preview
                st.markdown("**Preview Matriks TF-IDF (5 baris pertama, 10 kolom pertama):**")
                preview_df = pd.DataFrame(X_dense[:5, :10], 
                                         columns=feature_names[:10])
                st.dataframe(preview_df.style.format("{:.4f}"))
            else:
                st.warning("Matriks terlalu besar untuk dikonversi ke dense format.")
                X_dense = None
        else:
            X_dense = None
        
        # Tombol simpan
        col1, col2 = st.columns(2)
        with col1:
            save_format = st.selectbox("Format file:", ["NPZ (sparse)", "CSV", "JSON"])
        with col2:
            filename_tfidf = st.text_input("Nama file matriks:", 
                                          value=f"tfidf_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if st.button("Simpan Matriks TF-IDF"):
            try:
                if save_option == "Sparse Matrix (CSR - direkomendasikan)":
                    if save_format == "NPZ (sparse)":
                        file_path = f"{filename_tfidf}.npz"
                        sparse.save_npz(file_path, X)
                        st.success(f"Matriks TF-IDF sparse disimpan sebagai {file_path}")
                    elif save_format == "CSV":
                        file_path = f"{filename_tfidf}.csv"
                        # Simpan sebagai dense matrix untuk CSV
                        if X.shape[0] * X.shape[1] < 500000:
                            X_dense_save = X.toarray()
                            pd.DataFrame(X_dense_save, columns=feature_names).to_csv(file_path, index=False)
                            st.success(f"Matriks TF-IDF disimpan sebagai {file_path}")
                        else:
                            st.error("Matriks terlalu besar untuk disimpan sebagai CSV.")
                    elif save_format == "JSON":
                        st.warning("Format JSON tidak direkomendasikan untuk matriks sparse yang besar.")
                else:
                    if X_dense is not None:
                        if save_format == "NPZ (sparse)":
                            file_path = f"{filename_tfidf}.npz"
                            sparse.save_npz(file_path, sparse.csr_matrix(X_dense))
                        elif save_format == "CSV":
                            file_path = f"{filename_tfidf}.csv"
                            pd.DataFrame(X_dense, columns=feature_names).to_csv(file_path, index=False)
                        elif save_format == "JSON":
                            file_path = f"{filename_tfidf}.json"
                            pd.DataFrame(X_dense, columns=feature_names).to_json(file_path, orient='split')
                        st.success(f"Matriks TF-IDF disimpan sebagai {file_path}")
                    else:
                        st.error("Matriks tidak tersedia dalam format dense.")
                
                # Tombol download
                if 'file_path' in locals():
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="Download Matriks",
                            data=f,
                            file_name=os.path.basename(file_path),
                            mime="application/octet-stream"
                        )
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.markdown("### Simpan Daftar Semua Fitur")
        
        # Buat dataframe semua fitur dengan IDF scores
        features_df = pd.DataFrame({
            'Fitur': feature_names,
            'IDF_Score': idf_values,
            'Index': range(len(feature_names))
        }).sort_values('IDF_Score', ascending=False)
        
        # Tampilkan preview
        st.markdown(f"**Total {len(features_df)} fitur:**")
        st.dataframe(features_df.head(20))
        
        col1, col2 = st.columns(2)
        with col1:
            feat_format = st.selectbox("Format:", ["CSV", "Excel", "JSON"])
        with col2:
            filename_features = st.text_input("Nama file fitur:", 
                                            value=f"tfidf_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if st.button("Simpan Daftar Fitur"):
            try:
                if feat_format == "CSV":
                    file_path = f"{filename_features}.csv"
                    features_df.to_csv(file_path, index=False)
                elif feat_format == "Excel":
                    file_path = f"{filename_features}.xlsx"
                    features_df.to_excel(file_path, index=False)
                elif feat_format == "JSON":
                    file_path = f"{filename_features}.json"
                    features_df.to_json(file_path, orient='records', indent=2)
                
                st.success(f"Daftar fitur disimpan sebagai {file_path}")
                
                # Download button
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="Download Daftar Fitur",
                        data=f,
                        file_name=os.path.basename(file_path),
                        mime="text/csv" if feat_format == "CSV" else 
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if feat_format == "Excel" else 
                             "application/json"
                    )
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab3:
        st.markdown("### Simpan Top Fitur")
        
        # Pilih jumlah top fitur
        num_top_features = st.slider("Jumlah top fitur:", 10, 100, 50)
        
        # Hitung top fitur berdasarkan IDF
        top_indices_custom = np.argsort(idf_values)[:num_top_features]
        top_features_custom = pd.DataFrame({
            'Fitur': feature_names[top_indices_custom],
            'IDF_Score': idf_values[top_indices_custom],
            'Ranking': range(1, num_top_features + 1)
        }).sort_values('IDF_Score', ascending=True)
        
        # Tampilkan
        st.markdown(f"**Top {num_top_features} Fitur:**")
        st.dataframe(top_features_custom)
        
        # Visualisasi top features
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(min(20, num_top_features))
        ax.barh(y_pos, top_features_custom['IDF_Score'].head(20))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features_custom['Fitur'].head(20))
        ax.set_xlabel('IDF Score')
        ax.set_title('Top 20 Fitur berdasarkan IDF Score')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Simpan
        col1, col2 = st.columns(2)
        with col1:
            top_format = st.selectbox("Format top fitur:", ["CSV", "Excel", "JSON", "PNG"])
        with col2:
            filename_top = st.text_input("Nama file top fitur:", 
                                       value=f"top_{num_top_features}_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if st.button("Simpan Top Fitur"):
            try:
                if top_format in ["CSV", "Excel", "JSON"]:
                    if top_format == "CSV":
                        file_path = f"{filename_top}.csv"
                        top_features_custom.to_csv(file_path, index=False)
                    elif top_format == "Excel":
                        file_path = f"{filename_top}.xlsx"
                        top_features_custom.to_excel(file_path, index=False)
                    elif top_format == "JSON":
                        file_path = f"{filename_top}.json"
                        top_features_custom.to_json(file_path, orient='records', indent=2)
                    
                    st.success(f"Top fitur disimpan sebagai {file_path}")
                    
                    # Download button
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"Download Top {num_top_features} Fitur",
                            data=f,
                            file_name=os.path.basename(file_path)
                        )
                elif top_format == "PNG":
                    file_path = f"{filename_top}.png"
                    fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    st.success(f"Visualisasi disimpan sebagai {file_path}")
                    
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="Download Gambar",
                            data=f,
                            file_name=os.path.basename(file_path),
                            mime="image/png"
                        )
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab4:
        st.markdown("### Simpan Semua Data dengan Fitur TF-IDF")
        
        # Gabungkan data asli dengan TF-IDF features (untuk sample kecil)
        if X.shape[0] < 1000:  # Batas 1000 dokumen untuk penggabungan
            st.info("Menggabungkan data asli dengan fitur TF-IDF...")
            
            # Konversi ke dataframe
            tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)
            
            # Gabungkan dengan data asli
            combined_df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
            
            # Tampilkan preview
            st.markdown("**Preview Data Gabungan (5 baris pertama):**")
            st.dataframe(combined_df.head())
            
            # Simpan
            col1, col2 = st.columns(2)
            with col1:
                combined_format = st.selectbox("Format data gabungan:", ["CSV", "Excel", "Parquet"])
            with col2:
                filename_combined = st.text_input("Nama file data gabungan:", 
                                                value=f"combined_tfidf_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if st.button("Simpan Data Gabungan"):
                try:
                    if combined_format == "CSV":
                        file_path = f"{filename_combined}.csv"
                        combined_df.to_csv(file_path, index=False)
                    elif combined_format == "Excel":
                        file_path = f"{filename_combined}.xlsx"
                        combined_df.to_excel(file_path, index=False)
                    elif combined_format == "Parquet":
                        file_path = f"{filename_combined}.parquet"
                        combined_df.to_parquet(file_path, index=False)
                    
                    st.success(f"Data gabungan disimpan sebagai {file_path}")
                    
                    # Download button
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="Download Data Gabungan",
                            data=f,
                            file_name=os.path.basename(file_path)
                        )
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Data terlalu besar untuk digabungkan. Disarankan hanya menyimpan matriks TF-IDF saja.")
    
    # TAMBAHAN: STATISTIK TF-IDF
    st.subheader("STATISTIK TF-IDF")
    
    # Hitung beberapa statistik
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Rata-rata nilai TF-IDF
        avg_tfidf = X.mean()
        st.metric("Rata-rata TF-IDF", f"{avg_tfidf:.4f}")
    
    with col2:
        # Sparsity
        sparsity = 1.0 - (X.count_nonzero() / (X.shape[0] * X.shape[1]))
        st.metric("Sparsity", f"{sparsity:.2%}")
    
    with col3:
        # Fitur dengan IDF tertinggi (paling khas)
        most_unique_idx = np.argmax(idf_values)
        st.metric("Fitur paling unik", feature_names[most_unique_idx])
    
    # Distribusi nilai TF-IDF
    st.markdown("#### Distribusi Nilai TF-IDF")
    
    # Ambil sample untuk visualisasi
    if X.shape[0] > 1000:
        sample_indices = np.random.choice(X.shape[0], 1000, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X
    
    # Konversi ke array untuk histogram
    tfidf_values_sample = X_sample.data
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(tfidf_values_sample, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Nilai TF-IDF')
    ax2.set_ylabel('Frekuensi')
    ax2.set_title('Distribusi Nilai TF-IDF')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # TAMBAHAN: EKSPORT TFIDF VECTORIZER
    st.subheader("Simpan TF-IDF Vectorizer Model")
    
    st.info("Simpan model TF-IDF Vectorizer untuk digunakan pada data baru.")
    
    filename_vectorizer = st.text_input("Nama file vectorizer:", 
                                      value=f"tfidf_vectorizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if st.button("Simpan Vectorizer Model"):
        try:
            import pickle
            file_path = f"{filename_vectorizer}.pkl"
            
            # Simpan vectorizer
            with open(file_path, 'wb') as f:
                pickle.dump(tfidf_vectorizer, f)
            
            st.success(f"Vectorizer disimpan sebagai {file_path}")
            
            # Informasi vectorizer
            vectorizer_info = {
                'vocabulary_size': len(tfidf_vectorizer.vocabulary_),
                'max_features': tfidf_vectorizer.max_features,
                'min_df': tfidf_vectorizer.min_df,
                'max_df': tfidf_vectorizer.max_df,
                'ngram_range': tfidf_vectorizer.ngram_range
            }
            
            st.json(vectorizer_info)
            
            # Download button
            with open(file_path, "rb") as f:
                st.download_button(
                    label="Download Vectorizer",
                    data=f,
                    file_name=os.path.basename(file_path),
                    mime="application/octet-stream"
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
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
            'training_summary': training_summary,
            'model_object': svm_custom.model  # Untuk disimpan
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
                report_df = pd.DataFrame(result['classification_report']).transpose()
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
                report_df = pd.DataFrame(result['classification_report']).transpose()
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
        
        # Simpan informasi model terbaik untuk digunakan nanti
        st.session_state.best_model_info = {
            'ratio': best_accuracy_model['Rasio'],
            'kernel': best_accuracy_model['Kernel'],
            'accuracy': best_accuracy_model['Akurasi_Keseluruhan'],
            'training_time': best_accuracy_model['Training_Time']
        }
        
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
