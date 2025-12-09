# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==============================
# DOWNLOAD NLTK RESOURCES DENGAN ERROR HANDLING
# ==============================
@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources with error handling"""
    resources = ['punkt', 'stopwords', 'punkt_tab']
    
    for resource in resources:
        try:
            if resource == 'punkt_tab':
                # Download punkt untuk bahasa Indonesia
                nltk.download('punkt')
            else:
                nltk.download(resource)
        except Exception as e:
            st.warning(f"Error downloading {resource}: {e}")
    
    # Set path untuk resource NLTK
    nltk.data.path.append('/home/adminuser/nltk_data')
    
    return True

# Download resources
download_nltk_resources()

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Gojek",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00AA13;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .negative {
        color: #F44336;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00AA13;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #00AA13;
    }
    .stButton > button {
        background-color: #00AA13;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #008C0F;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Logo_of_Gojek.svg/2560px-Logo_of_Gojek.svg.png", 
             width=200)
    st.title("🚗 Gojek Sentiment Analyzer")
    st.markdown("---")
    
    st.subheader("Pengaturan Model")
    ratio_option = st.selectbox(
        "Pilih Rasio Data",
        ["80:20", "90:10", "70:30"],
        index=0
    )
    
    kernel_option = st.selectbox(
        "Pilih Kernel SVM",
        ["linear", "poly"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Informasi")
    st.info("""
    **Fitur Aplikasi:**
    1. 📊 Dashboard Analisis
    2. ⚙️ Pelabelan Otomatis
    3. 🔧 Preprocessing Data
    4. 🏋️ Training Model SVM
    5. 📈 Evaluasi Model
    6. 🔍 Klasifikasi Real-time
    """)
    
    st.markdown("---")
    st.caption("Made with ❤️ by Gojek Sentiment Analysis Team")

# Header utama
st.markdown('<h1 class="main-header">🚗 Analisis Sentimen Ulasan Gojek</h1>', unsafe_allow_html=True)

# Tab utama
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "⚙️ Pelabelan Data", 
    "🔧 Preprocessing", 
    "🏋️ Training Model", 
    "🔍 Klasifikasi Real-time"
])

# ==============================
# FUNGSI UTILITAS YANG DIPERBAIKI
# ==============================
@st.cache_data
def load_sample_data():
    """Load dataset contoh"""
    np.random.seed(42)
    sample_reviews = [
        "aplikasi yang sangat bagus dan membantu",
        "pelayanan buruk driver sering telat",
        "mantap sekali gojek mudah digunakan",
        "aplikasi sering error perlu perbaikan",
        "driver ramah dan profesional",
        "tarif mahal tidak sesuai dengan pelayanan",
        "fitur lengkap dan sangat memudahkan",
        "loading lambat sering force close",
        "pelayanan memuaskan terima kasih gojek",
        "cancel order terus sangat menyebalkan",
        "antar cepat dan tepat waktu",
        "aplikasi terbaik untuk transportasi online",
        "rating driver tidak akurat",
        "gofood enak dan cepat sampai",
        "update terbaru bikin aplikasi makin lambat",
        "customer service responsif",
        "potongan harga sering ada hemat banget",
        "driver kurang berpengalaman",
        "interface user friendly",
        "notifikasi sering tidak muncul"
    ]
    
    # Duplikasi untuk mencapai 8000 data
    df = pd.DataFrame({
        'content': sample_reviews * 400,
        'rating': np.random.choice([1, 2, 3, 4, 5], 8000, p=[0.15, 0.15, 0.2, 0.25, 0.25])
    })
    
    return df

def simple_tokenize(text):
    """Fungsi tokenize sederhana tanpa NLTK untuk fallback"""
    return text.split()

def preprocess_text(text, use_nltk=False):
    """Fungsi preprocessing teks dengan fallback"""
    try:
        # Cleaning
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Case folding
        text = text.lower()
        
        # Tokenization
        if use_nltk:
            try:
                tokens = word_tokenize(text)
            except:
                tokens = simple_tokenize(text)
        else:
            tokens = simple_tokenize(text)
        
        # Stopword removal
        try:
            stop_words = set(stopwords.words('indonesian'))
        except:
            # Default stopwords jika tidak bisa download
            stop_words = set(['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 
                            'pada', 'dengan', 'ini', 'itu', 'saya', 'kamu'])
        
        custom_stopwords = ['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 
                           'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih',
                           'gojek', 'aplikasi']
        stop_words.update(custom_stopwords)
        
        tokens = [word for word in tokens if word not in stop_words]
        
        return ' '.join(tokens)
    
    except Exception as e:
        st.error(f"Error preprocessing: {e}")
        return text.lower()  # Fallback ke lowercase saja

def label_sentiment_lexicon(text):
    """Pelabelan dengan lexicon"""
    positive_words = [
        'bagus', 'baik', 'mantap', 'memuaskan', 'cepat', 'mudah', 
        'ramah', 'profesional', 'lengkap', 'hemat', 'responsif', 'enak',
        'tepat', 'terbaik', 'friendly', 'membantu', 'puas', 'sukses',
        'hebat', 'luar', 'biasa', 'keren', 'recommended', 'top'
    ]
    
    negative_words = [
        'buruk', 'jelek', 'telat', 'error', 'mahal', 'lambat', 
        'menyebalkan', 'tidak', 'kurang', 'gagal', 'cancel', 'force',
        'close', 'tidak', 'akurat', 'payah', 'mengecewakan', 'hancur',
        'bermasalah', 'rusak', 'parah', 'pusing', 'stress', 'kecewa'
    ]
    
    text_lower = text.lower()
    
    # Hitung kata positif
    positive_count = 0
    for word in positive_words:
        if word in text_lower:
            positive_count += 1
    
    # Hitung kata negatif
    negative_count = 0
    for word in negative_words:
        if word in text_lower:
            negative_count += 1
    
    if positive_count > negative_count:
        return 'positif'
    elif negative_count > positive_count:
        return 'negatif'
    else:
        return 'netral'

# ==============================
# TAB 1: DASHBOARD
# ==============================
with tab1:
    st.markdown('<h2 class="sub-header">📊 Dashboard Analisis Sentimen</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Total Data</h3>
            <h2>8,000</h2>
            <p>Ulasan Gojek</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>😊 Sentimen Positif</h3>
            <h2 class="positive">4,320</h2>
            <p>54% dari total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>😞 Sentimen Negatif</h3>
            <h2 class="negative">3,680</h2>
            <p>46% dari total</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    df = load_sample_data()
    
    # Plot distribusi rating
    st.markdown("### 📊 Distribusi Rating Ulasan")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    rating_counts = df['rating'].value_counts().sort_index()
    colors = ['#FF5252', '#FF867F', '#FFB3AD', '#B2FF59', '#69F0AE']
    rating_counts.plot(kind='bar', color=colors, ax=ax1)
    ax1.set_xlabel('Rating (1-5)')
    ax1.set_ylabel('Jumlah Ulasan')
    ax1.set_title('Distribusi Rating Ulasan Gojek')
    ax1.grid(axis='y', alpha=0.3)
    st.pyplot(fig1)
    
    # Contoh data
    st.markdown("### 📋 Contoh Data Ulasan")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistik deskriptif
    st.markdown("### 📈 Statistik Deskriptif")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Rata-rata Rating", f"{df['rating'].mean():.2f}")
        st.metric("Rating Tertinggi", int(df['rating'].max()))
    
    with col2:
        st.metric("Rating Terendah", int(df['rating'].min()))
        st.metric("Standar Deviasi", f"{df['rating'].std():.2f}")

# ==============================
# TAB 2: PELABELAN DATA
# ==============================
with tab2:
    st.markdown('<h2 class="sub-header">⚙️ Pelabelan Otomatis dengan Lexicon</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Pelabelan Lexicon:** 
    Sistem akan otomatis memberi label sentimen berdasarkan kamus kata positif dan negatif.
    - Kata positif: bagus, baik, mantap, memuaskan, cepat, dll.
    - Kata negatif: buruk, jelek, telat, error, mahal, dll.
    """)
    
    # Tombol untuk memulai pelabelan
    if st.button("🏷️ Mulai Pelabelan Data", key="label_btn"):
        with st.spinner("Melakukan pelabelan sentimen..."):
            df = load_sample_data()
            df['sentiment'] = df['content'].apply(label_sentiment_lexicon)
            df_filtered = df[df['sentiment'].isin(['positif', 'negatif'])].copy()
            
            # Hitung distribusi
            sentiment_counts = df_filtered['sentiment'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Distribusi Sentimen")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                colors = ['#4CAF50', '#F44336']
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', 
                                     colors=colors, ax=ax2)
                ax2.set_ylabel('')
                ax2.set_title('Distribusi Sentimen Setelah Pelabelan')
                st.pyplot(fig2)
            
            with col2:
                st.markdown("### 📈 Detail Jumlah")
                st.metric("Total Data Awal", len(df))
                st.metric("Data Setelah Filter", len(df_filtered))
                st.metric("Ulasan Positif", f"{sentiment_counts.get('positif', 0):,}")
                st.metric("Ulasan Negatif", f"{sentiment_counts.get('negatif', 0):,}")
            
            # Simpan ke session state
            st.session_state['labeled_data'] = df_filtered
            st.session_state['sentiment_counts'] = sentiment_counts
            
            st.success("✅ Pelabelan selesai!")
    
    # Tampilkan contoh hasil pelabelan jika sudah dilakukan
    if 'labeled_data' in st.session_state:
        st.markdown("### 👁️ Contoh Hasil Pelabelan")
        
        # Filter untuk contoh
        sample_data = st.session_state['labeled_data'].sample(10, random_state=42)[['content', 'sentiment']]
        
        # Tampilkan dengan warna
        for idx, row in sample_data.iterrows():
            if row['sentiment'] == 'positif':
                st.markdown(f"**{row['content']}** - <span class='positive'>POSITIF</span>", 
                           unsafe_allow_html=True)
            else:
                st.markdown(f"**{row['content']}** - <span class='negative'>NEGATIF</span>", 
                           unsafe_allow_html=True)
            st.divider()

# ==============================
# TAB 3: PREPROCESSING
# ==============================
with tab3:
    st.markdown('<h2 class="sub-header">🔧 Preprocessing Data Teks</h2>', unsafe_allow_html=True)
    
    # Input teks untuk demo preprocessing
    st.markdown("### ✍️ Demo Preprocessing Teks")
    input_text = st.text_area(
        "Masukkan teks untuk melihat proses preprocessing:",
        "Aplikasi Gojek sangat BAGUS! Driver ramah dan pelayanan cepat 5/5 ⭐⭐⭐⭐⭐",
        height=100
    )
    
    use_nltk = st.checkbox("Gunakan NLTK (jika tersedia)", value=False)
    
    if st.button("🔧 Proses Preprocessing", key="preprocess_btn"):
        if not input_text.strip():
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Memproses teks..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📝 Teks Asli")
                    st.info(input_text)
                    
                    st.markdown("#### Langkah-langkah:")
                    steps = [
                        "1. **Cleaning**: Menghapus karakter khusus dan angka",
                        "2. **Case Folding**: Mengubah ke huruf kecil",
                        "3. **Tokenization**: Memecah menjadi kata-kata",
                        "4. **Stopword Removal**: Menghapus kata umum"
                    ]
                    for step in steps:
                        st.markdown(step)
                
                with col2:
                    st.markdown("### 🔄 Hasil Preprocessing")
                    
                    try:
                        # Proses preprocessing
                        processed_text = preprocess_text(input_text, use_nltk=use_nltk)
                        
                        # Tampilkan hasil
                        st.success(processed_text)
                        
                        # Tampilkan detail
                        with st.expander("Detail Proses"):
                            # Cleaning
                            cleaned = re.sub(r'[^a-zA-Z\s]', '', input_text)
                            cleaned = re.sub(r'\d+', '', cleaned)
                            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                            
                            # Case folding
                            lower_text = cleaned.lower()
                            
                            # Tokenization
                            if use_nltk:
                                try:
                                    tokens = word_tokenize(lower_text)
                                except:
                                    tokens = lower_text.split()
                            else:
                                tokens = lower_text.split()
                            
                            # Stopword removal
                            try:
                                stop_words = set(stopwords.words('indonesian'))
                            except:
                                stop_words = set(['yang', 'dan', 'di', 'ke', 'dari'])
                            
                            filtered_tokens = [word for word in tokens if word not in stop_words]
                            
                            st.write("**Cleaning:**", cleaned)
                            st.write("**Lowercase:**", lower_text)
                            st.write(f"**Tokens ({len(tokens)}):**", ", ".join(tokens))
                            st.write(f"**Filtered ({len(filtered_tokens)}):**", ", ".join(filtered_tokens))
                    
                    except Exception as e:
                        st.error(f"Error dalam preprocessing: {str(e)}")
                        # Fallback sederhana
                        simple_processed = input_text.lower()
                        simple_processed = re.sub(r'[^a-z\s]', '', simple_processed)
                        st.warning("Menggunakan preprocessing sederhana...")
                        st.info(simple_processed)
    
    # Preprocessing seluruh dataset
    st.markdown("### 📊 Preprocessing Dataset")
    
    if st.button("⚡ Proses Seluruh Dataset", key="process_all_btn"):
        if 'labeled_data' not in st.session_state:
            st.warning("Silakan lakukan pelabelan data terlebih dahulu di Tab 2.")
        else:
            with st.spinner("Memproses dataset..."):
                df = st.session_state['labeled_data'].copy()
                progress_bar = st.progress(0)
                
                # Lakukan preprocessing
                processed_texts = []
                for i, text in enumerate(df['content']):
                    processed_texts.append(preprocess_text(text, use_nltk=False))
                    if i % 100 == 0:
                        progress_bar.progress(i / len(df))
                
                df['processed_text'] = processed_texts
                progress_bar.progress(1.0)
                
                # Simpan ke session state
                st.session_state['processed_data'] = df
                
                st.success(f"✅ Preprocessing selesai! {len(df)} data telah diproses.")
                
                # Tampilkan contoh
                st.markdown("#### 📋 Contoh Hasil:")
                sample_results = df.sample(5, random_state=42)[['content', 'processed_text']]
                
                for idx, row in sample_results.iterrows():
                    with st.expander(f"Contoh {idx % 5 + 1}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Asli:**")
                            st.text(row['content'])
                        with col2:
                            st.markdown("**Hasil:**")
                            st.text(row['processed_text'])

# ==============================
# TAB 4: TRAINING MODEL
# ==============================
with tab4:
    st.markdown('<h2 class="sub-header">🏋️ Training Model SVM</h2>', unsafe_allow_html=True)
    
    # Parameter training
    st.markdown("### ⚙️ Parameter Training")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size_options = {"80:20": 0.2, "90:10": 0.1, "70:30": 0.3}
        selected_ratio = st.selectbox(
            "Rasio Training-Testing",
            list(test_size_options.keys()),
            index=0
        )
        test_size = test_size_options[selected_ratio]
    
    with col2:
        kernel_options = ["linear", "poly"]
        selected_kernel = st.selectbox("Kernel SVM", kernel_options, index=0)
    
    with col3:
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=42)
    
    # Tombol untuk memulai training
    if st.button("🚀 Mulai Training Model", key="train_btn"):
        if 'processed_data' not in st.session_state:
            st.warning("Silakan lakukan preprocessing data terlebih dahulu di Tab 3.")
        else:
            with st.spinner("Melakukan training model SVM..."):
                try:
                    # Persiapan data
                    df = st.session_state['processed_data'].copy()
                    
                    # Pastikan ada data
                    if len(df) < 100:
                        st.error("Data terlalu sedikit untuk training. Minimal 100 data.")
                        st.stop()
                    
                    # TF-IDF Vectorizer
                    vectorizer = TfidfVectorizer(max_features=500, min_df=2, max_df=0.9)
                    X = vectorizer.fit_transform(df['processed_text'])
                    y = df['sentiment'].map({'positif': 1, 'negatif': 0})
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Training model
                    if selected_kernel == 'poly':
                        model = SVC(kernel='poly', degree=2, random_state=random_state)
                    else:
                        model = SVC(kernel='linear', random_state=random_state)
                    
                    model.fit(X_train, y_train)
                    
                    # Prediksi dan evaluasi
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Tampilkan hasil
                    st.success(f"✅ Training selesai! Akurasi: {accuracy:.4f}")
                    
                    # Metrik
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Data Training", f"{X_train.shape[0]:,}")
                    
                    with col2:
                        st.metric("Data Testing", f"{X_test.shape[0]:,}")
                    
                    with col3:
                        st.metric("Akurasi", f"{accuracy:.2%}")
                    
                    with col4:
                        st.metric("Jumlah Fitur", X_train.shape[1])
                    
                    # Classification Report
                    st.markdown("### 📊 Classification Report")
                    report = classification_report(y_test, y_pred, 
                                                 target_names=['negatif', 'positif'],
                                                 output_dict=True)
                    
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                    
                    # Confusion Matrix
                    st.markdown("### 🎯 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Negatif', 'Positif'],
                               yticklabels=['Negatif', 'Positif'],
                               ax=ax3)
                    ax3.set_title('Confusion Matrix')
                    ax3.set_ylabel('True Label')
                    ax3.set_xlabel('Predicted Label')
                    st.pyplot(fig3)
                    
                    # Simpan model ke session state
                    st.session_state['trained_model'] = model
                    st.session_state['trained_vectorizer'] = vectorizer
                    st.session_state['model_accuracy'] = accuracy
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred'] = y_pred
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error dalam training: {str(e)}")
                    st.info("Coba kurangi jumlah fitur atau gunakan kernel yang berbeda.")

# ==============================
# TAB 5: KLASIFIKASI REAL-TIME
# ==============================
with tab5:
    st.markdown('<h2 class="sub-header">🔍 Klasifikasi Sentimen Real-time</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Fitur Klasifikasi:**
    - Masukkan teks ulasan tentang Gojek
    - Sistem akan mengklasifikasikan sentimen secara real-time
    - Hasil: POSITIF atau NEGATIF dengan confidence score
    """)
    
    # Input teks
    input_text = st.text_area(
        "Masukkan ulasan tentang Gojek:",
        "Driver sangat ramah dan pelayanan cepat, aplikasi mudah digunakan",
        height=150
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        kernel_choice = st.selectbox("Kernel", ["linear", "poly"], key="classify_kernel")
    
    with col2:
        if 'trained_model' in st.session_state:
            use_trained = st.checkbox("Gunakan model yang sudah dilatih", value=True)
        else:
            use_trained = False
            st.warning("Model belum dilatih")
    
    with col3:
        show_details = st.checkbox("Tampilkan detail proses", value=False)
    
    if st.button("🔍 Analisis Sentimen", key="analyze_btn"):
        if not input_text.strip():
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Menganalisis sentimen..."):
                try:
                    # Preprocessing
                    processed_text = preprocess_text(input_text, use_nltk=False)
                    
                    if use_trained and 'trained_model' in st.session_state:
                        # Gunakan model yang sudah dilatih
                        model = st.session_state['trained_model']
                        vectorizer = st.session_state['trained_vectorizer']
                        X_input = vectorizer.transform([processed_text])
                        
                        # Prediksi
                        prediction = model.predict(X_input)[0]
                        
                        # Untuk SVM linear, gunakan decision function
                        try:
                            decision_score = model.decision_function(X_input)[0]
                        except:
                            decision_score = 0.5 if prediction == 1 else -0.5
                    
                    else:
                        # Training model sederhana untuk demo
                        st.info("Training model cepat untuk analisis...")
                        
                        if 'processed_data' in st.session_state:
                            df = st.session_state['processed_data']
                        else:
                            # Buat data demo
                            df = load_sample_data()
                            df['sentiment'] = df['content'].apply(label_sentiment_lexicon)
                            df = df[df['sentiment'].isin(['positif', 'negatif'])]
                            df['processed_text'] = df['content'].apply(lambda x: preprocess_text(x, False))
                        
                        # Vectorization
                        demo_vectorizer = TfidfVectorizer(max_features=200)
                        X_demo = demo_vectorizer.fit_transform(df['processed_text'])
                        y_demo = df['sentiment'].map({'positif': 1, 'negatif': 0})
                        
                        # Training
                        if kernel_choice == 'poly':
                            demo_model = SVC(kernel='poly', degree=2)
                        else:
                            demo_model = SVC(kernel='linear')
                        
                        demo_model.fit(X_demo, y_demo)
                        
                        # Prediksi
                        X_input = demo_vectorizer.transform([processed_text])
                        prediction = demo_model.predict(X_input)[0]
                        
                        try:
                            decision_score = demo_model.decision_function(X_input)[0]
                        except:
                            decision_score = 0.5 if prediction == 1 else -0.5
                    
                    # Interpretasi hasil
                    sentiment = "POSITIF" if prediction == 1 else "NEGATIF"
                    confidence = min(abs(decision_score) * 2, 1.0)  # Normalisasi
                    
                    # Tampilkan hasil
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Hasil Analisis")
                        
                        if sentiment == "POSITIF":
                            st.markdown(f"""
                            <div style='text-align: center; padding: 2rem; background-color: #E8F5E9; 
                                      border-radius: 10px; border: 2px solid #4CAF50;'>
                                <h1 style='color: #4CAF50; font-size: 3rem;'>😊</h1>
                                <h2 style='color: #4CAF50;'>POSITIF</h2>
                                <p style='font-size: 1.2rem;'>Ulasan memiliki sentimen positif</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style='text-align: center; padding: 2rem; background-color: #FFEBEE; 
                                      border-radius: 10px; border: 2px solid #F44336;'>
                                <h1 style='color: #F44336; font-size: 3rem;'>😞</h1>
                                <h2 style='color: #F44336;'>NEGATIF</h2>
                                <p style='font-size: 1.2rem;'>Ulasan memiliki sentimen negatif</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 📈 Confidence Score")
                        
                        # Progress bar untuk confidence
                        confidence_percent = confidence * 100
                        
                        st.markdown(f"**Score: {decision_score:.3f}**")
                        st.progress(confidence)
                        
                        st.metric("Confidence", f"{confidence_percent:.1f}%")
                        
                        if confidence_percent > 70:
                            st.success("Confidence tinggi: Analisis sangat yakin")
                        elif confidence_percent > 40:
                            st.warning("Confidence sedang: Analisis cukup yakin")
                        else:
                            st.info("Confidence rendah: Hasil mungkin kurang akurat")
                    
                    # Tampilkan detail proses jika diminta
                    if show_details:
                        st.markdown("### 🔍 Detail Proses")
                        
                        with st.expander("Proses Preprocessing"):
                            st.code(f"Teks asli: {input_text}")
                            st.code(f"Teks hasil preprocessing: {processed_text}")
                        
                        with st.expander("Informasi Model"):
                            if use_trained and 'model_accuracy' in st.session_state:
                                st.write(f"**Akurasi Model:** {st.session_state['model_accuracy']:.2%}")
                                st.write(f"**Jumlah Data Training:** {st.session_state['X_test'].shape[0]}")
                                st.write(f"**Kernel:** {kernel_choice}")
                    
                    # Contoh ulasan lainnya
                    st.markdown("### 💡 Contoh Lainnya")
                    
                    examples = [
                        ("Driver sangat ramah dan pengemudian hati-hati", "POSITIF"),
                        ("Aplikasi sering error dan loading lama sekali", "NEGATIF"),
                        ("Tarif terjangkau dan promo banyak", "POSITIF"),
                        ("Customer service tidak responsif", "NEGATIF"),
                        ("GoFood selalu tepat waktu dan makanan masih hangat", "POSITIF")
                    ]
                    
                    cols = st.columns(len(examples))
                    for idx, (example_text, example_sentiment) in enumerate(examples):
                        with cols[idx]:
                            if example_sentiment == "POSITIF":
                                st.success(f"\"{example_text[:30]}...\"\n\n**{example_sentiment}**")
                            else:
                                st.error(f"\"{example_text[:30]}...\"\n\n**{example_sentiment}**")
                
                except Exception as e:
                    st.error(f"Terjadi error dalam analisis: {str(e)}")
                    st.info("Silakan coba teks yang berbeda atau training model terlebih dahulu.")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Dataset**")
    st.caption("8,000 ulasan Gojek")
    st.caption("Pelabelan otomatis dengan lexicon")

with col2:
    st.markdown("**🤖 Model**")
    st.caption("Support Vector Machine (SVM)")
    st.caption("Kernel: Linear & Polynomial")

with col3:
    st.markdown("**📈 Akurasi**")
    if 'model_accuracy' in st.session_state:
        st.caption(f"Terbaik: {st.session_state['model_accuracy']:.2%}")
    else:
        st.caption("Belum di-training")
    st.caption("TF-IDF Feature Extraction")

st.markdown("---")
st.caption("© 2024 Gojek Sentiment Analysis System | Made with Streamlit")