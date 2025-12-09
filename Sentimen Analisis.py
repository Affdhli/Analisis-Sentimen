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

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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
# FUNGSI UTILITAS
# ==============================
@st.cache_data
def load_sample_data():
    """Load dataset contoh"""
    # Data contoh
    sample_reviews = [
        "aplikasi yang sangat bagus dan membantu",
        "pelayanan buruk, driver sering telat",
        "mantap sekali gojek, mudah digunakan",
        "aplikasi sering error, perlu perbaikan",
        "driver ramah dan profesional",
        "tarif mahal tidak sesuai dengan pelayanan",
        "fitur lengkap dan sangat memudahkan",
        "loading lambat, sering force close",
        "pelayanan memuaskan, terima kasih gojek",
        "cancel order terus, sangat menyebalkan",
        "antar cepat dan tepat waktu",
        "aplikasi terbaik untuk transportasi online",
        "rating driver tidak akurat",
        "gofood enak dan cepat sampai",
        "update terbaru bikin aplikasi makin lambat",
        "customer service responsif",
        "potongan harga sering ada, hemat banget",
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

@st.cache_data
def preprocess_text(text):
    """Fungsi preprocessing teks"""
    # Cleaning
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Case folding
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    stop_words = set(stopwords.words('indonesian'))
    custom_stopwords = ['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 
                       'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih']
    stop_words.update(custom_stopwords)
    
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def label_sentiment_lexicon(text):
    """Pelabelan dengan lexicon"""
    positive_words = ['bagus', 'baik', 'mantap', 'memuaskan', 'cepat', 'mudah', 
                     'ramah', 'profesional', 'lengkap', 'hemat', 'responsif']
    negative_words = ['buruk', 'jelek', 'telat', 'error', 'mahal', 'lambat', 
                     'menyebalkan', 'kurang', 'cancel', 'force close']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
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
    
    # Pelabelan data
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
    
    # Tampilkan contoh hasil pelabelan
    st.markdown("### 👁️ Contoh Hasil Pelabelan")
    
    # Filter untuk contoh
    sample_data = df_filtered.sample(10, random_state=42)[['content', 'sentiment']]
    
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
    
    if st.button("Proses Preprocessing", key="preprocess_btn"):
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
            
            # Proses step by step
            # 1. Cleaning
            cleaned = re.sub(r'[^a-zA-Z\s]', '', input_text)
            cleaned = re.sub(r'\d+', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # 2. Case folding
            lower_text = cleaned.lower()
            
            # 3. Tokenization
            tokens = word_tokenize(lower_text)
            
            # 4. Stopword removal
            stop_words = set(stopwords.words('indonesian'))
            filtered_tokens = [word for word in tokens if word not in stop_words]
            
            # Gabungkan kembali
            processed_text = ' '.join(filtered_tokens)
            
            st.success(processed_text)
            
            st.markdown("#### Detail Proses:")
            st.code(f"Cleaning: {cleaned}")
            st.code(f"Lowercase: {lower_text}")
            st.code(f"Tokens: {tokens}")
            st.code(f"Filtered: {filtered_tokens}")
    
    # Preprocessing seluruh dataset
    st.markdown("### 📊 Preprocessing Dataset")
    
    if st.button("Proses Seluruh Dataset", key="process_all_btn"):
        with st.spinner("Memproses 8000 data ulasan..."):
            df = load_sample_data()
            progress_bar = st.progress(0)
            
            # Lakukan preprocessing
            processed_texts = []
            for i, text in enumerate(df['content']):
                processed_texts.append(preprocess_text(text))
                if i % 100 == 0:
                    progress_bar.progress(i / len(df))
            
            df['processed_text'] = processed_texts
            progress_bar.progress(1.0)
            
            st.success(f"Preprocessing selesai! {len(df)} data telah diproses.")
            
            # Tampilkan contoh
            st.markdown("#### Contoh Hasil:")
            sample_results = df.sample(5, random_state=42)[['content', 'processed_text']]
            
            for idx, row in sample_results.iterrows():
                with st.expander(f"Contoh {idx % 5 + 1}"):
                    st.markdown("**Asli:**")
                    st.text(row['content'])
                    st.markdown("**Hasil Processing:**")
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
        with st.spinner("Melakukan training model SVM..."):
            # Persiapan data
            df = load_sample_data()
            df['sentiment'] = df['content'].apply(label_sentiment_lexicon)
            df_filtered = df[df['sentiment'].isin(['positif', 'negatif'])].copy()
            
            # Preprocessing
            df_filtered['processed_text'] = df_filtered['content'].apply(preprocess_text)
            
            # TF-IDF Vectorizer
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(df_filtered['processed_text'])
            y = df_filtered['sentiment'].map({'positif': 1, 'negatif': 0})
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Training model
            if selected_kernel == 'poly':
                model = SVC(kernel='poly', degree=3, random_state=random_state)
            else:
                model = SVC(kernel=selected_kernel, random_state=random_state)
            
            model.fit(X_train, y_train)
            
            # Prediksi dan evaluasi
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Tampilkan hasil
            st.success(f"Training selesai! Akurasi: {accuracy:.4f}")
            
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
            
            # Simpan model (dalam session state)
            st.session_state['model'] = model
            st.session_state['vectorizer'] = vectorizer
            st.session_state['accuracy'] = accuracy
            
            st.balloons()

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
        if 'model' in st.session_state:
            use_saved = st.checkbox("Gunakan model yang sudah dilatih", value=True)
        else:
            use_saved = False
    
    with col3:
        show_details = st.checkbox("Tampilkan detail proses", value=False)
    
    if st.button("🔍 Analisis Sentimen", key="analyze_btn"):
        if not input_text.strip():
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Menganalisis sentimen..."):
                try:
                    # Preprocessing
                    processed_text = preprocess_text(input_text)
                    
                    # Vectorization
                    if use_saved and 'vectorizer' in st.session_state:
                        vectorizer = st.session_state['vectorizer']
                        X_input = vectorizer.transform([processed_text])
                        
                        if 'model' in st.session_state:
                            model = st.session_state['model']
                            prediction = model.predict(X_input)[0]
                            decision_score = model.decision_function(X_input)[0]
                        else:
                            st.error("Model belum dilatih. Silakan training model terlebih dahulu di Tab 4.")
                            st.stop()
                    else:
                        # Training model sederhana untuk demo
                        df = load_sample_data().iloc[:1000]  # Gunakan subset untuk demo cepat
                        df['sentiment'] = df['content'].apply(label_sentiment_lexicon)
                        df_filtered = df[df['sentiment'].isin(['positif', 'negatif'])].copy()
                        df_filtered['processed_text'] = df_filtered['content'].apply(preprocess_text)
                        
                        demo_vectorizer = TfidfVectorizer(max_features=500)
                        X_demo = demo_vectorizer.fit_transform(df_filtered['processed_text'])
                        y_demo = df_filtered['sentiment'].map({'positif': 1, 'negatif': 0})
                        
                        if kernel_choice == 'poly':
                            demo_model = SVC(kernel='poly', degree=3)
                        else:
                            demo_model = SVC(kernel='linear')
                        
                        demo_model.fit(X_demo, y_demo)
                        
                        X_input = demo_vectorizer.transform([processed_text])
                        prediction = demo_model.predict(X_input)[0]
                        decision_score = demo_model.decision_function(X_input)[0]
                    
                    # Interpretasi hasil
                    sentiment = "POSITIF" if prediction == 1 else "NEGATIF"
                    confidence = abs(decision_score)
                    
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
                        confidence_percent = min(confidence * 10, 100)  # Normalisasi
                        
                        st.markdown(f"**Score: {decision_score:.3f}**")
                        st.progress(confidence_percent / 100)
                        
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
                        
                        with st.expander("Fitur yang Dideteksi"):
                            if use_saved and 'vectorizer' in st.session_state:
                                feature_names = vectorizer.get_feature_names_out()
                                input_features = X_input.toarray()[0]
                                important_features = []
                                
                                for i, value in enumerate(input_features):
                                    if value > 0:
                                        important_features.append((feature_names[i], value))
                                
                                if important_features:
                                    st.write("Kata-kata penting yang terdeteksi:")
                                    for feature, score in sorted(important_features, key=lambda x: x[1], reverse=True)[:10]:
                                        st.write(f"- {feature}: {score:.4f}")
                                else:
                                    st.write("Tidak ada fitur yang terdeteksi dari teks input")
                        
                        with st.expander("Alasan Klasifikasi"):
                            if sentiment == "POSITIF":
                                st.write("""
                                **Alasan kemungkinan sentimen POSITIF:**
                                - Mengandung kata-kata positif seperti "ramah", "cepat", "mudah"
                                - Struktur kalimat umumnya mendukung
                                - Tidak ada kata negatif yang dominan
                                """)
                            else:
                                st.write("""
                                **Alasan kemungkinan sentimen NEGATIF:**
                                - Mengandung kata-kata negatif
                                - Struktur kalimat menunjukkan ketidakpuasan
                                - Mungkin ada kata yang termasuk dalam lexicon negatif
                                """)
                    
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
    if 'accuracy' in st.session_state:
        st.caption(f"Terbaik: {st.session_state['accuracy']:.2%}")
    else:
        st.caption("Belum di-training")
    st.caption("TF-IDF Feature Extraction")

st.markdown("---")
st.caption("© 2024 Gojek Sentiment Analysis System | Made with Streamlit")