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
import io
import base64
from datetime import datetime

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Analisis Sentimen Gojek",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00B14F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .positive {
        color: #00B14F;
    }
    .negative {
        color: #FF4B4B;
    }
    .stProgress > div > div > div > div {
        background-color: #00B14F;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🚗 Analisis Sentimen Ulasan Gojek</h1>", unsafe_allow_html=True)
st.markdown("**Aplikasi untuk menganalisis sentimen dari ulasan pengguna Gojek menggunakan SVM**")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Gojek.svg/2560px-Gojek.svg.png", width=150)
    st.title("⚙️ Pengaturan")
    
    # Data upload
    st.subheader("📁 Data Source")
    data_option = st.radio(
        "Pilih sumber data:",
        ["Gunakan Data Contoh", "Upload File CSV"],
        index=0
    )
    
    if data_option == "Upload File CSV":
        uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Data berhasil diupload! ({len(df)} baris)")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                df = None
        else:
            df = None
            st.info("📁 Silakan upload file CSV")
    else:
        df = None
        st.info("📊 Akan menggunakan data contoh")
    
    # Preprocessing settings
    st.subheader("🔧 Preprocessing")
    remove_stopwords = st.checkbox("Hapus Stopwords", value=True)
    max_features = st.slider("Max Features TF-IDF", 1000, 5000, 3000, 500)
    
    # Model settings
    st.subheader("🤖 Model Settings")
    test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.2, 0.05)
    kernel_type = st.selectbox("Kernel Type", ["linear", "poly", "rbf"])
    
    # Analysis settings
    st.subheader("📈 Analysis")
    show_wordcloud = st.checkbox("Tampilkan WordCloud", value=True)
    show_confusion = st.checkbox("Tampilkan Confusion Matrix", value=True)
    
    st.divider()
    if st.button("🔄 Reset All", type="secondary"):
        st.rerun()

# Lexicon untuk pelabelan
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

# Fungsi-fungsi helper
@st.cache_data
def load_sample_data(n_samples=8000):
    """Membuat data contoh"""
    np.random.seed(42)
    
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
    
    data_content = []
    data_sentiment = []
    
    for i in range(n_samples):
        if np.random.random() > 0.4:
            base_text = np.random.choice(positive_samples)
            sentiment = 'positif'
        else:
            base_text = np.random.choice(negative_samples)
            sentiment = 'negatif'
        
        variations = ["", "sangat", "sekali", "banget", "saya rasa", "menurut saya"]
        variation = np.random.choice(variations)
        
        if variation:
            content = f"{variation} {base_text}"
        else:
            content = base_text
            
        data_content.append(content)
        data_sentiment.append(sentiment)
    
    return pd.DataFrame({
        'content': data_content,
        'sentimen': data_sentiment
    })

def clean_text(text):
    """Fungsi untuk membersihkan teks"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lexicon_sentiment_analysis_binary(text):
    """Pelabelan sentimen dengan lexicon"""
    if not isinstance(text, str):
        return 'neutral'
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count == negative_count:
        strong_positive = any(word in text_lower for word in ['sangat baik', 'sangat bagus', 'luar biasa', 'terbaik'])
        strong_negative = any(word in text_lower for word in ['sangat buruk', 'sangat jelek', 'parah sekali', 'penipu'])
        
        if strong_positive:
            return 'positive'
        elif strong_negative:
            return 'negative'
        else:
            return 'positive'
    
    return 'positive' if positive_count > negative_count else 'negative'

def count_words(text):
    """Menghitung jumlah kata"""
    if not isinstance(text, str):
        return 0
    return len(text.split())

# Tab utama
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 Preprocessing", "🤖 Model Training", "🎯 Prediction"])

with tab1:
    st.markdown("<h2 class='sub-header'>📊 Data Overview</h2>", unsafe_allow_html=True)
    
    if df is None:
        with st.spinner("Membuat data contoh..."):
            df = load_sample_data(2000)  # Gunakan 2000 data untuk demo lebih cepat
        st.success(f"✅ Data contoh berhasil dibuat! ({len(df)} baris)")
    
    # Tampilkan informasi data
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data", f"{len(df):,}")
    with col2:
        total_words = df['content'].apply(count_words).sum()
        st.metric("Total Kata", f"{total_words:,}")
    with col3:
        avg_words = df['content'].apply(count_words).mean()
        st.metric("Rata-rata Kata", f"{avg_words:.1f}")
    with col4:
        if 'sentimen' in df.columns:
            pos_count = (df['sentimen'] == 'positif').sum()
            st.metric("Sentimen Positif", f"{pos_count:,}")
    
    # Tampilkan data
    st.subheader("Preview Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistik
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Sentimen Awal")
        if 'sentimen' in df.columns:
            sentiment_counts = df['sentimen'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#00B14F', '#FF4B4B', '#FFC107']
            ax.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                  autopct='%1.1f%%', colors=colors[:len(sentiment_counts)], startangle=90)
            ax.set_title('Distribusi Sentimen Awal')
            st.pyplot(fig)
    
    with col2:
        st.subheader("Distribusi Jumlah Kata")
        df['word_count'] = df['content'].apply(count_words)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['word_count'], bins=30, edgecolor='black', alpha=0.7, color='#00B14F')
        ax.axvline(df['word_count'].mean(), color='red', linestyle='dashed', linewidth=2, 
                  label=f'Rata-rata: {df["word_count"].mean():.1f}')
        ax.set_xlabel('Jumlah Kata')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Distribusi Jumlah Kata per Ulasan')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

with tab2:
    st.markdown("<h2 class='sub-header'>🔍 Text Preprocessing</h2>", unsafe_allow_html=True)
    
    if 'df' not in locals():
        st.warning("Silakan muat data terlebih dahulu di tab Data Overview")
        st.stop()
    
    # Proses pelabelan
    with st.spinner("Melakukan pelabelan sentimen..."):
        progress_bar = st.progress(0)
        
        # Langkah 1: Pelabelan
        df['sentiment_label'] = df['content'].apply(lexicon_sentiment_analysis_binary)
        progress_bar.progress(25)
        
        # Langkah 2: Filter hanya positif dan negatif
        df = df[df['sentiment_label'].isin(['positive', 'negative'])].copy()
        progress_bar.progress(50)
        
        # Langkah 3: Cleaning
        df['cleaned_text'] = df['content'].apply(clean_text)
        progress_bar.progress(75)
        
        # Langkah 4: Tokenization
        df['tokens'] = df['cleaned_text'].apply(word_tokenize)
        df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
        df['word_count_processed'] = df['processed_text'].apply(count_words)
        
        progress_bar.progress(100)
    
    st.success("✅ Preprocessing selesai!")
    
    # Tampilkan hasil
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data setelah preprocessing", f"{len(df):,}")
    
    with col2:
        pos_count = (df['sentiment_label'] == 'positive').sum()
        st.metric("Positif", f"{pos_count:,}")
    
    with col3:
        neg_count = (df['sentiment_label'] == 'negative').sum()
        st.metric("Negatif", f"{neg_count:,}")
    
    # WordCloud
    if show_wordcloud:
        st.subheader("WordCloud Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### WordCloud Positif")
            positive_text = ' '.join(df[df['sentiment_label'] == 'positive']['processed_text'].tolist())
            if positive_text:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                    max_words=100, colormap='viridis').generate(positive_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        with col2:
            st.markdown("#### WordCloud Negatif")
            negative_text = ' '.join(df[df['sentiment_label'] == 'negative']['processed_text'].tolist())
            if negative_text:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                    max_words=100, colormap='Reds').generate(negative_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
    
    # Contoh hasil preprocessing
    st.subheader("Contoh Hasil Preprocessing")
    sample_idx = st.slider("Pilih contoh", 0, min(10, len(df)-1), 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Text:**")
        st.text_area("", df['content'].iloc[sample_idx], height=150, disabled=True)
        st.metric("Jumlah Kata", df['word_count'].iloc[sample_idx])
    
    with col2:
        st.markdown("**Processed Text:**")
        st.text_area("", df['processed_text'].iloc[sample_idx], height=150, disabled=True)
        st.metric("Jumlah Kata setelah preprocessing", df['word_count_processed'].iloc[sample_idx])

with tab3:
    st.markdown("<h2 class='sub-header'>🤖 Model Training & Evaluation</h2>", unsafe_allow_html=True)
    
    if 'processed_text' not in df.columns:
        st.warning("Silakan lakukan preprocessing terlebih dahulu di tab Preprocessing")
        st.stop()
    
    # TF-IDF Vectorization
    with st.spinner("Melakukan ekstraksi fitur TF-IDF..."):
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=3,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        X = tfidf_vectorizer.fit_transform(df['processed_text'])
        y = df['sentiment_label'].map({'positive': 1, 'negative': 0})
        
        st.success(f"✅ TF-IDF selesai! Dimensi: {X.shape}")
    
    # Split data
    with st.spinner("Membagi data training dan testing..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Data", f"{X_train.shape[0]:,}")
            st.metric("Positive (train)", f"{sum(y_train == 1):,}")
            st.metric("Negative (train)", f"{sum(y_train == 0):,}")
        
        with col2:
            st.metric("Testing Data", f"{X_test.shape[0]:,}")
            st.metric("Positive (test)", f"{sum(y_test == 1):,}")
            st.metric("Negative (test)", f"{sum(y_test == 0):,}")
    
    # Training model
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner(f"Melatih model SVM dengan kernel {kernel_type}..."):
            # Create and train model
            svm_model = SVC(
                kernel=kernel_type,
                random_state=42,
                C=1.0,
                probability=True if kernel_type in ['poly', 'rbf'] else False
            )
            
            svm_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = svm_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['negative', 'positive'], output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Display results
            st.success(f"✅ Training selesai! Akurasi: {accuracy:.4f}")
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Precision", f"{report['positive']['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{report['positive']['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{report['positive']['f1-score']:.4f}")
            
            # Confusion Matrix
            if show_confusion:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Negative', 'Positive'],
                           yticklabels=['Negative', 'Positive'],
                           ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            
            # Classification Report
            st.subheader("Detailed Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Save model to session state
            st.session_state['model'] = svm_model
            st.session_state['vectorizer'] = tfidf_vectorizer
            st.session_state['accuracy'] = accuracy
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
    
    # Jika model sudah ada di session state
    if 'model' in st.session_state:
        st.divider()
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Kernel:** {kernel_type}")
            st.info(f"**Accuracy:** {st.session_state['accuracy']:.4f}")
        
        with col2:
            # Download model
            model_bytes = pickle.dumps({
                'model': st.session_state['model'],
                'vectorizer': st.session_state['vectorizer'],
                'accuracy': st.session_state['accuracy']
            })
            
            st.download_button(
                label="📥 Download Model",
                data=model_bytes,
                file_name=f"svm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream"
            )

with tab4:
    st.markdown("<h2 class='sub-header'>🎯 Real-time Prediction</h2>", unsafe_allow_html=True)
    
    # Input text area
    st.subheader("Masukkan teks untuk dianalisis:")
    user_input = st.text_area(
        "Tulis ulasan tentang Gojek di sini:",
        height=150,
        placeholder="Contoh: 'Aplikasi Gojek sangat bagus dan mudah digunakan, driver ramah dan cepat sampai tujuan...'"
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        analyze_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)
    
    if analyze_btn and user_input:
        if 'model' not in st.session_state:
            st.warning("⚠️ Silakan train model terlebih dahulu di tab Model Training")
        else:
            # Preprocess input
            cleaned_text = clean_text(user_input)
            tokens = word_tokenize(cleaned_text)
            processed_text = ' '.join(tokens)
            
            # Transform with TF-IDF
            text_vectorized = st.session_state['vectorizer'].transform([processed_text])
            
            # Predict
            prediction = st.session_state['model'].predict(text_vectorized)[0]
            sentiment = 'POSITIF 🟢' if prediction == 1 else 'NEGATIF 🔴'
            
            # Get probability if available
            try:
                probabilities = st.session_state['model'].predict_proba(text_vectorized)[0]
                confidence = probabilities[prediction]
            except:
                confidence = None
            
            # Display results
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Hasil Analisis")
                if sentiment == 'POSITIF 🟢':
                    st.success(f"## {sentiment}")
                else:
                    st.error(f"## {sentiment}")
                
                if confidence:
                    st.metric("Confidence", f"{confidence:.2%}")
            
            with col2:
                st.markdown("### Text Processed")
                st.info(processed_text)
                st.metric("Jumlah Kata", len(tokens))
            
            with col3:
                st.markdown("### Kata Kunci Terdeteksi")
                text_lower = user_input.lower()
                pos_found = [word for word in positive_words if word in text_lower]
                neg_found = [word for word in negative_words if word in text_lower]
                
                if pos_found:
                    st.markdown("**Kata Positif:**")
                    for word in pos_found[:5]:
                        st.markdown(f"- <span class='positive'>{word}</span>", unsafe_allow_html=True)
                
                if neg_found:
                    st.markdown("**Kata Negatif:**")
                    for word in neg_found[:5]:
                        st.markdown(f"- <span class='negative'>{word}</span>", unsafe_allow_html=True)
    
    # Batch prediction
    st.divider()
    st.subheader("Batch Prediction")
    
    batch_input = st.text_area(
        "Masukkan beberapa ulasan (satu per baris):",
        height=100,
        placeholder="Contoh:\nAplikasi bagus\nPelayanan buruk\nDriver ramah"
    )
    
    if st.button("Analisis Batch") and batch_input:
        if 'model' not in st.session_state:
            st.warning("⚠️ Silakan train model terlebih dahulu")
        else:
            lines = batch_input.strip().split('\n')
            results = []
            
            for line in lines:
                if line.strip():
                    cleaned = clean_text(line.strip())
                    tokens = word_tokenize(cleaned)
                    processed = ' '.join(tokens)
                    vectorized = st.session_state['vectorizer'].transform([processed])
                    pred = st.session_state['model'].predict(vectorized)[0]
                    sentiment = 'POSITIF' if pred == 1 else 'NEGATIF'
                    results.append({
                        'Text': line.strip(),
                        'Sentiment': sentiment,
                        'Processed': processed
                    })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Analisis Sentimen Gojek menggunakan SVM | Made with Streamlit</p>
    <p>© 2024 Sentiment Analysis App</p>
</div>
""", unsafe_allow_html=True)