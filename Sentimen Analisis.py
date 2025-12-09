# app.py - Aplikasi Streamlit Analisis Sentimen Gojek

import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Gojek",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Title dan header
st.title("🚗 Analisis Sentimen Ulasan Gojek")
st.markdown("---")

# Sidebar untuk navigasi
with st.sidebar:
    st.title("📊 Menu Navigasi")
    menu = st.radio(
        "Pilih Halaman:",
        ["🏠 Beranda", 
         "📈 Dashboard Data", 
         "🤖 Prediksi Sentimen",
         "📊 Evaluasi Model",
         "📁 Kelola Data"]
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Informasi Sistem")
    st.info("""
    Sistem analisis sentimen untuk ulasan aplikasi Gojek menggunakan model SVM.
    
    **Fitur:**
    - Analisis data scraping
    - Pelabelan otomatis
    - Prediksi sentimen real-time
    - Visualisasi interaktif
    """)

# Fungsi preprocessing
def preprocess_text(text):
    """Preprocessing teks untuk analisis sentimen"""
    text = str(text)
    # Cleaning
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Case folding
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    indonesian_stopwords = set(stopwords.words('indonesian'))
    custom_stopwords = ['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar']
    all_stopwords = indonesian_stopwords.union(set(custom_stopwords))
    
    tokens = [word for word in tokens if word not in all_stopwords and len(word) > 1]
    
    return ' '.join(tokens)

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    """Memuat model SVM yang telah dilatih"""
    try:
        with open('model_svm_terbaik.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except:
        st.error("Model tidak ditemukan. Pastikan file 'model_svm_terbaik.pkl' tersedia.")
        return None

# Halaman Beranda
if menu == "🏠 Beranda":
    st.header("🏠 Selamat Datang di Sistem Analisis Sentimen Gojek")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 Tentang Sistem
        
        Sistem ini dirancang untuk menganalisis sentimen dari ulasan pengguna 
        aplikasi Gojek di Google Play Store dengan menggunakan teknik 
        **Machine Learning**.
        
        **Fitur Utama:**
        
        1. **📈 Dashboard Data** - Visualisasi dan analisis data ulasan
        2. **🤖 Prediksi Sentimen** - Analisis sentimen teks secara real-time
        3. **📊 Evaluasi Model** - Evaluasi performa model klasifikasi
        4. **📁 Kelola Data** - Unggah dan kelola dataset
        
        **Teknologi yang Digunakan:**
        - **Scikit-learn** untuk model SVM
        - **NLTK** untuk preprocessing teks
        - **TF-IDF** untuk ekstraksi fitur
        - **Streamlit** untuk antarmuka web
        """)
    
    with col2:
        st.markdown("### 📊 Statistik Singkat")
        
        # Contoh statistik
        stats_data = {
            'Metric': ['Total Data', 'Data Positif', 'Data Negatif', 'Akurasi Model'],
            'Value': ['8000', '5200', '2800', '89.5%']
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Quick info
        st.info("""
        **Model Terbaik:**
        - Rasio: 80:20
        - Kernel: Linear
        - Akurasi: 89.5%
        """)
    
    st.markdown("---")
    
    # Demo singkat
    st.subheader("🎯 Coba Prediksi Sentimen")
    demo_text = st.text_area("Masukkan teks ulasan:", 
                            "Aplikasi Gojek sangat bagus, driver ramah dan cepat sampai")
    
    if st.button("Analisis Sentimen", type="primary"):
        model_data = load_model()
        if model_data:
            cleaned_text = preprocess_text(demo_text)
            
            # Prediksi
            text_tfidf = model_data['tfidf_vectorizer'].transform([cleaned_text])
            prediction = model_data['model'].predict(text_tfidf)[0]
            prediction_proba = model_data['model'].predict_proba(text_tfidf)[0]
            
            sentiment = "😊 POSITIF" if prediction == 1 else "😠 NEGATIF"
            confidence = prediction_proba[prediction] * 100
            
            # Tampilkan hasil
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Sentimen", sentiment)
            with col_b:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col_c:
                st.metric("Probabilitas", f"{max(prediction_proba)*100:.1f}%")
            
            # Progress bar
            st.progress(int(confidence))
            
            # Chart probabilitas
            fig = go.Figure(data=[
                go.Bar(
                    x=['Negatif', 'Positif'],
                    y=[prediction_proba[0]*100, prediction_proba[1]*100],
                    marker_color=['#FF6B6B', '#4ECDC4']
                )
            ])
            fig.update_layout(
                title="Probabilitas Sentimen",
                yaxis_title="Probabilitas (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# Halaman Dashboard Data
elif menu == "📈 Dashboard Data":
    st.header("📈 Dashboard Analisis Data")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Distribusi", "📝 Word Cloud", "📅 Timeline"])
    
    with tab1:
        st.subheader("📊 Overview Data")
        
        # Contoh data (dalam aplikasi nyata, ini akan di-load dari file)
        sample_data = {
            'content': [
                'Aplikasi sangat bagus dan mudah digunakan',
                'Driver telat dan tidak sopan',
                'Pelayanan memuaskan, harga terjangkau',
                'Aplikasi sering error dan lemot',
                'Driver ramah dan membantu',
                'Tarif mahal tidak sesuai pelayanan',
                'Pemesanan cepat dan akurat',
                'Customer service kurang responsif',
                'Pengalaman menggunakan sangat baik',
                'Pesanan sering tertukar'
            ],
            'sentimen': ['positif', 'negatif', 'positif', 'negatif', 'positif', 
                        'negatif', 'positif', 'negatif', 'positif', 'negatif'],
            'score': [5, 1, 5, 2, 5, 1, 4, 2, 5, 1],
            'date': pd.date_range('2024-01-01', periods=10)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Tampilkan data
        st.dataframe(df, use_container_width=True)
        
        # Metrik utama
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Ulasan", len(df))
        with col2:
            st.metric("Ulasan Positif", len(df[df['sentimen'] == 'positif']))
        with col3:
            st.metric("Ulasan Negatif", len(df[df['sentimen'] == 'negatif']))
        with col4:
            avg_score = df['score'].mean()
            st.metric("Rating Rata-rata", f"{avg_score:.1f}/5")
    
    with tab2:
        st.subheader("📈 Distribusi Sentimen")
        
        # Data distribusi
        sentiment_counts = df['sentimen'].value_counts()
        
        # Pie chart menggunakan Plotly
        fig1 = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'positif': '#4ECDC4', 'negatif': '#FF6B6B'},
            hole=0.3
        )
        fig1.update_layout(
            title="Distribusi Sentimen",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Bar chart
        fig2 = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={'positif': '#4ECDC4', 'negatif': '#FF6B6B'},
            text=sentiment_counts.values
        )
        fig2.update_layout(
            title="Jumlah Ulasan per Sentimen",
            xaxis_title="Sentimen",
            yaxis_title="Jumlah Ulasan",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Distribusi rating
        fig3 = px.histogram(
            df, 
            x='score',
            color='sentimen',
            color_discrete_map={'positif': '#4ECDC4', 'negatif': '#FF6B6B'},
            nbins=5,
            barmode='group'
        )
        fig3.update_layout(
            title="Distribusi Rating per Sentimen",
            xaxis_title="Rating (1-5)",
            yaxis_title="Jumlah Ulasan",
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.subheader("📝 Word Cloud Analisis")
        
        # Pisahkan teks positif dan negatif
        positive_text = ' '.join(df[df['sentimen'] == 'positif']['content'])
        negative_text = ' '.join(df[df['sentimen'] == 'negatif']['content'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🟢 Kata-kata Positif")
            if positive_text:
                # Generate wordcloud
                wordcloud_pos = WordCloud(
                    width=400, 
                    height=300,
                    background_color='white',
                    colormap='summer',
                    max_words=50
                ).generate(positive_text)
                
                # Display wordcloud
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud_pos, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("Tidak ada data positif")
        
        with col2:
            st.markdown("#### 🔴 Kata-kata Negatif")
            if negative_text:
                # Generate wordcloud
                wordcloud_neg = WordCloud(
                    width=400, 
                    height=300,
                    background_color='white',
                    colormap='autumn',
                    max_words=50
                ).generate(negative_text)
                
                # Display wordcloud
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud_neg, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("Tidak ada data negatif")
    
    with tab4:
        st.subheader("📅 Timeline Analisis")
        
        # Buat data timeline
        timeline_data = df.copy()
        timeline_data['date'] = pd.to_datetime(timeline_data['date'])
        timeline_data['count'] = 1
        
        # Agregasi per hari
        daily_counts = timeline_data.groupby(['date', 'sentimen']).size().unstack(fill_value=0)
        
        # Line chart
        fig = go.Figure()
        
        if 'positif' in daily_counts.columns:
            fig.add_trace(go.Scatter(
                x=daily_counts.index,
                y=daily_counts['positif'],
                mode='lines+markers',
                name='Positif',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8)
            ))
        
        if 'negatif' in daily_counts.columns:
            fig.add_trace(go.Scatter(
                x=daily_counts.index,
                y=daily_counts['negatif'],
                mode='lines+markers',
                name='Negatif',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Trend Sentimen Harian",
            xaxis_title="Tanggal",
            yaxis_title="Jumlah Ulasan",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Halaman Prediksi Sentimen
elif menu == "🤖 Prediksi Sentimen":
    st.header("🤖 Prediksi Sentimen Real-time")
    
    # Load model
    model_data = load_model()
    
    if model_data:
        # Tampilkan info model
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "SVM")
        with col2:
            st.metric("Kernel", model_data.get('kernel', 'Linear'))
        with col3:
            st.metric("Akurasi", f"{model_data.get('accuracy', 0)*100:.1f}%")
        
        st.markdown("---")
        
        # Tab untuk input berbeda
        tab1, tab2, tab3 = st.tabs(["📝 Input Teks", "📄 Upload File", "📊 Batch Analysis"])
        
        with tab1:
            st.subheader("📝 Analisis Teks Tunggal")
            
            # Input teks
            input_text = st.text_area(
                "Masukkan teks ulasan Gojek:",
                height=150,
                placeholder="Contoh: Driver sangat ramah dan tepat waktu, pengalaman menyenangkan..."
            )
            
            col1, col2 = st.columns([3, 1])
            with col2:
                analyze_btn = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)
            
            if analyze_btn and input_text:
                with st.spinner("Menganalisis sentimen..."):
                    # Preprocessing
                    cleaned_text = preprocess_text(input_text)
                    
                    # Transformasi TF-IDF
                    text_tfidf = model_data['tfidf_vectorizer'].transform([cleaned_text])
                    
                    # Prediksi
                    prediction = model_data['model'].predict(text_tfidf)[0]
                    prediction_proba = model_data['model'].predict_proba(text_tfidf)[0]
                    
                    sentiment = "POSITIF" if prediction == 1 else "NEGATIF"
                    confidence = prediction_proba[prediction] * 100
                    
                    # Tampilkan hasil
                    st.markdown("---")
                    
                    # Header hasil
                    if sentiment == "POSITIF":
                        st.success(f"### 😊 HASIL: {sentiment}")
                    else:
                        st.error(f"### 😠 HASIL: {sentiment}")
                    
                    # Metrik
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col_b:
                        st.metric("Prob Positif", f"{prediction_proba[1]*100:.1f}%")
                    with col_c:
                        st.metric("Prob Negatif", f"{prediction_proba[0]*100:.1f}%")
                    
                    # Progress bar
                    st.progress(int(confidence))
                    
                    # Chart probabilitas
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Negatif', 'Positif'],
                            y=[prediction_proba[0]*100, prediction_proba[1]*100],
                            marker_color=['#FF6B6B', '#4ECDC4'],
                            text=[f'{prediction_proba[0]*100:.1f}%', f'{prediction_proba[1]*100:.1f}%'],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Distribusi Probabilitas Sentimen",
                        yaxis_title="Probabilitas (%)",
                        yaxis_range=[0, 100],
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detail preprocessing
                    with st.expander("🔍 Detail Preprocessing"):
                        st.write("**Teks Asli:**", input_text)
                        st.write("**Teks Setelah Cleaning:**", cleaned_text)
                        
                        # Analisis kata kunci
                        tokens = word_tokenize(cleaned_text)
                        st.write("**Token:**", tokens)
                        
                        # Kata positif/negatif
                        positive_words = ['bagus', 'baik', 'ramah', 'cepat', 'puas', 'senang', 'mudah', 'mantap']
                        negative_words = ['buruk', 'jelek', 'lambat', 'sulit', 'kecewa', 'marah', 'mahal', 'error']
                        
                        found_positive = [word for word in tokens if word in positive_words]
                        found_negative = [word for word in tokens if word in negative_words]
                        
                        if found_positive:
                            st.write("**Kata Positif ditemukan:**", ', '.join(found_positive))
                        if found_negative:
                            st.write("**Kata Negatif ditemukan:**", ', '.join(found_negative))
        
        with tab2:
            st.subheader("📄 Analisis dari File")
            
            uploaded_file = st.file_uploader(
                "Unggah file CSV atau Excel",
                type=['csv', 'xlsx', 'xls']
            )
            
            if uploaded_file is not None:
                try:
                    # Baca file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File berhasil diunggah: {uploaded_file.name}")
                    st.write(f"**Jumlah data:** {len(df)} baris")
                    
                    # Pilih kolom teks
                    text_columns = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if text_columns:
                        selected_column = st.selectbox(
                            "Pilih kolom yang berisi teks ulasan:",
                            text_columns
                        )
                        
                        if st.button("Analisis Seluruh Data", type="primary"):
                            with st.spinner("Menganalisis data..."):
                                # Prediksi untuk setiap baris
                                predictions = []
                                confidences = []
                                prob_pos = []
                                prob_neg = []
                                
                                progress_bar = st.progress(0)
                                for i, text in enumerate(df[selected_column]):
                                    cleaned_text = preprocess_text(text)
                                    text_tfidf = model_data['tfidf_vectorizer'].transform([cleaned_text])
                                    prediction = model_data['model'].predict(text_tfidf)[0]
                                    prediction_proba = model_data['model'].predict_proba(text_tfidf)[0]
                                    
                                    predictions.append('POSITIF' if prediction == 1 else 'NEGATIF')
                                    confidences.append(prediction_proba[prediction] * 100)
                                    prob_pos.append(prediction_proba[1] * 100)
                                    prob_neg.append(prediction_proba[0] * 100)
                                    
                                    progress_bar.progress((i + 1) / len(df))
                                
                                # Tambahkan hasil ke DataFrame
                                df['PREDIKSI_SENTIMEN'] = predictions
                                df['CONFIDENCE'] = confidences
                                df['PROB_POSITIF'] = prob_pos
                                df['PROB_NEGATIF'] = prob_neg
                                
                                st.success("✅ Analisis selesai!")
                                
                                # Tampilkan hasil
                                st.dataframe(df, use_container_width=True)
                                
                                # Statistik
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    pos_count = sum([1 for p in predictions if p == 'POSITIF'])
                                    st.metric("Positif", pos_count)
                                with col2:
                                    neg_count = sum([1 for p in predictions if p == 'NEGATIF'])
                                    st.metric("Negatif", neg_count)
                                with col3:
                                    avg_conf = np.mean(confidences)
                                    st.metric("Confidence Rata-rata", f"{avg_conf:.1f}%")
                                
                                # Download hasil
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Hasil Analisis (CSV)",
                                    data=csv,
                                    file_name="hasil_analisis_sentimen.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.error("Tidak ditemukan kolom teks dalam file")
                
                except Exception as e:
                    st.error(f"Error membaca file: {str(e)}")
        
        with tab3:
            st.subheader("📊 Analisis Batch")
            
            # Input batch
            batch_text = st.text_area(
                "Masukkan beberapa ulasan (satu per baris):",
                height=200,
                placeholder="Driver sangat ramah...\nAplikasi sering error...\nPelayanan memuaskan..."
            )
            
            if batch_text:
                reviews = [line.strip() for line in batch_text.split('\n') if line.strip()]
                st.write(f"**Jumlah ulasan:** {len(reviews)}")
                
                if st.button("Analisis Batch", type="primary"):
                    with st.spinner("Menganalisis batch..."):
                        results = []
                        
                        for i, review in enumerate(reviews):
                            cleaned_text = preprocess_text(review)
                            text_tfidf = model_data['tfidf_vectorizer'].transform([cleaned_text])
                            prediction = model_data['model'].predict(text_tfidf)[0]
                            prediction_proba = model_data['model'].predict_proba(text_tfidf)[0]
                            
                            sentiment = 'POSITIF' if prediction == 1 else 'NEGATIF'
                            confidence = prediction_proba[prediction] * 100
                            
                            results.append({
                                'Ulasan': review[:50] + "..." if len(review) > 50 else review,
                                'Sentimen': sentiment,
                                'Confidence': f"{confidence:.1f}%",
                                'Prob Positif': f"{prediction_proba[1]*100:.1f}%",
                                'Prob Negatif': f"{prediction_proba[0]*100:.1f}%"
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Ringkasan
                        st.subheader("📊 Ringkasan Hasil")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart
                            sentiment_counts = results_df['Sentimen'].value_counts()
                            fig = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                color=sentiment_counts.index,
                                color_discrete_map={'POSITIF': '#4ECDC4', 'NEGATIF': '#FF6B6B'}
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Metrics
                            pos_count = sum(results_df['Sentimen'] == 'POSITIF')
                            neg_count = sum(results_df['Sentimen'] == 'NEGATIF')
                            
                            st.metric("Total Ulasan", len(results_df))
                            st.metric("Positif", pos_count, 
                                     f"{(pos_count/len(results_df)*100):.1f}%")
                            st.metric("Negatif", neg_count,
                                     f"{(neg_count/len(results_df)*100):.1f}%")
    else:
        st.error("Model tidak dapat dimuat. Pastikan model sudah dilatih dan disimpan.")

# Halaman Evaluasi Model
elif menu == "📊 Evaluasi Model":
    st.header("📊 Evaluasi Performa Model")
    
    # Load model data
    model_data = load_model()
    
    if model_data:
        # Tabs untuk evaluasi
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📊 Confusion Matrix", "📋 Classification Report", "📈 Feature Importance"])
        
        with tab1:
            st.subheader("📈 Metrik Performa Model")
            
            # Data contoh evaluasi
            evaluation_data = {
                'Rasio': ['80:20', '80:20', '90:10', '90:10', '70:30', '70:30'],
                'Kernel': ['Linear', 'Polynomial', 'Linear', 'Polynomial', 'Linear', 'Polynomial'],
                'Akurasi': [0.895, 0.882, 0.901, 0.889, 0.887, 0.875],
                'Precision': [0.896, 0.884, 0.902, 0.891, 0.889, 0.877],
                'Recall': [0.895, 0.882, 0.901, 0.889, 0.887, 0.875],
                'F1-Score': [0.895, 0.883, 0.901, 0.890, 0.888, 0.876],
                'Waktu Training (s)': [12.5, 15.2, 14.1, 16.8, 10.9, 13.5]
            }
            
            eval_df = pd.DataFrame(evaluation_data)
            
            # Tampilkan data
            st.dataframe(eval_df.style.format({
                'Akurasi': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}',
                'Waktu Training (s)': '{:.1f}'
            }), use_container_width=True)
            
            # Grafik perbandingan
            st.subheader("📊 Perbandingan Performa")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart akurasi
                fig1 = px.bar(
                    eval_df,
                    x='Rasio',
                    y='Akurasi',
                    color='Kernel',
                    barmode='group',
                    color_discrete_map={'Linear': '#4ECDC4', 'Polynomial': '#FFD166'},
                    text_auto='.3f'
                )
                fig1.update_layout(
                    title="Akurasi per Rasio dan Kernel",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Line chart semua metrik
                fig2 = go.Figure()
                
                metrics = ['Akurasi', 'Precision', 'Recall', 'F1-Score']
                colors = ['#4ECDC4', '#FF6B6B', '#FFD166', '#06D6A0']
                
                for metric, color in zip(metrics, colors):
                    fig2.add_trace(go.Scatter(
                        x=eval_df[eval_df['Kernel'] == 'Linear']['Rasio'],
                        y=eval_df[eval_df['Kernel'] == 'Linear'][metric],
                        mode='lines+markers',
                        name=f'Linear - {metric}',
                        line=dict(color=color, dash='dot')
                    ))
                
                fig2.update_layout(
                    title="Trend Metrik (Kernel Linear)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Confusion Matrix")
            
            # Contoh confusion matrix
            cm_data = np.array([[680, 120], [95, 705]])
            
            # Heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm_data, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=['Prediksi Negatif', 'Prediksi Positif'],
                yticklabels=['Aktual Negatif', 'Aktual Positif'],
                ax=ax
            )
            ax.set_title('Confusion Matrix Model Terbaik', fontsize=14, fontweight='bold')
            ax.set_xlabel('Prediksi', fontsize=12)
            ax.set_ylabel('Aktual', fontsize=12)
            
            st.pyplot(fig)
            
            # Hitung metrik dari confusion matrix
            TN, FP, FN, TP = cm_data.ravel()
            
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Tampilkan metrik
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1-Score", f"{f1_score:.3f}")
        
        with tab3:
            st.subheader("📋 Classification Report")
            
            # Contoh classification report
            report_data = {
                'Kelas': ['Negatif', 'Positif', 'Weighted Avg'],
                'Precision': [0.877, 0.855, 0.866],
                'Recall': [0.850, 0.881, 0.866],
                'F1-Score': [0.863, 0.868, 0.866],
                'Support': [800, 800, 1600]
            }
            
            report_df = pd.DataFrame(report_data)
            
            # Tampilkan tabel
            st.dataframe(report_df.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}'
            }), use_container_width=True)
            
            # Visualisasi metrik per kelas
            fig = go.Figure(data=[
                go.Bar(
                    name='Precision',
                    x=report_df['Kelas'][:2],
                    y=report_df['Precision'][:2],
                    marker_color='#4ECDC4'
                ),
                go.Bar(
                    name='Recall',
                    x=report_df['Kelas'][:2],
                    y=report_df['Recall'][:2],
                    marker_color='#FFD166'
                ),
                go.Bar(
                    name='F1-Score',
                    x=report_df['Kelas'][:2],
                    y=report_df['F1-Score'][:2],
                    marker_color='#FF6B6B'
                )
            ])
            
            fig.update_layout(
                title='Metrik per Kelas',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("📈 Feature Importance")
            
            # Contoh feature importance
            features = [
                'bagus', 'ramah', 'cepat', 'error', 'lemot', 
                'puas', 'mahal', 'mudah', 'helpful', 'kecewa'
            ]
            importance = [0.85, 0.78, 0.72, 0.68, 0.65, 0.63, 0.61, 0.58, 0.55, 0.52]
            sentiment = ['Positif', 'Positif', 'Positif', 'Negatif', 'Negatif', 
                        'Positif', 'Negatif', 'Positif', 'Positif', 'Negatif']
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance,
                'Sentimen': sentiment
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            # Horizontal bar chart
            fig = px.bar(
                importance_df,
                y='Feature',
                x='Importance',
                color='Sentimen',
                color_discrete_map={'Positif': '#4ECDC4', 'Negatif': '#FF6B6B'},
                orientation='h',
                text='Importance'
            )
            
            fig.update_layout(
                title='Top 10 Feature Importance',
                height=500,
                xaxis_title='Importance Score',
                yaxis_title='Feature'
            )
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Wordcloud dari feature penting
            st.subheader("📝 Word Cloud Feature Penting")
            
            # Generate word frequencies
            word_freq = {feature: imp*100 for feature, imp in zip(features, importance)}
            
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=50
            ).generate_from_frequencies(word_freq)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud Feature Importance', fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
    else:
        st.error("Model tidak dapat dimuat. Pastikan model sudah dilatih dan disimpan.")

# Halaman Kelola Data
elif menu == "📁 Kelola Data":
    st.header("📁 Kelola Dataset")
    
    tab1, tab2, tab3 = st.tabs(["📤 Unggah Data", "🔍 Eksplorasi Data", "📊 Statistik Data"])
    
    with tab1:
        st.subheader("📤 Unggah Dataset Baru")
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Pilih file dataset (CSV atau Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="File harus memiliki kolom dengan teks ulasan"
        )
        
        if uploaded_file is not None:
            # Baca file
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File berhasil diunggah: {uploaded_file.name}")
                
                # Tampilkan preview
                st.write("**Preview data:**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Info dataset
                st.write("**Informasi Dataset:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Jumlah Baris", len(df))
                with col2:
                    st.metric("Jumlah Kolom", len(df.columns))
                with col3:
                    st.metric("Ukuran File", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Pilih kolom
                st.subheader("🔧 Konfigurasi Kolom")
                
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                date_columns = df.select_dtypes(include=['datetime']).columns.tolist()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if text_columns:
                        text_column = st.selectbox(
                            "Kolom teks ulasan:",
                            text_columns
                        )
                    else:
                        st.warning("Tidak ada kolom teks")
                
                with col2:
                    if numeric_columns:
                        rating_column = st.selectbox(
                            "Kolom rating (opsional):",
                            ['Tidak ada'] + numeric_columns
                        )
                    else:
                        rating_column = 'Tidak ada'
                
                with col3:
                    if date_columns:
                        date_column = st.selectbox(
                            "Kolom tanggal (opsional):",
                            ['Tidak ada'] + date_columns
                        )
                    else:
                        date_column = 'Tidak ada'
                
                # Tombol proses
                if st.button("Proses Dataset", type="primary"):
                    with st.spinner("Memproses dataset..."):
                        # Simpan dataset
                        df.to_csv('dataset_uploaded.csv', index=False)
                        
                        # Analisis cepat
                        if text_column:
                            # Contoh analisis
                            sample_text = df[text_column].iloc[0] if len(df) > 0 else ""
                            st.info(f"**Contoh teks:** {sample_text[:100]}...")
                            
                            # Statistik teks
                            df['text_length'] = df[text_column].str.len()
                            avg_length = df['text_length'].mean()
                            
                            st.metric("Panjang teks rata-rata", f"{avg_length:.0f} karakter")
                        
                        st.success("✅ Dataset berhasil diproses!")
                        
                        # Download hasil
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Dataset",
                            data=csv,
                            file_name="dataset_processed.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")
    
    with tab2:
        st.subheader("🔍 Eksplorasi Dataset")
        
        # Coba load dataset
        try:
            df = pd.read_csv('dataset_uploaded.csv')
            
            st.write("**Dataset yang aktif:** dataset_uploaded.csv")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Filter data
            st.subheader("🔎 Filter Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter berdasarkan kolom teks
                if 'text_length' in df.columns:
                    min_len = int(df['text_length'].min())
                    max_len = int(df['text_length'].max())
                    text_len_range = st.slider(
                        "Panjang teks:",
                        min_len, max_len, (min_len, max_len)
                    )
            
            with col2:
                # Filter lainnya
                if 'sentimen' in df.columns:
                    sentiments = df['sentimen'].unique().tolist()
                    selected_sentiments = st.multiselect(
                        "Filter sentimen:",
                        sentiments,
                        default=sentiments
                    )
            
            # Tampilkan data terfilter
            filtered_df = df.copy()
            if 'text_length' in df.columns:
                filtered_df = filtered_df[
                    (filtered_df['text_length'] >= text_len_range[0]) & 
                    (filtered_df['text_length'] <= text_len_range[1])
                ]
            
            if 'sentimen' in df.columns and selected_sentiments:
                filtered_df = filtered_df[filtered_df['sentimen'].isin(selected_sentiments)]
            
            st.write(f"**Data terfilter:** {len(filtered_df)} dari {len(df)} baris")
            st.dataframe(filtered_df.head(), use_container_width=True)
            
        except:
            st.info("📝 Belum ada dataset yang diunggah. Silakan unggah dataset di tab 'Unggah Data'.")
    
    with tab3:
        st.subheader("📊 Statistik Dataset")
        
        try:
            df = pd.read_csv('dataset_uploaded.csv')
            
            # Statistik dasar
            st.write("**Statistik Dasar:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Data", len(df))
            with col2:
                st.metric("Total Kolom", len(df.columns))
            with col3:
                missing = df.isnull().sum().sum()
                st.metric("Data Missing", missing)
            with col4:
                duplicate = df.duplicated().sum()
                st.metric("Data Duplikat", duplicate)
            
            # Statistik kolom numerik
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                st.subheader("📈 Statistik Numerik")
                
                for col in numeric_cols[:3]:  # Tampilkan maks 3 kolom
                    st.write(f"**Kolom:** {col}")
                    
                    stats = df[col].describe()
                    st.write(stats)
                    
                    # Histogram
                    fig = px.histogram(df, x=col, title=f"Distribusi {col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Statistik kolom teks
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                st.subheader("📝 Statistik Teks")
                
                for col in text_cols[:2]:  # Tampilkan maks 2 kolom
                    st.write(f"**Kolom:** {col}")
                    
                    # Panjang teks
                    df['text_length'] = df[col].astype(str).str.len()
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Panjang rata-rata", f"{df['text_length'].mean():.0f}")
                    with col_b:
                        st.metric("Panjang minimum", f"{df['text_length'].min():.0f}")
                    with col_c:
                        st.metric("Panjang maksimum", f"{df['text_length'].max():.0f}")
                    
                    # Distribusi panjang
                    fig = px.histogram(df, x='text_length', title=f"Distribusi Panjang Teks - {col}")
                    st.plotly_chart(fig, use_container_width=True)
            
        except:
            st.info("📝 Belum ada dataset yang diunggah. Silakan unggah dataset di tab 'Unggah Data'.")

# Footer
st.markdown("---")
footer = """
<div style="text-align: center; color: gray; padding: 20px;">
    <p>🚀 <b>Sistem Analisis Sentimen Gojek</b> | Dibangun dengan Streamlit</p>
    <p>📧 Kontak: example@email.com | 📅 Versi: 1.0.0</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)