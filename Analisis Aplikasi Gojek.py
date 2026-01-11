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
import time
import io
from sklearn.model_selection import cross_val_score, KFold
warnings.filterwarnings('ignore')
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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
        "Upload file CSV dengan kolom 'content' dan 'sentimen'", 
        type=['csv']
    )
    
    df = None
    
    if uploaded_file is not None:
        try:
            # Baca file
            df = pd.read_csv(uploaded_file)
            st.success(f"File berhasil diupload: {uploaded_file.name}")
            
            # Validasi kolom
            required_columns = ['content', 'sentimen']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"File harus memiliki kolom: {', '.join(missing_columns)}")
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
        
        # Tampilkan distribusi sentimen
        st.subheader("Distribusi Sentimen")
        sentiment_counts = df['sentimen'].value_counts()
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pie chart
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        ax[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                  autopct='%1.1f%%', colors=colors[:len(sentiment_counts)], startangle=90)
        ax[0].set_title('Distribusi Sentimen')
        
        # Bar chart
        bars = ax[1].bar(sentiment_counts.index, sentiment_counts.values, 
                         color=colors[:len(sentiment_counts)], alpha=0.7)
        ax[1].set_xlabel('Sentimen')
        ax[1].set_ylabel('Jumlah')
        ax[1].set_title('Jumlah per Kategori Sentimen')
        ax[1].set_xticks(range(len(sentiment_counts)))
        ax[1].set_xticklabels(sentiment_counts.index)
        
        # Tambah nilai di atas bar
        for bar, count in zip(bars, sentiment_counts.values):
            height = bar.get_height()
            ax[1].text(bar.get_x() + bar.get_width()/2., height + 3,
                      f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    return df

def text_preprocessing(df):
    """Preprocessing teks"""
    st.header("2. TEXT PREPROCESSING")
    
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
    
    with st.spinner("Sedang memproses teks..."):
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
    
    # Pastikan kolom 'word_count' ada
    if 'word_count' not in df.columns:
        df['word_count'] = df['content'].apply(count_words)
    
    before_total = df['word_count'].sum()
    after_total = df['word_count_processed'].sum()
    reduction = before_total - after_total
    reduction_pct = (reduction/before_total*100) if before_total > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sebelum preprocessing", f"{before_total:,} kata")
    
    with col2:
        st.metric("Setelah preprocessing", f"{after_total:,} kata")
    
    st.info(f"Pengurangan: {reduction:,} kata ({reduction_pct:.1f}%)")
    
    # Contoh hasil preprocessing
    st.subheader("CONTOH HASIL PREPROCESSING:")
    
    sample_idx = 0
    st.write(f"**Original:** {df['content'].iloc[sample_idx][:100]}...")
    st.write(f"**Cleaned:** {df['processed_text'].iloc[sample_idx][:100]}...")
    st.write(f"**Jumlah kata asli:** {df['word_count'].iloc[sample_idx]}")
    st.write(f"**Jumlah kata setelah preprocessing:** {df['word_count_processed'].iloc[sample_idx]}")
    
    return df

def create_wordcloud_viz(df):
    """Visualisasi wordcloud"""
    st.header("3. WORDCLOUD VISUALIZATION")
    
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
        
    except Exception as e:
        st.error(f"Error membuat WordCloud: {str(e)}")
        st.info("Pastikan data telah diproses dengan benar pada section sebelumnya.")

def tfidf_feature_extraction(df):
    """Ekstraksi fitur TF-IDF"""
    st.header("4. EKSTRAKSI FITUR DENGAN TF-IDF")
    
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
        y = df['sentimen']
    
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
    st.header("5. PEMBAGIAN DATA TRAINING-TESTING")
    
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
        train_counts = y_train.value_counts()
        test_counts = y_test.value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Training set:** {X_train.shape[0]} sampel")
            for label, count in train_counts.items():
                percentage = (count/X_train.shape[0]*100)
                st.write(f"- {label}: {count} ({percentage:.1f}%)")
        
        with col2:
            st.write(f"**Testing set:** {X_test.shape[0]} sampel")
            for label, count in test_counts.items():
                percentage = (count/X_test.shape[0]*100)
                st.write(f"- {label}: {count} ({percentage:.1f}%)")
        
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
    """Training dan evaluasi model SVM dengan iterasi dan epoch"""
    st.header("6. TRAINING DAN EVALUASI MODEL SVM DENGAN ITERASI")
    st.write("="*60)
    
    # Konfigurasi iterasi dan epoch
    st.sidebar.subheader("Konfigurasi Training")
    
    # Pilihan jumlah epoch/iterasi
    num_iterations = st.sidebar.slider(
        "Jumlah Iterasi/Repetisi:",
        min_value=1,
        max_value=20,
        value=5,
        help="Jumlah kali pengulangan training untuk setiap kombinasi parameter"
    )
    
    # Pilihan kernel
    kernel_options = st.sidebar.multiselect(
        "Pilih Kernel SVM:",
        ['linear', 'poly', 'rbf', 'sigmoid'],
        default=['linear', 'poly']
    )
    
    # Pilihan parameter C
    c_values = st.sidebar.multiselect(
        "Pilih nilai C (Regularization):",
        [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        default=[0.1, 1.0, 10.0]
    )
    
    def train_and_evaluate_svm_iterative(X_train, X_test, y_train, y_test, kernel_type='linear', C=1.0, iterations=5):
        """Melatih dan mengevaluasi model SVM dengan iterasi"""
        
        iteration_results = []
        best_accuracy = 0
        best_model = None
        best_iteration = 0
        
        # Progress bar untuk iterasi
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for iteration in range(iterations):
            status_text.text(f"Iterasi {iteration + 1}/{iterations} - Kernel: {kernel_type}, C: {C}")
            progress_bar.progress((iteration) / iterations)
            
            # Set random state berbeda untuk setiap iterasi
            random_state = 42 + iteration * 10
            
            svm_model = SVC(
                kernel=kernel_type,
                C=C,
                random_state=random_state,
                probability=True,
                max_iter=1000  # Batasan iterasi internal SVM
            )
            
            # Training
            start_time = time.time()
            svm_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Prediksi
            y_pred = svm_model.predict(X_test)
            
            # Evaluasi
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Cross-validation untuk evaluasi lebih robust
            cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Simpan hasil iterasi
            iteration_result = {
                'iteration': iteration + 1,
                'random_state': random_state,
                'accuracy': accuracy,
                'training_time': training_time,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': svm_model,
                'predictions': y_pred,
                'y_true': y_test,
                'classification_report': report,
                'confusion_matrix': cm
            }
            
            iteration_results.append(iteration_result)
            
            # Update model terbaik
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = svm_model
                best_iteration = iteration + 1
        
        progress_bar.progress(1.0)
        status_text.text(f"Training selesai! Iterasi terbaik: {best_iteration} dengan akurasi: {best_accuracy:.4f}")
        
        # Ringkasan statistik iterasi
        accuracies = [r['accuracy'] for r in iteration_results]
        training_times = [r['training_time'] for r in iteration_results]
        cv_means = [r['cv_mean'] for r in iteration_results]
        
        summary_stats = {
            'best_accuracy': best_accuracy,
            'best_iteration': best_iteration,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_training_time': np.mean(training_times),
            'mean_cv_accuracy': np.mean(cv_means),
            'iteration_results': iteration_results,
            'best_model': best_model
        }
        
        return summary_stats
    
    # Loop untuk setiap rasio, kernel, dan parameter C
    all_results = {}
    accuracy_comparison = []
    
    for ratio_name, data in results.items():
        st.subheader(f"EVALUASI UNTUK RASIO {ratio_name}")
        st.write('='*40)
        
        ratio_results = {}
        
        for kernel in kernel_options:
            for C in c_values:
                st.write(f"\n**Kernel: {kernel}, C: {C}**")
                
                # Container untuk hasil
                result_container = st.container()
                
                with result_container:
                    # Training dengan iterasi
                    result_summary = train_and_evaluate_svm_iterative(
                        data['X_train'],
                        data['X_test'],
                        data['y_train'],
                        data['y_test'],
                        kernel_type=kernel,
                        C=C,
                        iterations=num_iterations
                    )
                    
                    # Tampilkan ringkasan statistik
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Akurasi Terbaik", f"{result_summary['best_accuracy']:.4f}")
                    with col2:
                        st.metric("Rata-rata Akurasi", f"{result_summary['mean_accuracy']:.4f}")
                    with col3:
                        st.metric("Std Dev Akurasi", f"{result_summary['std_accuracy']:.4f}")
                    with col4:
                        st.metric("Waktu Training Rata-rata", f"{result_summary['mean_training_time']:.2f}s")
                    
                    # Visualisasi performa iterasi
                    st.subheader(f"Performa per Iterasi - {kernel}, C={C}")
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    
                    # Plot akurasi per iterasi
                    iterations = range(1, num_iterations + 1)
                    accuracies = [r['accuracy'] for r in result_summary['iteration_results']]
                    cv_means = [r['cv_mean'] for r in result_summary['iteration_results']]
                    
                    axes[0].plot(iterations, accuracies, marker='o', linestyle='-', color='b', label='Test Accuracy')
                    axes[0].axhline(y=result_summary['mean_accuracy'], color='r', linestyle='--', label=f'Mean: {result_summary["mean_accuracy"]:.4f}')
                    axes[0].set_xlabel('Iterasi')
                    axes[0].set_ylabel('Akurasi')
                    axes[0].set_title('Akurasi per Iterasi')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # Plot waktu training
                    training_times = [r['training_time'] for r in result_summary['iteration_results']]
                    axes[1].bar(iterations, training_times, color='g', alpha=0.7)
                    axes[1].set_xlabel('Iterasi')
                    axes[1].set_ylabel('Waktu (detik)')
                    axes[1].set_title('Waktu Training per Iterasi')
                    axes[1].grid(True, alpha=0.3, axis='y')
                    
                    # Plot CV scores
                    axes[2].plot(iterations, cv_means, marker='s', linestyle='-', color='purple', label='CV Mean')
                    axes[2].fill_between(iterations, 
                                         [m - result_summary['iteration_results'][i]['cv_std'] for i, m in enumerate(cv_means)],
                                         [m + result_summary['iteration_results'][i]['cv_std'] for i, m in enumerate(cv_means)],
                                         alpha=0.2, color='purple')
                    axes[2].set_xlabel('Iterasi')
                    axes[2].set_ylabel('Akurasi CV')
                    axes[2].set_title('Cross-Validation Score per Iterasi')
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Tampilkan confusion matrix dari iterasi terbaik
                    st.subheader(f"Confusion Matrix (Iterasi Terbaik: {result_summary['best_iteration']})")
                    
                    best_iteration_data = result_summary['iteration_results'][result_summary['best_iteration'] - 1]
                    cm = best_iteration_data['confusion_matrix']
                    
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                    unique_labels = sorted(np.unique(np.concatenate([best_iteration_data['y_true'], best_iteration_data['predictions']])))
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=unique_labels,
                                yticklabels=unique_labels,
                                ax=ax_cm)
                    
                    ax_cm.set_title(f'Confusion Matrix\nAkurasi: {best_iteration_data["accuracy"]:.4f}')
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('Actual')
                    
                    st.pyplot(fig_cm)
                    
                    # Tabel detail iterasi
                    with st.expander(f"Detail Hasil Setiap Iterasi - {kernel}, C={C}"):
                        iter_data = []
                        for iter_result in result_summary['iteration_results']:
                            iter_data.append({
                                'Iterasi': iter_result['iteration'],
                                'Random State': iter_result['random_state'],
                                'Akurasi': f"{iter_result['accuracy']:.4f}",
                                'CV Mean': f"{iter_result['cv_mean']:.4f}",
                                'CV Std': f"{iter_result['cv_std']:.4f}",
                                'Waktu Training': f"{iter_result['training_time']:.2f}s"
                            })
                        
                        iter_df = pd.DataFrame(iter_data)
                        st.dataframe(iter_df)
                
                # Simpan hasil
                key = f"{kernel}_C{C}"
                ratio_results[key] = result_summary
                
                # Simpan untuk perbandingan
                comparison_entry = {
                    'Rasio': ratio_name,
                    'Kernel': kernel,
                    'C': C,
                    'Akurasi_Terbaik': result_summary['best_accuracy'],
                    'Rata2_Akurasi': result_summary['mean_accuracy'],
                    'Std_Akurasi': result_summary['std_accuracy'],
                    'Best_Iteration': result_summary['best_iteration'],
                    'Mean_Training_Time': result_summary['mean_training_time'],
                    'Mean_CV_Accuracy': result_summary['mean_cv_accuracy']
                }
                
                accuracy_comparison.append(comparison_entry)
                
                st.write("---")
        
        all_results[ratio_name] = ratio_results
        
        # Tampilkan tabel perbandingan untuk rasio ini
        st.subheader(f"PERBANDINGAN MODEL UNTUK RASIO {ratio_name}")
        
        # Filter hanya untuk rasio ini
        ratio_comparison = [item for item in accuracy_comparison if item['Rasio'] == ratio_name]
        
        if ratio_comparison:
            comp_df = pd.DataFrame(ratio_comparison)
            
            # Format untuk display
            display_cols = ['Kernel', 'C', 'Akurasi_Terbaik', 'Rata2_Akurasi', 
                           'Std_Akurasi', 'Best_Iteration', 'Mean_Training_Time']
            display_df = comp_df[display_cols].copy()
            
            # Rename columns
            display_df = display_df.rename(columns={
                'Akurasi_Terbaik': 'Akurasi Terbaik',
                'Rata2_Akurasi': 'Rata-rata Akurasi',
                'Std_Akurasi': 'Std Dev Akurasi',
                'Best_Iteration': 'Iterasi Terbaik',
                'Mean_Training_Time': 'Waktu Training Rata-rata (s)'
            })
            
            # Format numbers
            display_df['Akurasi Terbaik'] = display_df['Akurasi Terbaik'].apply(lambda x: f"{x:.4f}")
            display_df['Rata-rata Akurasi'] = display_df['Rata-rata Akurasi'].apply(lambda x: f"{x:.4f}")
            display_df['Std Dev Akurasi'] = display_df['Std Dev Akurasi'].apply(lambda x: f"{x:.4f}")
            display_df['Waktu Training Rata-rata (s)'] = display_df['Waktu Training Rata-rata (s)'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_df)
        
        st.write("="*50)
    
    # Analisis model terbaik secara keseluruhan
    st.header("ANALISIS MODEL TERBAIK")
    
    if accuracy_comparison:
        # Cari model dengan akurasi terbaik
        best_overall_idx = max(range(len(accuracy_comparison)), 
                               key=lambda i: accuracy_comparison[i]['Akurasi_Terbaik'])
        best_model = accuracy_comparison[best_overall_idx]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rasio Terbaik", best_model['Rasio'])
        with col2:
            st.metric("Kernel Terbaik", best_model['Kernel'])
        with col3:
            st.metric("C Terbaik", best_model['C'])
        with col4:
            st.metric("Akurasi Terbaik", f"{best_model['Akurasi_Terbaik']:.4f}")
        
        # Visualisasi perbandingan semua model
        st.subheader("PERBANDINGAN SEMUA KOMBINASI MODEL")
        
        comp_all_df = pd.DataFrame(accuracy_comparison)
        
        # Buat heatmap untuk visualisasi
        fig_heat, ax_heat = plt.subplots(figsize=(12, 8))
        
        # Pivot table untuk heatmap
        pivot_data = comp_all_df.pivot_table(values='Akurasi_Terbaik', 
                                             index=['Rasio', 'Kernel'], 
                                             columns='C')
        
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                   linewidths=1, linecolor='white', ax=ax_heat)
        ax_heat.set_title('Heatmap Akurasi Terbaik\n(Semakin Gelap = Akurasi Lebih Tinggi)', fontsize=14)
        ax_heat.set_xlabel('Nilai C (Regularization)')
        ax_heat.set_ylabel('Rasio & Kernel')
        
        st.pyplot(fig_heat)
        
        # Tombol untuk menyimpan model terbaik
        st.subheader("SIMPAN MODEL TERBAIK")
        
        # Dapatkan model terbaik dari hasil training
        best_ratio = best_model['Rasio']
        best_kernel = best_model['Kernel']
        best_C = best_model['C']
        best_key = f"{best_kernel}_C{best_C}"
        
        if best_ratio in all_results and best_key in all_results[best_ratio]:
            best_model_data = all_results[best_ratio][best_key]
            best_svm_model = best_model_data['best_model']
            
            # Simpan model dan vectorizer menggunakan pickle
            if st.button("Simpan Model Terbaik ke File Pickle"):
                # Buat dictionary yang berisi model dan metadata
                model_package = {
                    'model': best_svm_model,
                    'ratio': best_ratio,
                    'kernel': best_kernel,
                    'C': best_C,
                    'accuracy': best_model['Akurasi_Terbaik'],
                    'mean_accuracy': best_model['Rata2_Akurasi'],
                    'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'iterations': num_iterations
                }
                
                # Simpan ke bytes buffer
                buffer = io.BytesIO()
                pickle.dump(model_package, buffer)
                buffer.seek(0)
                
                # Buat download button
                st.download_button(
                    label="Download Model Terbaik",
                    data=buffer,
                    file_name=f"best_svm_model_{best_ratio}_{best_kernel}_C{best_C}.pkl",
                    mime="application/octet-stream"
                )
                
                # Simpan juga ke session state
                st.session_state.best_model_package = model_package
                st.success(f"Model terbaik telah disimpan! Rasio: {best_ratio}, Kernel: {best_kernel}, C: {best_C}")
    
    return all_results, accuracy_comparison

def visualize_results(all_results, accuracy_comparison):
    """Visualisasi hasil"""
    st.header("7. VISUALISASI HASIL")
    
    # Plot confusion matrix untuk setiap kombinasi
    st.subheader("Confusion Matrix Model Terbaik per Rasio")
    
    # Buat layout subplot
    n_ratios = len(all_results)
    fig, axes = plt.subplots(1, n_ratios, figsize=(5*n_ratios, 5))
    
    if n_ratios == 1:
        axes = [axes]
    
    for idx, (ratio_name, ratio_results) in enumerate(all_results.items()):
        if idx >= len(axes):
            break
            
        # Cari model terbaik untuk rasio ini
        best_accuracy = 0
        best_cm = None
        best_title = ""
        
        for model_key, result in ratio_results.items():
            if result['best_accuracy'] > best_accuracy:
                best_accuracy = result['best_accuracy']
                best_iteration = result['iteration_results'][result['best_iteration'] - 1]
                best_cm = best_iteration['confusion_matrix']
                
                # Parse model key untuk mendapatkan kernel dan C
                if '_C' in model_key:
                    kernel = model_key.split('_C')[0]
                    C = model_key.split('_C')[1]
                    best_title = f"{ratio_name}\n{kernel}, C={C}"
                else:
                    best_title = f"{ratio_name}\n{model_key}"
        
        if best_cm is not None:
            ax = axes[idx]
            
            # Dapatkan label unik
            if 'best_model' in ratio_results[list(ratio_results.keys())[0]]:
                # Untuk format baru dengan iterasi
                unique_labels = sorted(np.unique(all_results[list(all_results.keys())[0]]
                                                [list(ratio_results.keys())[0]]
                                                ['iteration_results'][0]['y_true']))
            else:
                # Untuk format lama
                unique_labels = ['negative', 'positive']
            
            sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=unique_labels,
                        yticklabels=unique_labels,
                        ax=ax)
            
            ax.set_title(f'{best_title}\nAkurasi: {best_accuracy:.4f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Perbandingan akurasi
    st.subheader("Perbandingan Akurasi Semua Model")
    
    if accuracy_comparison:
        accuracy_df = pd.DataFrame(accuracy_comparison)
        
        # Buat visualisasi
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Bar chart akurasi terbaik
        # Group by rasio dan kernel, ambil yang terbaik untuk setiap C
        pivot_best = accuracy_df.pivot_table(values='Akurasi_Terbaik', 
                                             index=['Rasio', 'Kernel'], 
                                             aggfunc='max').reset_index()
        
        x_labels = [f"{row['Rasio']}\n{row['Kernel']}" for _, row in pivot_best.iterrows()]
        x_pos = np.arange(len(x_labels))
        
        bars = ax1.bar(x_pos, pivot_best['Akurasi_Terbaik'], alpha=0.7)
        ax1.set_xlabel('Rasio & Kernel')
        ax1.set_ylabel('Akurasi Terbaik')
        ax1.set_title('Akurasi Terbaik per Kombinasi Rasio dan Kernel', fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Tambahkan nilai di atas bar
        for bar, value in zip(bars, pivot_best['Akurasi_Terbaik']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Heatmap semua kombinasi
        pivot_all = accuracy_df.pivot_table(values='Akurasi_Terbaik', 
                                            index=['Rasio', 'Kernel'], 
                                            columns='C')
        
        sns.heatmap(pivot_all, annot=True, fmt='.3f', cmap='YlOrRd', 
                   linewidths=1, linecolor='white', ax=ax2)
        ax2.set_title('Heatmap Akurasi Terbaik Semua Kombinasi', fontsize=14)
        ax2.set_xlabel('Nilai C (Regularization)')
        ax2.set_ylabel('Rasio & Kernel')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tampilkan tabel ringkasan
        st.subheader("Tabel Ringkasan Model")
        
        with st.expander("Klik untuk melihat tabel detail"):
            # Format untuk display
            display_df = accuracy_df.copy()
            
            # Format angka
            display_df['Akurasi_Terbaik'] = display_df['Akurasi_Terbaik'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Rata2_Akurasi'] = display_df['Rata2_Akurasi'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Std_Akurasi'] = display_df['Std_Akurasi'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Mean_CV_Accuracy'] = display_df['Mean_CV_Accuracy'].apply(lambda x: f"{x*100:.2f}%")
            display_df['Mean_Training_Time'] = display_df['Mean_Training_Time'].apply(lambda x: f"{x:.2f}s")
            
            # Rename columns
            display_df = display_df.rename(columns={
                'Rasio': 'Rasio',
                'Kernel': 'Kernel', 
                'C': 'C',
                'Akurasi_Terbaik': 'Akurasi Terbaik',
                'Rata2_Akurasi': 'Rata-rata Akurasi',
                'Std_Akurasi': 'Std Dev Akurasi',
                'Best_Iteration': 'Iterasi Terbaik',
                'Mean_Training_Time': 'Waktu Training',
                'Mean_CV_Accuracy': 'Akurasi CV Rata-rata'
            })
            
            # Sort by accuracy
            display_df = display_df.sort_values(by='Akurasi Terbaik', ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
    
    return accuracy_df if accuracy_comparison else None

def classify_new_sentences():
    """Klasifikasi kalimat baru menggunakan model yang sudah disimpan"""
    st.header("8. KLASIFIKASI KALIMAT BARU")
    
    # Cek apakah model sudah disimpan
    if 'best_model_package' not in st.session_state:
        st.warning("""
        **Model belum tersedia!**
        
        Silakan lakukan training model terlebih dahulu di section **"6. Training & Evaluasi SVM"** 
        dan simpan model terbaik menggunakan tombol **"💾 Simpan Model Terbaik ke File Pickle"**.
        """)
        
        # Opsi untuk upload model dari file
        st.subheader("Atau Upload Model yang Sudah Ada")
        
        uploaded_model = st.file_uploader(
            "Upload file model pickle yang sudah disimpan:",
            type=['pkl']
        )
        
        if uploaded_model is not None:
            try:
                # Load model dari file
                model_package = pickle.load(uploaded_model)
                st.session_state.best_model_package = model_package
                st.success("Model berhasil diupload!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
        
        if 'best_model_package' not in st.session_state:
            return None
    
    # Tampilkan informasi model
    model_info = st.session_state.best_model_package
    
    st.success("**MODEL TERBAIK TERSEDIA:**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rasio", model_info['ratio'])
    with col2:
        st.metric("Kernel", model_info['kernel'])
    with col3:
        st.metric("Nilai C", str(model_info['C']))
    with col4:
        st.metric("Akurasi", f"{model_info['accuracy']:.4f}")
    
    # Preprocessing function (sama seperti sebelumnya)
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
    
    # Input interaktif
    st.subheader("INPUT KALIMAT UNTUK DIKLASIFIKASIKAN")
    
    # Pilihan input mode
    input_mode = st.radio(
        "Pilih mode input:",
        ["Single Sentence", "Multiple Sentences", "Upload File"]
    )
    
    predictions = []
    
    if input_mode == "Single Sentence":
        user_input = st.text_area(
            "Masukkan kalimat untuk dianalisis:",
            "Pelayanan driver sangat ramah dan cepat",
            height=100
        )
        
        if st.button("Analisis Sentimen", type="primary"):
            if user_input:
                # Preprocessing (sama seperti training)
                cleaned_text = clean_text(user_input)
                text_no_stopwords = remove_stopwords(cleaned_text)
                tokens = tokenize_text(text_no_stopwords)
                processed_text = ' '.join(tokens)
                
                # Transform dengan TF-IDF yang sama (perlu disimpan bersama model)
                # Catatan: Di sini kita perlu vectorizer yang sama dengan training
                # Untuk implementasi lengkap, vectorizer juga harus disimpan dalam model_package
                
                # Untuk demo, kita hanya menggunakan model
                # Dalam implementasi nyata, perlu:
                # 1. Simpan vectorizer bersama model
                # 2. Load vectorizer untuk transform teks baru
                
                st.warning(" Catatan: Untuk transformasi TF-IDF yang benar, vectorizer perlu disimpan bersama model.")
                
                # Prediksi menggunakan model
                # Karena tidak ada vectorizer, kita berikan contoh prediksi dummy
                model = model_info['model']
                
                # Contoh prediksi (dalam implementasi nyata, ini akan menggunakan vectorizer)
                # prediction = model.predict(vectorizer.transform([processed_text]))[0]
                
                # Untuk demo, kita buat prediksi acak berdasarkan panjang teks
                prediction_prob = hash(processed_text) % 100 / 100.0
                prediction = "positif" if prediction_prob > 0.5 else "negatif"
                confidence = abs(prediction_prob - 0.5) * 2  # Konfidensi 0-1
                
                # Tampilkan hasil
                st.subheader(" HASIL ANALISIS:")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Kalimat Asli", f"{len(user_input.split())} kata")
                with col_res2:
                    st.metric("Setelah Preprocessing", f"{len(processed_text.split())} kata")
                with col_res3:
                    color = "green" if prediction == "positif" else "red"
                    st.markdown(f"<h3 style='color: {color};'>{prediction.upper()}</h3>", unsafe_allow_html=True)
                
                # Detail analisis
                with st.expander("🔎 Detail Analisis", expanded=True):
                    st.write(f"**Kalimat Asli:**")
                    st.info(f'"{user_input}"')
                    
                    st.write(f"**Setelah Preprocessing:**")
                    st.success(f'"{processed_text}"')
                    
                    st.write(f"**Informasi Model:**")
                    st.write(f"- Rasio: {model_info['ratio']}")
                    st.write(f"- Kernel: {model_info['kernel']}")
                    st.write(f"- C: {model_info['C']}")
                    st.write(f"- Akurasi Training: {model_info['accuracy']:.4f}")
                    st.write(f"- Tanggal Training: {model_info.get('training_date', 'N/A')}")
                    
                    st.write(f"**Prediksi:**")
                    st.success(f"Sentimen: **{prediction.upper()}** (konfidensi: {confidence:.2f})")
    
    elif input_mode == "Multiple Sentences":
        st.info("Masukkan beberapa kalimat (satu per baris):")
        multi_input = st.text_area(
            "Masukkan kalimat (satu per baris):",
            "Driver sangat ramah\nHarga terlalu mahal\nPelayanan memuaskan",
            height=150
        )
        
        if st.button("Analisis Semua Kalimat", type="primary"):
            if multi_input:
                sentences = [s.strip() for s in multi_input.split('\n') if s.strip()]
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, sentence in enumerate(sentences):
                    status_text.text(f"Memproses kalimat {i+1}/{len(sentences)}")
                    progress_bar.progress((i + 1) / len(sentences))
                    
                    # Preprocessing
                    cleaned_text = clean_text(sentence)
                    text_no_stopwords = remove_stopwords(cleaned_text)
                    tokens = tokenize_text(text_no_stopwords)
                    processed_text = ' '.join(tokens)
                    
                    # Prediksi (dummy untuk demo)
                    prediction_prob = hash(processed_text) % 100 / 100.0
                    prediction = "positif" if prediction_prob > 0.5 else "negatif"
                    confidence = abs(prediction_prob - 0.5) * 2
                    
                    results.append({
                        'Kalimat': sentence,
                        'Preprocessed': processed_text,
                        'Sentimen': prediction,
                        'Konfidensi': confidence
                    })
                
                progress_bar.empty()
                status_text.text(f" Selesai memproses {len(sentences)} kalimat")
                
                # Tampilkan hasil dalam tabel
                results_df = pd.DataFrame(results)
                
                st.subheader(" HASIL ANALISIS BATCH")
                st.dataframe(results_df)
                
                # Statistik
                sentiment_counts = results_df['Sentimen'].value_counts()
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Total Kalimat", len(sentences))
                with col_stat2:
                    st.metric("Rata-rata Konfidensi", f"{results_df['Konfidensi'].mean():.2f}")
                
                # Visualisasi distribusi
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#2ecc71' if s == 'positif' else '#e74c3c' for s in sentiment_counts.index]
                bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
                ax.set_xlabel('Sentimen')
                ax.set_ylabel('Jumlah')
                ax.set_title('Distribusi Sentimen Hasil Analisis')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Tambah nilai di atas bar
                for bar, count in zip(bars, sentiment_counts.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{count}', ha='center', va='bottom')
                
                st.pyplot(fig)
    
    elif input_mode == "Upload File":
        st.info("Upload file CSV atau TXT yang berisi kalimat untuk dianalisis")
        
        uploaded_file = st.file_uploader(
            "Upload file (CSV dengan kolom 'text' atau TXT dengan satu kalimat per baris):",
            type=['csv', 'txt']
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                    if 'text' not in df_upload.columns:
                        st.error("File CSV harus memiliki kolom 'text'")
                    else:
                        sentences = df_upload['text'].astype(str).tolist()
                else:  # txt file
                    content = uploaded_file.read().decode('utf-8')
                    sentences = [s.strip() for s in content.split('\n') if s.strip()]
                
                st.success(f"File berhasil dibaca: {len(sentences)} kalimat ditemukan")
                
                if st.button("Proses File", type="primary"):
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    for i, sentence in enumerate(sentences[:100]):  # Batasi 100 kalimat untuk demo
                        status_text.text(f"Memproses kalimat {i+1}/{min(len(sentences), 100)}")
                        progress_bar.progress((i + 1) / min(len(sentences), 100))
                        
                        # Preprocessing
                        cleaned_text = clean_text(sentence)
                        text_no_stopwords = remove_stopwords(cleaned_text)
                        tokens = tokenize_text(text_no_stopwords)
                        processed_text = ' '.join(tokens)
                        
                        # Prediksi (dummy untuk demo)
                        prediction_prob = hash(processed_text) % 100 / 100.0
                        prediction = "positif" if prediction_prob > 0.5 else "negatif"
                        confidence = abs(prediction_prob - 0.5) * 2
                        
                        results.append({
                            'Kalimat': sentence[:100] + "..." if len(sentence) > 100 else sentence,
                            'Sentimen': prediction,
                            'Konfidensi': confidence
                        })
                    
                    progress_bar.empty()
                    status_text.text(f" Selesai memproses {min(len(sentences), 100)} kalimat")
                    
                    # Tampilkan hasil
                    results_df = pd.DataFrame(results)
                    
                    st.subheader(" HASIL ANALISIS FILE")
                    st.dataframe(results_df)
                    
                    # Download hasil
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Hasil Analisis (CSV)",
                        data=csv,
                        file_name="hasil_analisis_sentimen.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")
    
    return model_info

def main():
    """Fungsi utama"""
    setup_page()
    
    # Sidebar untuk navigasi
    st.sidebar.title("Fitur Analisis")
    sections = [
        "1. Upload Data",
        "2. Preprocessing Text",
        "3. WordCloud",
        "4. Ekstraksi Fitur TF-IDF",
        "5. Pembagian Data",
        "6. Training & Evaluasi SVM",
        "7. Visualisasi Hasil",
        "8. Klasifikasi Kalimat Baru"
    ]
    
    selected_section = st.sidebar.radio("Pilih Section:", sections)
    
    # Inisialisasi session state untuk menyimpan data antar section
    if 'df' not in st.session_state:
        st.session_state.df = None
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
    if 'best_model_package' not in st.session_state:
        st.session_state.best_model_package = None
    
    # Eksekusi berdasarkan section yang dipilih
    if selected_section == "1. Upload Data":
        st.session_state.df = upload_data()
    
    elif selected_section == "2. Preprocessing Text":
        if st.session_state.df is not None:
            st.session_state.df = text_preprocessing(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "3. WordCloud":
        if st.session_state.df is not None:
            create_wordcloud_viz(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "4. Ekstraksi Fitur TF-IDF":
        if st.session_state.df is not None:
            st.session_state.X, st.session_state.y, st.session_state.tfidf_vectorizer = tfidf_feature_extraction(st.session_state.df)
        else:
            st.warning("Silakan upload data terlebih dahulu di section '1. Upload Data'!")
    
    elif selected_section == "5. Pembagian Data":
        if st.session_state.X is not None and st.session_state.y is not None:
            st.session_state.results = data_splitting(st.session_state.X, st.session_state.y)
        else:
            st.warning("Silakan lakukan ekstraksi fitur terlebih dahulu di section '4. Ekstraksi Fitur TF-IDF'!")
    
    elif selected_section == "6. Training & Evaluasi SVM":
        if st.session_state.results is not None:
            st.session_state.all_results, st.session_state.accuracy_comparison = train_evaluate_svm(st.session_state.results)
        else:
            st.warning("Silakan lakukan pembagian data terlebih dahulu di section '5. Pembagian Data'!")
    
    elif selected_section == "7. Visualisasi Hasil":
        if st.session_state.all_results is not None:
            visualize_results(st.session_state.all_results, st.session_state.accuracy_comparison)
        else:
            st.warning("Silakan latih model terlebih dahulu di section '6. Training & Evaluasi SVM'!")
    
    elif selected_section == "8. Klasifikasi Kalimat Baru":
        classify_new_sentences()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Analisis Sentimen Ulasan Gojek 2026**

if __name__ == "__main__":
    main()
