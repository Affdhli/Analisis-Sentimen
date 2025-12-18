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
        st.subheader(f"📊 PERBANDINGAN KERNEL UNTUK RASIO {ratio_name}")
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
    st.header("📈 RINGKASAN SEMUA MODEL")
    
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
    st.subheader("📊 VISUALISASI PERBANDINGAN SEMUA MODEL")
    
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
        st.subheader("🔍 ANALISIS PERFORMA PER KELAS")
        
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
        st.write("**📊 Rekomendasi Model Berdasarkan Keseimbangan Akurasi:**")
        
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
