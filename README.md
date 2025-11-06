# UTS_Kecerdasan_Buatan
# Deteksi Ujaran Kebencian (Hate Speech Detection)  
Ujian Tengah Semester - Mata Kuliah Kecerdasan Buatan    
Nama: Anindha Latiefa Zahra
NIM : 312210323
Kelas: TI.22.A.SE.1 
Dosen Pengampu: Dr. Muhamad Fatchan, S.Kom., M.Kom.
Program Studi: Teknik Informatika 
Universitas Pelita Bangsa

## Latar Belakang
Media sosial telah menjadi sarana komunikasi utama masyarakat modern. Namun, meningkatnya penggunaan media sosial juga memunculkan berbagai bentuk penyalahgunaan bahasa, termasuk ujaran kebencian (hate speech). Ujaran kebencian dapat menimbulkan perpecahan sosial dan berdampak negatif terhadap individu maupun kelompok tertentu.

Oleh karena itu, diperlukan sistem berbasis Kecerdasan Buatan (AI) yang mampu mendeteksi dan mengklasifikasikan ujaran kebencian secara otomatis. Proyek ini bertujuan membangun model deteksi ujaran kebencian menggunakan teknik pemrosesan bahasa alami (NLP) dan pembelajaran mesin (Machine Learning) untuk membedakan antara teks yang mengandung ujaran kebencian dan yang tidak.

## Metodologi

### 1. Dataset
Dataset yang digunakan adalah `labeled_data.csv`, yang berisi teks (tweet) dengan label:
- 0 → Tidak mengandung ujaran kebencian  
- 1 → Mengandung ujaran kebencian  

Dataset dimuat menggunakan `pandas` dan diperiksa distribusi labelnya untuk memastikan keseimbangan data.

### 2. Preprocessing Teks
Tahapan pembersihan data dilakukan agar model memahami makna teks dengan lebih baik:
- Case Folding: Mengubah semua huruf menjadi huruf kecil.  
- Menghapus Angka & Tanda Baca: Menghapus karakter non-alfabet.  
- Tokenisasi: Memecah teks menjadi kata-kata.  
- Stopword Removal: Menghapus kata umum yang tidak bermakna kontekstual (seperti “dan”, “yang”, “di”).  
- Lemmatization: Mengubah kata ke bentuk dasar menggunakan `WordNetLemmatizer`.

### 3. Representasi Fitur
Model menggunakan representasi teks dengan TF-IDF (Term Frequency–Inverse Document Frequency) untuk mengubah teks menjadi vektor numerik.

### 4. Model Klasifikasi
Algoritma yang digunakan adalah Logistic Regression — salah satu metode Machine Learning klasik yang efektif untuk klasifikasi teks biner.  
Langkah-langkah pelatihan:
- Membagi data menjadi 80% training dan 20% testing.
- Melatih model dengan data training.
- Memprediksi hasil pada data testing.

### 5. Evaluasi
Model dievaluasi menggunakan metrik berikut:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score

## Hasil dan Analisis

### 1. Distribusi Label Dataset
Hasil visualisasi menunjukkan bahwa dataset terdiri dari dua kelas utama:
- Kelas 0 (Tidak Hate): proporsi terbesar
- Kelas 1 (Hate Speech): proporsi lebih kecil namun signifikan  

```python
plt.pie(df['class'].value_counts(), labels=['Tidak Hate', 'Hate'], autopct='%1.1f%%')
plt.title('Distribusi Label Dataset')
