# ⚡ Sistem Audit Lembur Pembangkit

Web app audit lembur otomatis berbasis AI (Sentence Transformer).

## Cara Deploy ke Streamlit Cloud

### 1. Buat akun GitHub
Pergi ke https://github.com → Sign up (gratis)

### 2. Buat repository baru
- Klik tombol "+" → "New repository"
- Nama: `audit-lembur`
- Pilih: Public
- Klik "Create repository"

### 3. Upload file
Upload ketiga file ini ke repository:
- `app.py`
- `requirements.txt`
- `README.md`

### 4. Deploy ke Streamlit Cloud
- Pergi ke https://streamlit.io → Sign in with GitHub
- Klik "New app"
- Pilih repository: `audit-lembur`
- Main file: `app.py`
- Klik "Deploy!"

### 5. Selesai!
Streamlit akan memberikan link permanen seperti:
`https://audit-lembur-namakamu.streamlit.app`

Share link ini ke tim — mereka tinggal buka di browser.

## Cara Pakai

1. Upload `Referensi_Lembur_v2.xlsx` di sidebar kiri
2. Isi nama auditor
3. Upload file data lembur yang akan diperiksa
4. Klik "Mulai Audit Sekarang"
5. Download hasil audit

## Kolom yang diperlukan di file data
- NIP
- Nama
- Tanggal
- Deskripsi
