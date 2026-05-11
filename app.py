import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from datetime import datetime
import io
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Audit Lembur Pembangkit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}

.stApp{background:#F0F4F8}

section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid #E2E8F0}

.brand{background:linear-gradient(135deg,#1E3A5F,#2a5298);border-radius:16px;padding:24px 28px;margin-bottom:24px;position:relative;overflow:hidden}
.brand::before{content:'';position:absolute;top:-40px;right:-40px;width:160px;height:160px;background:radial-gradient(circle,rgba(255,255,255,.15),transparent 70%);border-radius:50%}
.brand-title{font-size:24px;font-weight:600;color:#FFFFFF;margin:0 0 4px;letter-spacing:-.5px}
.brand-sub{font-size:12px;color:rgba(255,255,255,.7);font-family:'DM Mono',monospace;margin:0}

.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.mcard{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:16px 20px;position:relative;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.mcard::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0}
.mcard.app::after{background:#22c55e}.mcard.rej::after{background:#ef4444}
.mcard.con::after{background:#f59e0b}.mcard.rev::after{background:#94a3b8}
.mval{font-size:32px;font-weight:700;color:#1a202c;line-height:1;margin-bottom:4px}
.mlbl{font-size:11px;color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.8px;font-family:'DM Mono',monospace}
.mpct{font-size:12px;color:#94a3b8;margin-top:2px}

.info{background:#EFF6FF;border:1px solid #BFDBFE;border-left:3px solid #3b82f6;border-radius:8px;padding:10px 14px;margin:10px 0;font-size:13px;color:#1e40af}
.warn{background:#FFFBEB;border:1px solid #FDE68A;border-left:3px solid #f59e0b;border-radius:8px;padding:10px 14px;font-size:12px;color:#92400e;margin:8px 0}
.err{background:#FEF2F2;border:1px solid #FECACA;border-left:3px solid #ef4444;border-radius:8px;padding:10px 14px;font-size:12px;color:#991b1b;margin:8px 0}

.step{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:16px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.snum{display:inline-block;width:26px;height:26px;background:#1E3A5F;border-radius:50%;text-align:center;line-height:26px;font-size:11px;font-weight:600;color:#FFFFFF;font-family:'DM Mono',monospace;margin-right:8px}
.stitle{font-size:13px;font-weight:600;color:#1a202c}
.sdesc{font-size:12px;color:#64748b;margin-top:4px;margin-left:34px}

.log-line{font-family:'DM Mono',monospace;font-size:11px;color:#475569;padding:2px 0;border-bottom:1px solid #F1F5F9}

.section-title{font-size:16px;font-weight:600;color:#1a202c;margin:20px 0 12px}

#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.5rem;padding-bottom:2rem}

.stDownloadButton button{background:#1E3A5F!important;border:1px solid #2a5298!important;color:#FFFFFF!important;border-radius:8px!important;font-family:'DM Sans',sans-serif!important;font-weight:500!important}
.stDownloadButton button:hover{background:#2a5298!important}
.stButton>button[kind="primary"]{background:#1E3A5F!important;border:none!important;color:#FFFFFF!important;border-radius:8px!important;font-weight:600!important;font-size:15px!important;padding:12px!important}
.stButton>button[kind="primary"]:hover{background:#2a5298!important;box-shadow:0 4px 20px rgba(30,58,95,.3)!important}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS & THRESHOLDS
# ============================================================
THRESHOLD_KUAT   = 0.80
THRESHOLD_SEDANG = 0.60
THRESHOLD_LEMAH  = 0.45

RULES_APPROVE = [
    'forced outage', 'perbaikan darurat', 'gangguan mendadak', 'emergency',
    'trip', 'korektif', 'vibrasi tinggi', 'overheating', 'bocor', 'kebocoran',
    'pecah', 'kebakaran', 'recovery', 'restorasi', 'tagging darurat',
    'shutdown darurat', 'gagal start', 'troubleshooting', 'blackout',
    'grid collapse', 'supply darurat', 'forced derating', 'tidak terjadwal',
    'mendadak', 'darurat', 'insidental', 'unplanned', 'perbaikan', 'gangguan',
    'shutdown', 'turbin', 'boiler', 'rca', 'root cause', 'derating',
    'overhaul', 'tagging', 'vibrasi', 'relay', 'error', 'rusak',
]
RULES_REJECT = [
    'logsheet harian', 'log sheet', 'pencatatan harian', 'inspeksi rutin',
    'patroli rutin', 'piket', 'serah terima shift', 'koordinasi shift',
    'pelaporan harian', 'housekeeping', 'pengecekan kondisi normal',
    'monitoring rutin', 'kondisi normal', 'kondisi stabil', 'beban stabil',
]

STATUS_COLOR = {
    'APPROVED':    '#22c55e',
    'REJECTED':    '#ef4444',
    'CONDITIONAL': '#f59e0b',
    'REVIEW':      '#6b7280',
}


# ============================================================
# LOAD MODEL (cached agar tidak reload tiap interaksi)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


@st.cache_data(show_spinner=False)
def encode_referensi(_model, list_ref):
    return _model.encode(list_ref)


# ============================================================
# AUDIT FUNCTIONS
# ============================================================
def cek_keyword(desc_lower):
    found_approve = [w for w in RULES_APPROVE if w in desc_lower]
    found_reject  = [w for w in RULES_REJECT  if w in desc_lower]
    if found_approve and found_reject:
        return "CONDITIONAL", f"Konflik keyword: '{found_approve[0]}' vs '{found_reject[0]}'"
    if found_reject:
        return "REJECTED",    f"Kegiatan rutin terdeteksi: '{found_reject[0]}'"
    if found_approve:
        return "APPROVED",    f"Kegiatan darurat/insidental: '{found_approve[0]}'"
    return None, None


def audit_satu(deskripsi, df_ref, model, embeddings_ref):
    if pd.isna(deskripsi) or str(deskripsi).strip() == '':
        return 'REVIEW', '-', '-', '-', '-', 'Data kosong'

    desc       = str(deskripsi).strip()
    desc_lower = desc.lower()
    list_ref   = df_ref['Deskripsi Kegiatan'].astype(str).tolist()

    emb_desc = model.encode([desc])
    scores   = cosine_similarity(emb_desc, embeddings_ref)[0]
    best_idx = int(np.argmax(scores))
    best_skor = round(float(scores[best_idx]) * 100, 1)
    best_match = list_ref[best_idx]
    matched_row = df_ref.iloc[best_idx]

    ref_status = matched_row['Status']
    ref_bidang = matched_row['Bidang']
    ref_jenis  = matched_row['Jenis Gangguan']

    if best_skor >= THRESHOLD_KUAT * 100:
        return (ref_status, ref_bidang, ref_jenis,
                f"{best_skor}%", f"ST Kuat",
                f"[{ref_bidang} – {ref_jenis}] Makna sangat mirip ({best_skor}%): '{best_match[:65]}'")

    elif best_skor >= THRESHOLD_SEDANG * 100:
        return (ref_status, ref_bidang, ref_jenis,
                f"{best_skor}%", f"ST Sedang",
                f"[{ref_bidang} – {ref_jenis}] Makna cukup mirip ({best_skor}%): '{best_match[:65]}'")

    elif best_skor >= THRESHOLD_LEMAH * 100:
        kw_status, kw_alasan = cek_keyword(desc_lower)
        if kw_status:
            return (kw_status, '-', '-', f"{best_skor}%", "Keyword",
                    f"{kw_alasan}. Ref terdekat ({best_skor}%): '{best_match[:50]}'")
        return ('CONDITIONAL', ref_bidang, ref_jenis, f"{best_skor}%", "ST Lemah",
                f"Similarity rendah ({best_skor}%), butuh verifikasi manual. Ref: '{best_match[:55]}'")

    else:
        kw_status, kw_alasan = cek_keyword(desc_lower)
        if kw_status:
            return (kw_status, '-', '-', f"{best_skor}%", "Keyword",
                    f"{kw_alasan} (similarity terlalu rendah: {best_skor}%)")
        return ('REVIEW', '-', '-', f"{best_skor}%", "Tidak cocok",
                f"Deskripsi tidak dikenali (maks {best_skor}%). Butuh verifikasi manual.")


# ============================================================
# CHART FUNCTIONS
# ============================================================
def buat_chart(df_hasil):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#161b27')

    counts = df_hasil['AI Status'].value_counts()
    order  = ['APPROVED', 'REJECTED', 'CONDITIONAL', 'REVIEW']
    labels = [s for s in order if s in counts.index]
    values = [counts[s] for s in labels]
    colors = [STATUS_COLOR.get(s, '#6b7280') for s in labels]

    # Bar chart
    ax1.set_facecolor('#161b27')
    bars = ax1.bar(labels, values, color=colors, width=0.55, zorder=3)
    ax1.set_facecolor('#161b27')
    ax1.spines['bottom'].set_color('#2a2f3e')
    ax1.spines['left'].set_color('#2a2f3e')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(colors='#6b7a99', labelsize=10)
    ax1.set_title('Distribusi Keputusan', color='#e8e8e8', fontsize=12, pad=12)
    ax1.yaxis.grid(True, color='#1f2535', zorder=0)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha='center', va='bottom', color='#e8e8e8', fontsize=11, fontweight='600')

    # Pie chart
    ax2.set_facecolor('#161b27')
    wedges, texts, autotexts = ax2.pie(
        values, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=90,
        wedgeprops={'linewidth': 0, 'edgecolor': '#161b27'},
        pctdistance=0.75
    )
    for t in texts:
        t.set_color('#6b7a99')
        t.set_fontsize(10)
    for at in autotexts:
        at.set_color('#ffffff')
        at.set_fontsize(10)
        at.set_fontweight('600')
    ax2.set_title('Proporsi Keputusan', color='#e8e8e8', fontsize=12, pad=12)

    plt.tight_layout()
    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0">
        <span style="font-size:28px">⚡</span>
        <div style="font-size:16px;font-weight:600;color:#ffffff;margin-top:8px">Audit Lembur</div>
        <div style="font-size:11px;color:#6b7a99;font-family:'DM Mono',monospace">Sistem Evaluasi Otomatis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Pengaturan**")

    nama_auditor = st.text_input(
        "Nama Auditor",
        placeholder="contoh: Budi Santoso",
        help="Nama ini akan muncul di file hasil audit"
    )

    st.markdown("---")
    st.markdown("**📁 File Referensi**")

    ref_file = st.file_uploader(
        "Upload Referensi_Lembur_v2.xlsx",
        type=['xlsx'],
        help="File referensi berisi 560 contoh kegiatan yang sudah dikategorikan"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#4b5563;line-height:1.6">
        <div style="color:#6b7a99;font-weight:500;margin-bottom:6px">Threshold similarity</div>
        <div>🟢 Kuat: ≥ 80%</div>
        <div>🟡 Sedang: 60–79%</div>
        <div>🟠 Lemah: 45–59%</div>
        <div>⚪ Tidak cocok: &lt; 45%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#4b5563">
        v1.0 · Sentence Transformer<br>
        paraphrase-multilingual-MiniLM
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown("""
<div class="brand-header">
    <span class="brand-icon">⚡</span>
    <div class="brand-title">Sistem Audit Lembur Pembangkit</div>
    <div class="brand-subtitle">AI-powered · Sentence Transformer · 560 referensi kegiatan</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STEP 1: Cek referensi
# ============================================================
if not ref_file:
    st.markdown("""
    <div class="info-box">
        📋 Upload file <strong>Referensi_Lembur_v2.xlsx</strong> di sidebar kiri untuk memulai.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="step-box">
            <span class="step-number">1</span><span class="step-title">Upload referensi</span>
            <div class="step-desc">Upload file referensi di sidebar kiri</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-box">
            <span class="step-number">2</span><span class="step-title">Upload data lembur</span>
            <div class="step-desc">Upload file Excel data yang akan diaudit</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-box">
            <span class="step-number">3</span><span class="step-title">Download hasil</span>
            <div class="step-desc">Unduh file Excel hasil audit otomatis</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ============================================================
# LOAD REFERENSI
# ============================================================
try:
    df_ref = pd.read_excel(ref_file, sheet_name='Referensi')
    st.markdown(f"""
    <div class="info-box">
        ✅ Referensi berhasil dimuat — <strong>{len(df_ref)} entri</strong> dari {df_ref['Bidang'].nunique()} bidang
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Gagal membaca file referensi: {e}")
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================
with st.spinner("⏳ Memuat model AI... (hanya sekali, ~30 detik)"):
    model_st = load_model()

list_ref_cache = df_ref['Deskripsi Kegiatan'].astype(str).tolist()
with st.spinner("⏳ Memproses referensi..."):
    emb_ref = encode_referensi(model_st, tuple(list_ref_cache))


# ============================================================
# UPLOAD DATA BARU
# ============================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("### 📂 Upload Data Lembur")

data_file = st.file_uploader(
    "Pilih file Excel data lembur yang akan diaudit",
    type=['xlsx'],
    help="File harus memiliki kolom: NIP, Nama, Tanggal, Deskripsi"
)

if not data_file:
    st.markdown("""
    <div class="warn-box">
        ⚠️ Pastikan file Excel memiliki kolom: <strong>NIP, Nama, Tanggal, Deskripsi</strong>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ============================================================
# PREVIEW DATA
# ============================================================
try:
    df_baru = pd.read_excel(data_file)
    st.markdown(f"""
    <div class="info-box">
        ✅ Data berhasil dimuat — <strong>{len(df_baru)} baris</strong> siap diaudit
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👁 Preview data (5 baris pertama)"):
        st.dataframe(df_baru.head(), use_container_width=True)
except Exception as e:
    st.error(f"❌ Gagal membaca file data: {e}")
    st.stop()


# ============================================================
# TOMBOL AUDIT
# ============================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

if not nama_auditor.strip():
    st.markdown("""
    <div class="warn-box">
        ⚠️ Isi <strong>Nama Auditor</strong> di sidebar kiri sebelum memulai audit.
    </div>
    """, unsafe_allow_html=True)
    btn_disabled = True
else:
    btn_disabled = False

mulai = st.button(
    "🚀 Mulai Audit Sekarang",
    disabled=btn_disabled,
    use_container_width=True,
    type="primary"
)

if mulai:
    st.markdown("---")
    st.markdown("### ⏳ Memproses...")

    progress  = st.progress(0)
    status_tx = st.empty()
    total     = len(df_baru)
    hasil_rows = []

    for i, (_, row) in enumerate(df_baru.iterrows()):
        deskripsi = row.get('Deskripsi', '')
        nama      = row.get('Nama', '-')
        tanggal   = row.get('Tanggal', '-')

        status, bidang, jenis, skor, sumber, alasan = audit_satu(
            deskripsi, df_ref, model_st, emb_ref
        )

        hasil_rows.append({
            'Diperiksa Oleh'   : nama_auditor,
            'Tanggal Audit'    : datetime.now().strftime('%Y-%m-%d %H:%M'),
            'NIP'              : row.get('NIP', '-'),
            'Nama'             : nama,
            'Tanggal Lembur'   : tanggal,
            'Deskripsi'        : deskripsi,
            'AI Status'        : status,
            'Bidang Referensi' : bidang,
            'Jenis Gangguan'   : jenis,
            'Skor Similarity'  : skor,
            'Sumber Keputusan' : sumber,
            'Alasan Audit'     : alasan,
        })

        pct = int((i + 1) / total * 100)
        progress.progress(pct)
        status_tx.markdown(f"""
        <div style="font-size:12px;color:#6b7a99;font-family:'DM Mono',monospace">
            [{i+1}/{total}] {str(deskripsi)[:55]} → <span style="color:{STATUS_COLOR.get(status,'#6b7280')};font-weight:600">{status}</span>
        </div>
        """, unsafe_allow_html=True)

    df_hasil = pd.DataFrame(hasil_rows)
    status_tx.empty()
    progress.empty()
    st.success(f"✅ Audit selesai! {total} data berhasil diproses.")

    # ============================================================
    # HASIL — METRIC CARDS
    # ============================================================
    st.markdown("---")
    st.markdown("### 📊 Hasil Audit")

    c_app = (df_hasil['AI Status'] == 'APPROVED').sum()
    c_rej = (df_hasil['AI Status'] == 'REJECTED').sum()
    c_con = (df_hasil['AI Status'] == 'CONDITIONAL').sum()
    c_rev = (df_hasil['AI Status'] == 'REVIEW').sum()

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card approved">
            <div class="metric-value">{c_app}</div>
            <div class="metric-label">Approved</div>
            <div class="metric-pct">{c_app/total*100:.1f}% dari total</div>
        </div>
        <div class="metric-card rejected">
            <div class="metric-value">{c_rej}</div>
            <div class="metric-label">Rejected</div>
            <div class="metric-pct">{c_rej/total*100:.1f}% dari total</div>
        </div>
        <div class="metric-card conditional">
            <div class="metric-value">{c_con}</div>
            <div class="metric-label">Conditional</div>
            <div class="metric-pct">{c_con/total*100:.1f}% dari total</div>
        </div>
        <div class="metric-card review">
            <div class="metric-value">{c_rev}</div>
            <div class="metric-label">Perlu Review</div>
            <div class="metric-pct">{c_rev/total*100:.1f}% dari total</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # CHART
    # ============================================================
    fig = buat_chart(df_hasil)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ============================================================
    # TABEL HASIL
    # ============================================================
    st.markdown("---")
    st.markdown("### 📋 Detail Hasil")

    tab1, tab2, tab3, tab4 = st.tabs(["Semua", "⚠️ Perlu Review", "✅ Approved", "❌ Rejected"])

    with tab1:
        st.dataframe(df_hasil, use_container_width=True, height=400)
    with tab2:
        df_rev = df_hasil[df_hasil['AI Status'].isin(['REVIEW', 'CONDITIONAL'])]
        st.markdown(f"**{len(df_rev)} baris** memerlukan verifikasi manual")
        st.dataframe(df_rev, use_container_width=True, height=350)
    with tab3:
        st.dataframe(df_hasil[df_hasil['AI Status'] == 'APPROVED'],
                     use_container_width=True, height=350)
    with tab4:
        st.dataframe(df_hasil[df_hasil['AI Status'] == 'REJECTED'],
                     use_container_width=True, height=350)

    # ============================================================
    # DOWNLOAD
    # ============================================================
    st.markdown("---")
    st.markdown("### 📥 Download Hasil")

    nama_file_out = (
        f"Hasil_Audit_{nama_auditor.replace(' ','_')}_"
        f"{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_hasil.to_excel(writer, sheet_name='Hasil Audit', index=False)
        df_hasil[df_hasil['AI Status'].isin(['REVIEW','CONDITIONAL'])].to_excel(
            writer, sheet_name='Perlu Review', index=False)
        df_hasil[df_hasil['AI Status'] == 'APPROVED'].to_excel(
            writer, sheet_name='Approved', index=False)
        df_hasil[df_hasil['AI Status'] == 'REJECTED'].to_excel(
            writer, sheet_name='Rejected', index=False)
    buf.seek(0)

    st.download_button(
        label=f"📥 Download {nama_file_out}",
        data=buf,
        file_name=nama_file_out,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True
    )

    st.markdown(f"""
    <div class="info-box" style="margin-top:12px">
        📌 File hasil berisi 4 sheet: <strong>Hasil Audit · Perlu Review · Approved · Rejected</strong><br>
        👤 Auditor: <strong>{nama_auditor}</strong> · 
        📅 {datetime.now().strftime('%d %B %Y, %H:%M')}
    </div>
    """, unsafe_allow_html=True)
