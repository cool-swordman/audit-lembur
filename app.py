"""
Sistem Audit Lembur Pembangkit — v1.1.0
Streamlit Web App · Sentence Transformer AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import io

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Audit Lembur Pembangkit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================
MODEL_NAME       = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD_KUAT   = 0.80
THRESHOLD_SEDANG = 0.60
THRESHOLD_LEMAH  = 0.45

REQUIRED_COLUMNS = ['NIP', 'Nama', 'Tanggal', 'Deskripsi']
REF_SHEET_NAME   = 'Referensi'
REF_COL_DESC     = 'Deskripsi Kegiatan'
REF_COL_STATUS   = 'Status'
REF_COL_BIDANG   = 'Bidang'
REF_COL_JENIS    = 'Jenis Gangguan'

STATUS_COLOR = {
    'APPROVED':    '#22c55e',
    'REJECTED':    '#ef4444',
    'CONDITIONAL': '#f59e0b',
    'REVIEW':      '#6b7280',
}
STATUS_BG = {
    'APPROVED':    '#052e16',
    'REJECTED':    '#2d0707',
    'CONDITIONAL': '#2d1f07',
    'REVIEW':      '#1a1f2e',
}

RULES_APPROVE = [
    'forced outage', 'perbaikan darurat', 'gangguan mendadak', 'emergency',
    'trip', 'korektif', 'vibrasi tinggi', 'overheating', 'bocor', 'kebocoran',
    'pecah', 'kebakaran', 'recovery', 'restorasi', 'gagal start', 'troubleshooting',
    'blackout', 'grid collapse', 'supply darurat', 'forced derating',
    'tidak terjadwal', 'mendadak', 'darurat', 'insidental', 'unplanned',
    'perbaikan', 'gangguan', 'shutdown', 'turbin', 'boiler', 'rca', 'root cause',
    'derating', 'overhaul', 'tagging', 'vibrasi', 'relay', 'error', 'rusak',
]
RULES_REJECT = [
    'logsheet harian', 'log sheet', 'pencatatan harian', 'inspeksi rutin',
    'patroli rutin', 'piket', 'serah terima shift', 'koordinasi shift',
    'pelaporan harian', 'housekeeping', 'pengecekan kondisi normal',
    'monitoring rutin', 'kondisi normal', 'kondisi stabil', 'beban stabil',
]

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#F0FDF4;color:#e8e8e8}
section[data-testid="stSidebar"]{background:#161b27;border-right:1px solid #2a2f3e}
.brand{background:linear-gradient(135deg,#1a2332,#F0FDF4);border:1px solid #2a3f5f;border-radius:16px;padding:24px 28px;margin-bottom:24px;position:relative;overflow:hidden}
.brand::before{content:'';position:absolute;top:-40px;right:-40px;width:160px;height:160px;background:radial-gradient(circle,rgba(59,130,246,.12),transparent 70%);border-radius:50%}
.brand-title{font-size:24px;font-weight:600;color:#fff;margin:0 0 4px;letter-spacing:-.5px}
.brand-sub{font-size:12px;color:#6b7a99;font-family:'DM Mono',monospace;margin:0}
.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.mcard{background:#161b27;border:1px solid #2a2f3e;border-radius:12px;padding:16px 20px;position:relative;overflow:hidden}
.mcard::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:12px 12px 0 0}
.mcard.app::after{background:#22c55e}.mcard.rej::after{background:#ef4444}
.mcard.con::after{background:#f59e0b}.mcard.rev::after{background:#6b7280}
.mval{font-size:32px;font-weight:600;color:#fff;line-height:1;margin-bottom:4px}
.mlbl{font-size:11px;color:#6b7a99;font-weight:500;text-transform:uppercase;letter-spacing:.8px;font-family:'DM Mono',monospace}
.mpct{font-size:12px;color:#4b5563;margin-top:2px}
.info{background:#1a2332;border:1px solid #2a3f5f;border-left:3px solid #3b82f6;border-radius:8px;padding:10px 14px;margin:10px 0;font-size:13px;color:#93c5fd}
.warn{background:#1f1a0e;border:1px solid #78350f;border-left:3px solid #f59e0b;border-radius:8px;padding:10px 14px;font-size:12px;color:#fcd34d;margin:8px 0}
.err{background:#2d0707;border:1px solid #7f1d1d;border-left:3px solid #ef4444;border-radius:8px;padding:10px 14px;font-size:12px;color:#fca5a5;margin:8px 0}
.step{background:#161b27;border:1px solid #2a2f3e;border-radius:10px;padding:16px;margin-bottom:10px}
.snum{display:inline-block;width:26px;height:26px;background:#1e3a5f;border:1px solid #2a5f9e;border-radius:50%;text-align:center;line-height:26px;font-size:11px;font-weight:600;color:#60a5fa;font-family:'DM Mono',monospace;margin-right:8px}
.stitle{font-size:13px;font-weight:500;color:#e8e8e8}
.sdesc{font-size:12px;color:#6b7a99;margin-top:4px;margin-left:34px}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;font-family:'DM Mono',monospace;letter-spacing:.4px}
.b-app{background:#052e16;color:#22c55e;border:1px solid #166534}
.b-rej{background:#2d0707;color:#ef4444;border:1px solid #7f1d1d}
.b-con{background:#2d1f07;color:#f59e0b;border:1px solid #78350f}
.b-rev{background:#1a1f2e;color:#9ca3af;border:1px solid #374151}
.log-line{font-family:'DM Mono',monospace;font-size:11px;color:#6b7a99;padding:2px 0;border-bottom:1px solid #1f2535}
.section-title{font-size:16px;font-weight:600;color:#e8e8e8;margin:20px 0 12px}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.5rem;padding-bottom:2rem}
.stDownloadButton button{background:#1e3a5f!important;border:1px solid #2a5f9e!important;color:#60a5fa!important;border-radius:8px!important;font-family:'DM Sans',sans-serif!important;font-weight:500!important}
.stDownloadButton button:hover{background:#2a4f7f!important}
.stButton>button[kind="primary"]{background:#1e4d8c!important;border:1px solid #2a6abf!important;color:#ffffff!important;border-radius:8px!important;font-weight:600!important;font-size:15px!important;padding:12px!important}
.stButton>button[kind="primary"]:hover{background:#2558a8!important;transform:translateY(-1px);box-shadow:0 4px 20px rgba(59,130,246,.3)!important}
.stProgress .st-bo{background:#3b82f6!important}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#F0FDF4}
::-webkit-scrollbar-thumb{background:#2a2f3e;border-radius:3px}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INIT
# ============================================================
def init_state():
    defaults = {
        'df_ref':          None,
        'embeddings_ref':  None,
        'list_ref':        None,
        'df_hasil':        None,
        'audit_done':      False,
        'ref_loaded':      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ============================================================
# HELPERS — Model & Processing
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)


def validate_ref_file(df):
    required = [REF_COL_DESC, REF_COL_STATUS, REF_COL_BIDANG, REF_COL_JENIS]
    missing = [c for c in required if c not in df.columns]
    return missing


def validate_data_file(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def cek_keyword(desc_lower):
    found_approve = [w for w in RULES_APPROVE if w in desc_lower]
    found_reject  = [w for w in RULES_REJECT  if w in desc_lower]
    if found_approve and found_reject:
        return "CONDITIONAL", f"Konflik: '{found_approve[0]}' vs '{found_reject[0]}'"
    if found_reject:
        return "REJECTED",    f"Kegiatan rutin: '{found_reject[0]}'"
    if found_approve:
        return "APPROVED",    f"Kegiatan darurat: '{found_approve[0]}'"
    return None, None


def audit_satu(deskripsi, df_ref, model, embeddings_ref, list_ref):
    empty = ('REVIEW', '-', '-', '0.0%', '-', 'Data kosong atau tidak valid')
    if pd.isna(deskripsi) or str(deskripsi).strip() == '':
        return empty

    desc       = str(deskripsi).strip()
    desc_lower = desc.lower()

    try:
        emb_desc  = model.encode([desc])
        scores    = cosine_similarity(emb_desc, embeddings_ref)[0]
        best_idx  = int(np.argmax(scores))
        best_skor = round(float(scores[best_idx]) * 100, 1)
        best_match = list_ref[best_idx]
        row        = df_ref.iloc[best_idx]
        ref_status = row[REF_COL_STATUS]
        ref_bidang = row[REF_COL_BIDANG]
        ref_jenis  = row[REF_COL_JENIS]
    except Exception as e:
        return ('REVIEW', '-', '-', '-', '-', f'Error encoding: {str(e)[:60]}')

    skor_str = f"{best_skor}%"

    if best_skor >= THRESHOLD_KUAT * 100:
        return (ref_status, ref_bidang, ref_jenis, skor_str, 'Referensi Kuat',
                f"[{ref_bidang} – {ref_jenis}] Kecocokan makna sangat kuat ({best_skor}%): '{best_match[:65]}'")

    if best_skor >= THRESHOLD_SEDANG * 100:
        return (ref_status, ref_bidang, ref_jenis, skor_str, 'Referensi Sedang',
                f"[{ref_bidang} – {ref_jenis}] Kecocokan makna sedang ({best_skor}%): '{best_match[:65]}'")

    if best_skor >= THRESHOLD_LEMAH * 100:
        kw_status, kw_alasan = cek_keyword(desc_lower)
        if kw_status:
            return (kw_status, '-', '-', skor_str, 'Keyword',
                    f"{kw_alasan}. Referensi terdekat ({best_skor}%): '{best_match[:50]}'")
        return ('CONDITIONAL', ref_bidang, ref_jenis, skor_str, 'Referensi Lemah',
                f"Kecocokan rendah ({best_skor}%). Referensi: '{best_match[:55]}'. Verifikasi manual diperlukan.")

    kw_status, kw_alasan = cek_keyword(desc_lower)
    if kw_status:
        return (kw_status, '-', '-', skor_str, 'Keyword',
                f"{kw_alasan} (similarity sangat rendah: {best_skor}%)")
    return ('REVIEW', '-', '-', skor_str, 'Tidak Cocok',
            f"Deskripsi tidak dikenali (skor maks {best_skor}%). Butuh verifikasi manual.")


def buat_chart_distribusi(df_hasil):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#161b27')
    counts = df_hasil['AI Status'].value_counts()
    order  = [s for s in ['APPROVED','REJECTED','CONDITIONAL','REVIEW'] if s in counts.index]
    values = [counts[s] for s in order]
    colors = [STATUS_COLOR.get(s,'#6b7280') for s in order]

    ax1.set_facecolor('#161b27')
    bars = ax1.bar(order, values, color=colors, width=0.5, zorder=3)
    for sp in ['top','right']: ax1.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax1.spines[sp].set_color('#2a2f3e')
    ax1.tick_params(colors='#6b7a99', labelsize=9)
    ax1.yaxis.grid(True, color='#1f2535', zorder=0)
    ax1.set_title('Distribusi Keputusan', color='#e8e8e8', fontsize=12, pad=10)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.15,
                 str(val), ha='center', va='bottom', color='#e8e8e8', fontsize=11, fontweight='600')

    ax2.set_facecolor('#161b27')
    wedges, texts, autotexts = ax2.pie(
        values, labels=order, colors=colors, autopct='%1.0f%%',
        startangle=90, wedgeprops={'linewidth':0,'edgecolor':'#161b27'}, pctdistance=0.75
    )
    for t in texts:    t.set_color('#6b7a99'); t.set_fontsize(9)
    for at in autotexts: at.set_color('#fff'); at.set_fontsize(10); at.set_fontweight('600')
    ax2.set_title('Proporsi', color='#e8e8e8', fontsize=12, pad=10)

    plt.tight_layout()
    return fig


def export_excel(df_hasil):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_hasil.to_excel(writer, sheet_name='Hasil Audit', index=False)
        df_hasil[df_hasil['AI Status'].isin(['REVIEW','CONDITIONAL'])].to_excel(
            writer, sheet_name='Perlu Review', index=False)
        df_hasil[df_hasil['AI Status']=='APPROVED'].to_excel(
            writer, sheet_name='Approved', index=False)
        df_hasil[df_hasil['AI Status']=='REJECTED'].to_excel(
            writer, sheet_name='Rejected', index=False)
    buf.seek(0)
    return buf


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:6px 0 16px">
      <div style="font-size:26px">⚡</div>
      <div style="font-size:15px;font-weight:600;color:#fff;margin-top:6px">Audit Lembur</div>
      <div style="font-size:10px;color:#6b7a99;font-family:'DM Mono',monospace">Sistem Evaluasi Otomatis v1.1</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Nama auditor
    nama_auditor = st.text_input("👤 Nama Auditor", placeholder="contoh: Budi Santoso",
                                  help="Akan tercatat di file hasil audit")

    st.divider()

    # Upload referensi
    st.markdown("**📚 File Referensi**")
    ref_file = st.file_uploader("Referensi_Lembur_v2.xlsx", type=['xlsx'],
                                  key='ref_upload',
                                  help="Upload sekali, dipakai untuk semua sesi")

    if ref_file:
        if not st.session_state.ref_loaded:
            with st.spinner("Memuat referensi..."):
                try:
                    df_ref = pd.read_excel(ref_file, sheet_name=REF_SHEET_NAME)
                    missing = validate_ref_file(df_ref)
                    if missing:
                        st.error(f"Kolom tidak ditemukan: {missing}")
                    else:
                        model_st = load_model()
                        list_ref = df_ref[REF_COL_DESC].astype(str).tolist()
                        emb_ref  = model_st.encode(list_ref, show_progress_bar=False)
                        st.session_state.df_ref         = df_ref
                        st.session_state.list_ref       = list_ref
                        st.session_state.embeddings_ref = emb_ref
                        st.session_state.ref_loaded     = True
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.ref_loaded:
            df_r = st.session_state.df_ref
            st.markdown(f"""
            <div style="background:#052e16;border:1px solid #166534;border-radius:8px;padding:10px 12px;font-size:12px;color:#22c55e;margin-top:8px">
                ✅ {len(df_r)} referensi aktif<br>
                <span style="color:#4b5563">{df_r[REF_COL_BIDANG].nunique()} bidang · {df_r[REF_COL_JENIS].nunique()} jenis</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Ganti referensi", use_container_width=True):
                st.session_state.ref_loaded = False
                st.rerun()

    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:#4b5563;line-height:1.8">
      <div style="color:#6b7a99;font-weight:500;margin-bottom:4px">Threshold similarity</div>
      <div>🟢 Kuat &nbsp;&nbsp; ≥ 80%</div>
      <div>🟡 Sedang &nbsp;60–79%</div>
      <div>🟠 Lemah &nbsp;&nbsp;45–59%</div>
      <div>⚪ Rendah &nbsp;&lt; 45%</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div class="brand">
  <div style="font-size:28px;margin-bottom:10px">⚡</div>
  <div class="brand-title">Sistem Audit Lembur Pembangkit</div>
  <div class="brand-sub">AI · Sentence Transformer · paraphrase-multilingual-MiniLM-L12-v2</div>
</div>
""", unsafe_allow_html=True)


# Belum ada referensi — tampilkan panduan
if not st.session_state.ref_loaded:
    col1, col2, col3 = st.columns(3)
    for col, num, title, desc in [
        (col1, "1", "Upload referensi", "Upload Referensi_Lembur_v2.xlsx di sidebar kiri"),
        (col2, "2", "Upload & audit data", "Upload file lembur lalu klik Mulai Audit"),
        (col3, "3", "Download hasil", "Unduh Excel hasil audit 4 sheet"),
    ]:
        with col:
            st.markdown(f"""
            <div class="step">
              <span class="snum">{num}</span><span class="stitle">{title}</span>
              <div class="sdesc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info">
      📋 Mulai dengan upload <strong>Referensi_Lembur_v2.xlsx</strong> di sidebar kiri.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ============================================================
# SECTION 1: UPLOAD DATA
# ============================================================
st.markdown('<div class="section-title">📂 Upload Data Lembur</div>', unsafe_allow_html=True)

data_file = st.file_uploader("File Excel data lembur yang akan diaudit",
                               type=['xlsx'], key='data_upload',
                               help="Kolom wajib: NIP, Nama, Tanggal, Deskripsi")

if not data_file:
    st.markdown("""
    <div class="warn">
      ⚠️ Kolom wajib di file Excel: <strong>NIP · Nama · Tanggal · Deskripsi</strong>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    df_baru = pd.read_excel(data_file)
    missing_cols = validate_data_file(df_baru)
    if missing_cols:
        st.markdown(f"""
        <div class="err">
          ❌ Kolom tidak ditemukan: <strong>{', '.join(missing_cols)}</strong><br>
          Pastikan nama kolom persis: NIP, Nama, Tanggal, Deskripsi
        </div>
        """, unsafe_allow_html=True)
        st.stop()
except Exception as e:
    st.markdown(f'<div class="err">❌ Gagal membaca file: {e}</div>', unsafe_allow_html=True)
    st.stop()

col_info1, col_info2 = st.columns([3, 1])
with col_info1:
    st.markdown(f"""
    <div class="info">
      ✅ <strong>{len(df_baru)} baris</strong> siap diaudit dari file <em>{data_file.name}</em>
    </div>
    """, unsafe_allow_html=True)
with col_info2:
    kosong = df_baru['Deskripsi'].isna().sum()
    if kosong > 0:
        st.markdown(f'<div class="warn">⚠️ {kosong} baris kosong</div>', unsafe_allow_html=True)

with st.expander("👁 Preview data (5 baris pertama)"):
    st.dataframe(df_baru.head(), use_container_width=True)


# ============================================================
# SECTION 2: VALIDASI & TOMBOL AUDIT
# ============================================================
st.divider()

if not nama_auditor.strip():
    st.markdown('<div class="warn">⚠️ Isi <strong>Nama Auditor</strong> di sidebar kiri sebelum memulai.</div>',
                unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    mulai = st.button("🚀  Mulai Audit Sekarang",
                      disabled=not nama_auditor.strip(),
                      use_container_width=True, type="primary")


# ============================================================
# SECTION 3: PROSES AUDIT
# ============================================================
if mulai:
    model_st     = load_model()
    df_ref       = st.session_state.df_ref
    embeddings   = st.session_state.embeddings_ref
    list_ref     = st.session_state.list_ref
    total        = len(df_baru)
    hasil_rows   = []

    st.divider()
    st.markdown('<div class="section-title">⏳ Memproses Audit...</div>', unsafe_allow_html=True)

    progress  = st.progress(0)
    log_area  = st.empty()
    log_lines = []
    t_start   = datetime.now()

    for i, (_, row) in enumerate(df_baru.iterrows()):
        desc     = row.get('Deskripsi', '')
        nama_p   = row.get('Nama', '-')
        tanggal  = row.get('Tanggal', '-')

        status, bidang, jenis, skor, sumber, alasan = audit_satu(
            desc, df_ref, model_st, embeddings, list_ref
        )

        hasil_rows.append({
            'Diperiksa Oleh'   : nama_auditor,
            'Tanggal Audit'    : t_start.strftime('%Y-%m-%d %H:%M'),
            'NIP'              : row.get('NIP', '-'),
            'Nama'             : nama_p,
            'Tanggal Lembur'   : tanggal,
            'Deskripsi'        : desc,
            'AI Status'        : status,
            'Bidang Referensi' : bidang,
            'Jenis Gangguan'   : jenis,
            'Skor Similarity'  : skor,
            'Sumber Keputusan' : sumber,
            'Alasan Audit'     : alasan,
        })

        pct = int((i + 1) / total * 100)
        progress.progress(pct)

        color = STATUS_COLOR.get(status, '#6b7280')
        log_lines.append(
            f'<div class="log-line">[{i+1:03d}/{total}] '
            f'<span style="color:#e8e8e8">{str(desc)[:52]:<52}</span> '
            f'<span style="color:{color};font-weight:600">{status}</span> '
            f'<span style="color:#4b5563">({skor})</span></div>'
        )
        if len(log_lines) > 8:
            log_lines = log_lines[-8:]
        log_area.markdown(
            f'<div style="background:#0a0d14;border:1px solid #1f2535;border-radius:8px;padding:10px 14px">{"".join(log_lines)}</div>',
            unsafe_allow_html=True
        )

    t_end  = datetime.now()
    durasi = round((t_end - t_start).total_seconds(), 1)

    progress.empty()
    log_area.empty()

    st.session_state.df_hasil   = pd.DataFrame(hasil_rows)
    st.session_state.audit_done = True

    st.markdown(f"""
    <div style="background:#052e16;border:1px solid #166534;border-radius:10px;padding:14px 18px;margin:12px 0">
      ✅ <strong>Audit selesai!</strong> &nbsp;
      {total} data diproses dalam <strong>{durasi} detik</strong> &nbsp;·&nbsp;
      Auditor: <strong>{nama_auditor}</strong> &nbsp;·&nbsp;
      {t_end.strftime('%d %b %Y, %H:%M')}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SECTION 4: TAMPILKAN HASIL
# ============================================================
if st.session_state.audit_done and st.session_state.df_hasil is not None:
    df_hasil = st.session_state.df_hasil
    total    = len(df_hasil)
    nama_aud = df_hasil['Diperiksa Oleh'].iloc[0]

    c_app = (df_hasil['AI Status'] == 'APPROVED').sum()
    c_rej = (df_hasil['AI Status'] == 'REJECTED').sum()
    c_con = (df_hasil['AI Status'] == 'CONDITIONAL').sum()
    c_rev = (df_hasil['AI Status'] == 'REVIEW').sum()

    st.divider()
    st.markdown('<div class="section-title">📊 Ringkasan Hasil Audit</div>', unsafe_allow_html=True)

    # Metric cards
    st.markdown(f"""
    <div class="metric-row">
      <div class="mcard app">
        <div class="mval">{c_app}</div>
        <div class="mlbl">Approved</div>
        <div class="mpct">{c_app/total*100:.1f}% dari total</div>
      </div>
      <div class="mcard rej">
        <div class="mval">{c_rej}</div>
        <div class="mlbl">Rejected</div>
        <div class="mpct">{c_rej/total*100:.1f}% dari total</div>
      </div>
      <div class="mcard con">
        <div class="mval">{c_con}</div>
        <div class="mlbl">Conditional</div>
        <div class="mpct">{c_con/total*100:.1f}% dari total</div>
      </div>
      <div class="mcard rev">
        <div class="mval">{c_rev}</div>
        <div class="mlbl">Perlu Review</div>
        <div class="mpct">{c_rev/total*100:.1f}% dari total</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    fig = buat_chart_distribusi(df_hasil)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Alert jika banyak REVIEW
    if c_rev + c_con > total * 0.3:
        st.markdown(f"""
        <div class="warn">
          ⚠️ <strong>{c_rev + c_con} kasus</strong> ({(c_rev+c_con)/total*100:.0f}%) memerlukan verifikasi manual.
          Pertimbangkan untuk memperkaya database referensi dengan contoh kegiatan yang sering muncul.
        </div>
        """, unsafe_allow_html=True)

    # Tabel detail
    st.divider()
    st.markdown('<div class="section-title">📋 Detail Hasil</div>', unsafe_allow_html=True)

    tab_all, tab_rev, tab_app, tab_rej = st.tabs([
        f"🗂 Semua ({total})",
        f"⚠️ Perlu Review ({c_rev + c_con})",
        f"✅ Approved ({c_app})",
        f"❌ Rejected ({c_rej})"
    ])
    with tab_all:
        st.dataframe(df_hasil, use_container_width=True, height=380)
    with tab_rev:
        df_rv = df_hasil[df_hasil['AI Status'].isin(['REVIEW','CONDITIONAL'])]
        if len(df_rv):
            st.dataframe(df_rv, use_container_width=True, height=350)
        else:
            st.success("Tidak ada kasus yang perlu review manual!")
    with tab_app:
        st.dataframe(df_hasil[df_hasil['AI Status']=='APPROVED'],
                     use_container_width=True, height=350)
    with tab_rej:
        st.dataframe(df_hasil[df_hasil['AI Status']=='REJECTED'],
                     use_container_width=True, height=350)

    # Distribusi sumber keputusan
    with st.expander("🔍 Distribusi sumber keputusan"):
        sumber_counts = df_hasil['Sumber Keputusan'].value_counts()
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.dataframe(sumber_counts.rename("Jumlah").reset_index(), use_container_width=True)
        with col_s2:
            st.markdown("""
            <div style="font-size:12px;color:#6b7a99;line-height:2">
              <strong style="color:#e8e8e8">Keterangan sumber:</strong><br>
              🟢 <strong>Referensi Kuat</strong> — similarity ≥ 80%, sangat dipercaya<br>
              🟡 <strong>Referensi Sedang</strong> — similarity 60–79%, cukup dipercaya<br>
              🟠 <strong>Referensi Lemah</strong> — similarity 45–59%, perlu dicermati<br>
              🔤 <strong>Keyword</strong> — terdeteksi dari kata kunci<br>
              ⚪ <strong>Tidak Cocok</strong> — similarity terlalu rendah, perlu review
            </div>
            """, unsafe_allow_html=True)

    # Download
    st.divider()
    st.markdown('<div class="section-title">📥 Download Hasil</div>', unsafe_allow_html=True)

    nama_file_out = f"Hasil_Audit_{nama_aud.replace(' ','_')}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    buf = export_excel(df_hasil)

    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        st.download_button(
            label=f"📥  Download {nama_file_out}",
            data=buf,
            file_name=nama_file_out,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )

    st.markdown(f"""
    <div class="info" style="margin-top:10px;text-align:center;font-size:12px">
      📄 4 sheet: <strong>Hasil Audit · Perlu Review · Approved · Rejected</strong><br>
      👤 Auditor: <strong>{nama_aud}</strong> &nbsp;·&nbsp;
      📅 <strong>{datetime.now().strftime('%d %B %Y, %H:%M')}</strong>
    </div>
    """, unsafe_allow_html=True)

    # Tombol audit ulang
    st.divider()
    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        if st.button("🔁  Audit File Lain", use_container_width=True):
            st.session_state.df_hasil   = None
            st.session_state.audit_done = False
            st.rerun()
