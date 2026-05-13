"""
Sistem Audit Lembur Pembangkit — v1.2.0
Streamlit Web App · Sentence Transformer AI · Light Theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import io

st.set_page_config(
    page_title="Audit Lembur Pembangkit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    'APPROVED':    '#16a34a',
    'REJECTED':    '#dc2626',
    'CONDITIONAL': '#d97706',
    'REVIEW':      '#6b7280',
}
RULES_APPROVE = [
    'forced outage','perbaikan darurat','gangguan mendadak','emergency',
    'trip','korektif','vibrasi tinggi','overheating','bocor','kebocoran',
    'pecah','kebakaran','recovery','restorasi','gagal start','troubleshooting',
    'blackout','grid collapse','supply darurat','forced derating',
    'tidak terjadwal','mendadak','darurat','insidental','unplanned',
    'perbaikan','gangguan','shutdown','turbin','boiler','rca','root cause',
    'derating','overhaul','tagging','vibrasi','relay','error','rusak',
]
RULES_REJECT = [
    'logsheet harian','log sheet','pencatatan harian','inspeksi rutin',
    'patroli rutin','piket','serah terima shift','koordinasi shift',
    'pelaporan harian','housekeeping','pengecekan kondisi normal',
    'monitoring rutin','kondisi normal','kondisi stabil','beban stabil',
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif !important}
.stApp{background-color:#F0F4F8 !important}
section[data-testid="stSidebar"]{background-color:#FFFFFF !important;border-right:1px solid #E2E8F0 !important}
section[data-testid="stSidebar"] *{color:#1a202c !important}
.brand{background:linear-gradient(135deg,#1E3A5F,#2563EB);border-radius:16px;padding:28px 32px;margin-bottom:24px;position:relative;overflow:hidden}
.brand::before{content:'';position:absolute;top:-40px;right:-40px;width:180px;height:180px;background:radial-gradient(circle,rgba(255,255,255,0.15),transparent 70%);border-radius:50%}
.brand-title{font-size:26px;font-weight:700;color:#FFFFFF;margin:0 0 6px;letter-spacing:-0.5px}
.brand-sub{font-size:13px;color:rgba(255,255,255,0.75);font-family:'DM Mono',monospace;margin:0}
.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
.mcard{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:18px 22px;position:relative;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.06)}
.mcard::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0}
.mcard.app::after{background:#16a34a}.mcard.rej::after{background:#dc2626}
.mcard.con::after{background:#d97706}.mcard.rev::after{background:#94a3b8}
.mval{font-size:34px;font-weight:700;color:#1a202c;line-height:1;margin-bottom:4px}
.mlbl{font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;font-family:'DM Mono',monospace}
.mpct{font-size:12px;color:#94a3b8;margin-top:3px}
.info-box{background:#EFF6FF;border:1px solid #BFDBFE;border-left:4px solid #3b82f6;border-radius:8px;padding:12px 16px;margin:10px 0;font-size:13px;color:#1e40af}
.warn-box{background:#FFFBEB;border:1px solid #FDE68A;border-left:4px solid #f59e0b;border-radius:8px;padding:12px 16px;font-size:13px;color:#78350f;margin:8px 0}
.err-box{background:#FEF2F2;border:1px solid #FECACA;border-left:4px solid #ef4444;border-radius:8px;padding:12px 16px;font-size:13px;color:#991b1b;margin:8px 0}
.success-box{background:#F0FDF4;border:1px solid #BBF7D0;border-left:4px solid #16a34a;border-radius:8px;padding:12px 16px;font-size:13px;color:#14532d;margin:8px 0}
.step-card{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:18px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.step-num{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;background:#1E3A5F;border-radius:50%;font-size:12px;font-weight:700;color:#FFFFFF;font-family:'DM Mono',monospace;margin-right:10px}
.step-title{font-size:14px;font-weight:600;color:#1a202c}
.step-desc{font-size:12px;color:#64748b;margin-top:6px;margin-left:38px;line-height:1.5}
.log-container{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px 16px;font-family:'DM Mono',monospace;font-size:11px}
.log-line{color:#475569;padding:2px 0;border-bottom:1px solid #F1F5F9}
.section-title{font-size:17px;font-weight:700;color:#1a202c;margin:24px 0 14px}
.sidebar-brand{padding:8px 0 16px;border-bottom:1px solid #E2E8F0;margin-bottom:16px}
.threshold-info{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px;font-size:11px;color:#475569;line-height:2}
.ref-badge{background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:10px 12px;font-size:12px;color:#14532d;margin-top:8px}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.5rem;padding-bottom:2rem}
.stButton>button[kind="primary"]{background:#1E3A5F !important;border:none !important;color:#FFFFFF !important;border-radius:8px !important;font-weight:600 !important;font-size:15px !important;padding:14px !important}
.stButton>button[kind="primary"]:hover{background:#2563EB !important;box-shadow:0 4px 20px rgba(30,58,95,0.25) !important}
.stButton>button:not([kind="primary"]){background:#FFFFFF !important;border:1px solid #CBD5E1 !important;color:#374151 !important;border-radius:8px !important;font-weight:500 !important}
.stDownloadButton button{background:#1E3A5F !important;border:none !important;color:#FFFFFF !important;border-radius:8px !important;font-weight:600 !important;font-size:15px !important;padding:14px !important}
.stDownloadButton button:hover{background:#2563EB !important}
hr{border-color:#E2E8F0 !important}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#F1F5F9}
::-webkit-scrollbar-thumb{background:#CBD5E1;border-radius:3px}
</style>
""", unsafe_allow_html=True)

def init_state():
    defaults = {
        'df_ref':None,'embeddings_ref':None,'list_ref':None,
        'df_hasil':None,'audit_done':False,'ref_loaded':False,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v
init_state()

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

def validate_ref_file(df):
    return [c for c in [REF_COL_DESC,REF_COL_STATUS,REF_COL_BIDANG,REF_COL_JENIS] if c not in df.columns]

def validate_data_file(df):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]

def cek_keyword(desc_lower):
    found_approve=[w for w in RULES_APPROVE if w in desc_lower]
    found_reject=[w for w in RULES_REJECT if w in desc_lower]
    if found_approve and found_reject:
        return "CONDITIONAL",f"Konflik: '{found_approve[0]}' vs '{found_reject[0]}'"
    if found_reject:
        return "REJECTED",f"Kegiatan rutin: '{found_reject[0]}'"
    if found_approve:
        return "APPROVED",f"Kegiatan darurat: '{found_approve[0]}'"
    return None,None

def audit_satu(deskripsi,df_ref,model,embeddings_ref,list_ref):
    if pd.isna(deskripsi) or str(deskripsi).strip()=='':
        return ('REVIEW','-','-','0.0%','-','Data kosong atau tidak valid')
    desc=str(deskripsi).strip()
    desc_lower=desc.lower()
    try:
        emb_desc=model.encode([desc])
        scores=cosine_similarity(emb_desc,embeddings_ref)[0]
        best_idx=int(np.argmax(scores))
        best_skor=round(float(scores[best_idx])*100,1)
        best_match=list_ref[best_idx]
        row=df_ref.iloc[best_idx]
        ref_status=row[REF_COL_STATUS]
        ref_bidang=row[REF_COL_BIDANG]
        ref_jenis=row[REF_COL_JENIS]
    except Exception as e:
        return ('REVIEW','-','-','-','-',f'Error: {str(e)[:60]}')
    skor_str=f"{best_skor}%"
    if best_skor>=THRESHOLD_KUAT*100:
        return (ref_status,ref_bidang,ref_jenis,skor_str,'Referensi Kuat',
                f"[{ref_bidang} – {ref_jenis}] Kecocokan sangat kuat ({best_skor}%): '{best_match[:65]}'")
    if best_skor>=THRESHOLD_SEDANG*100:
        return (ref_status,ref_bidang,ref_jenis,skor_str,'Referensi Sedang',
                f"[{ref_bidang} – {ref_jenis}] Kecocokan sedang ({best_skor}%): '{best_match[:65]}'")
    if best_skor>=THRESHOLD_LEMAH*100:
        kw_status,kw_alasan=cek_keyword(desc_lower)
        if kw_status:
            return (kw_status,'-','-',skor_str,'Keyword',
                    f"{kw_alasan}. Referensi terdekat ({best_skor}%): '{best_match[:50]}'")
        return ('CONDITIONAL',ref_bidang,ref_jenis,skor_str,'Referensi Lemah',
                f"Kecocokan rendah ({best_skor}%). Ref: '{best_match[:55]}'. Verifikasi manual diperlukan.")
    kw_status,kw_alasan=cek_keyword(desc_lower)
    if kw_status:
        return (kw_status,'-','-',skor_str,'Keyword',f"{kw_alasan} (similarity rendah: {best_skor}%)")
    return ('REVIEW','-','-',skor_str,'Tidak Cocok',
            f"Deskripsi tidak dikenali (skor maks {best_skor}%). Butuh verifikasi manual.")

def buat_chart(df_hasil):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    fig.patch.set_facecolor('#FFFFFF')
    counts=df_hasil['AI Status'].value_counts()
    order=[s for s in ['APPROVED','REJECTED','CONDITIONAL','REVIEW'] if s in counts.index]
    values=[counts[s] for s in order]
    colors=[STATUS_COLOR.get(s,'#6b7280') for s in order]
    ax1.set_facecolor('#FFFFFF')
    bars=ax1.bar(order,values,color=colors,width=0.5,zorder=3)
    for sp in ['top','right']:ax1.spines[sp].set_visible(False)
    for sp in ['bottom','left']:ax1.spines[sp].set_color('#E2E8F0')
    ax1.tick_params(colors='#64748b',labelsize=9)
    ax1.yaxis.grid(True,color='#F1F5F9',zorder=0)
    ax1.set_title('Distribusi Keputusan',color='#1a202c',fontsize=12,pad=10,fontweight='600')
    for bar,val in zip(bars,values):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.15,
                 str(val),ha='center',va='bottom',color='#1a202c',fontsize=11,fontweight='700')
    ax2.set_facecolor('#FFFFFF')
    wedges,texts,autotexts=ax2.pie(values,labels=order,colors=colors,autopct='%1.0f%%',
        startangle=90,wedgeprops={'linewidth':1.5,'edgecolor':'#FFFFFF'},pctdistance=0.75)
    for t in texts:t.set_color('#64748b');t.set_fontsize(9)
    for at in autotexts:at.set_color('#FFFFFF');at.set_fontsize(10);at.set_fontweight('700')
    ax2.set_title('Proporsi',color='#1a202c',fontsize=12,pad=10,fontweight='600')
    plt.tight_layout(pad=2)
    return fig

def export_excel(df_hasil):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer:
        df_hasil.to_excel(writer,sheet_name='Hasil Audit',index=False)
        df_hasil[df_hasil['AI Status'].isin(['REVIEW','CONDITIONAL'])].to_excel(writer,sheet_name='Perlu Review',index=False)
        df_hasil[df_hasil['AI Status']=='APPROVED'].to_excel(writer,sheet_name='Approved',index=False)
        df_hasil[df_hasil['AI Status']=='REJECTED'].to_excel(writer,sheet_name='Rejected',index=False)
    buf.seek(0)
    return buf

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:28px">⚡</div>
        <div style="font-size:16px;font-weight:700;color:#1E3A5F;margin-top:6px">Audit Lembur</div>
        <div style="font-size:10px;color:#94a3b8;font-family:'DM Mono',monospace">Sistem Evaluasi Otomatis v1.2</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**👤 Nama Auditor**")
    nama_auditor=st.text_input("nama_auditor",placeholder="contoh: Budi Santoso",
                                label_visibility="collapsed",
                                help="Nama ini akan tercatat di file hasil audit")
    st.divider()
    st.markdown("**📚 File Referensi**")
    ref_file=st.file_uploader("Upload Referensi_Lembur_v2.xlsx",type=['xlsx'],
                               key='ref_upload',help="Upload sekali, berlaku untuk semua sesi")
    if ref_file:
        if not st.session_state.ref_loaded:
            with st.spinner("Memuat referensi & model AI..."):
                try:
                    df_ref=pd.read_excel(ref_file,sheet_name=REF_SHEET_NAME)
                    missing=validate_ref_file(df_ref)
                    if missing:
                        st.error(f"Kolom tidak ditemukan: {missing}")
                    else:
                        model_st=load_model()
                        list_ref=df_ref[REF_COL_DESC].astype(str).tolist()
                        emb_ref=model_st.encode(list_ref,show_progress_bar=False)
                        st.session_state.df_ref=df_ref
                        st.session_state.list_ref=list_ref
                        st.session_state.embeddings_ref=emb_ref
                        st.session_state.ref_loaded=True
                except Exception as e:
                    st.error(f"Gagal memuat: {e}")
        if st.session_state.ref_loaded:
            df_r=st.session_state.df_ref
            st.markdown(f"""
            <div class="ref-badge">
                ✅ <strong>{len(df_r)}</strong> referensi aktif<br>
                <span style="color:#64748b;font-size:11px">
                {df_r[REF_COL_BIDANG].nunique()} bidang &nbsp;·&nbsp;
                {df_r[REF_COL_JENIS].nunique()} jenis gangguan</span>
            </div>""", unsafe_allow_html=True)
            if st.button("🔄 Ganti file referensi",use_container_width=True):
                st.session_state.ref_loaded=False
                st.rerun()
    st.divider()
    st.markdown("""
    <div class="threshold-info">
        <div style="font-size:11px;font-weight:600;color:#374151;margin-bottom:4px;font-family:'DM Mono',monospace">THRESHOLD SIMILARITY</div>
        🟢 Kuat &nbsp;&nbsp;&nbsp; ≥ 80%<br>
        🟡 Sedang &nbsp; 60 – 79%<br>
        🟠 Lemah &nbsp;&nbsp; 45 – 59%<br>
        ⚪ Rendah &nbsp; &lt; 45%
    </div>""", unsafe_allow_html=True)

# MAIN
st.markdown("""
<div class="brand">
    <div style="font-size:32px;margin-bottom:10px">⚡</div>
    <div class="brand-title">Sistem Audit Lembur Pembangkit</div>
    <div class="brand-sub">AI-powered &nbsp;·&nbsp; Sentence Transformer &nbsp;·&nbsp; paraphrase-multilingual-MiniLM-L12-v2</div>
</div>""", unsafe_allow_html=True)

if not st.session_state.ref_loaded:
    st.markdown('<div class="info-box">📋 Upload file <strong>Referensi_Lembur_v2.xlsx</strong> di sidebar kiri untuk memulai.</div>',unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    steps=[
        ("1","Upload Referensi","Upload Referensi_Lembur_v2.xlsx di sidebar kiri. Hanya perlu dilakukan sekali."),
        ("2","Upload Data Lembur","Upload file Excel data lembur yang ingin diaudit. Kolom wajib: NIP, Nama, Tanggal, Deskripsi."),
        ("3","Download Hasil","Klik Mulai Audit, tunggu proses, lalu download file Excel hasil audit 4 sheet."),
    ]
    for col,(num,title,desc) in zip([col1,col2,col3],steps):
        with col:
            st.markdown(f'<div class="step-card"><span class="step-num">{num}</span><span class="step-title">{title}</span><div class="step-desc">{desc}</div></div>',unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="section-title">📂 Upload Data Lembur</div>',unsafe_allow_html=True)
data_file=st.file_uploader("Pilih file Excel data lembur yang akan diaudit",type=['xlsx'],key='data_upload',help="Kolom wajib: NIP, Nama, Tanggal, Deskripsi")

if not data_file:
    st.markdown('<div class="warn-box">⚠️ Kolom wajib di file Excel: <strong>NIP &nbsp;·&nbsp; Nama &nbsp;·&nbsp; Tanggal &nbsp;·&nbsp; Deskripsi</strong></div>',unsafe_allow_html=True)
    st.stop()

try:
    df_baru=pd.read_excel(data_file)
    missing_cols=validate_data_file(df_baru)
    if missing_cols:
        st.markdown(f'<div class="err-box">❌ Kolom tidak ditemukan: <strong>{", ".join(missing_cols)}</strong><br>Pastikan nama kolom persis: NIP, Nama, Tanggal, Deskripsi</div>',unsafe_allow_html=True)
        st.stop()
except Exception as e:
    st.markdown(f'<div class="err-box">❌ Gagal membaca file: {e}</div>',unsafe_allow_html=True)
    st.stop()

col_a,col_b=st.columns([3,1])
with col_a:
    st.markdown(f'<div class="info-box">✅ <strong>{len(df_baru)} baris</strong> siap diaudit dari file <em>{data_file.name}</em></div>',unsafe_allow_html=True)
with col_b:
    kosong=df_baru['Deskripsi'].isna().sum()
    if kosong>0:
        st.markdown(f'<div class="warn-box">⚠️ {kosong} baris kosong</div>',unsafe_allow_html=True)

with st.expander("👁️ Preview data (5 baris pertama)"):
    st.dataframe(df_baru.head(),use_container_width=True)

st.divider()
if not nama_auditor.strip():
    st.markdown('<div class="warn-box">⚠️ Isi <strong>Nama Auditor</strong> di sidebar kiri sebelum memulai audit.</div>',unsafe_allow_html=True)
col_b1,col_b2,col_b3=st.columns([1,2,1])
with col_b2:
    mulai=st.button("🚀  Mulai Audit Sekarang",disabled=not nama_auditor.strip(),use_container_width=True,type="primary")

if mulai:
    model_st=load_model()
    df_ref=st.session_state.df_ref
    embeddings=st.session_state.embeddings_ref
    list_ref=st.session_state.list_ref
    total=len(df_baru)
    hasil_rows=[]
    st.divider()
    st.markdown('<div class="section-title">⏳ Sedang Memproses Audit...</div>',unsafe_allow_html=True)
    progress=st.progress(0)
    log_area=st.empty()
    log_lines=[]
    t_start=datetime.now()
    for i,(_,row) in enumerate(df_baru.iterrows()):
        desc=row.get('Deskripsi','')
        nama_p=row.get('Nama','-')
        tanggal=row.get('Tanggal','-')
        status,bidang,jenis,skor,sumber,alasan=audit_satu(desc,df_ref,model_st,embeddings,list_ref)
        hasil_rows.append({
            'Diperiksa Oleh':nama_auditor,'Tanggal Audit':t_start.strftime('%Y-%m-%d %H:%M'),
            'NIP':row.get('NIP','-'),'Nama':nama_p,'Tanggal Lembur':tanggal,'Deskripsi':desc,
            'AI Status':status,'Bidang Referensi':bidang,'Jenis Gangguan':jenis,
            'Skor Similarity':skor,'Sumber Keputusan':sumber,'Alasan Audit':alasan,
        })
        pct=int((i+1)/total*100)
        progress.progress(pct)
        color=STATUS_COLOR.get(status,'#6b7280')
        log_lines.append(f'<div class="log-line">[{i+1:03d}/{total}] <span style="color:#1a202c">{str(desc)[:50]:<50}</span> <span style="color:{color};font-weight:700">{status}</span> <span style="color:#94a3b8">({skor})</span></div>')
        if len(log_lines)>8:log_lines=log_lines[-8:]
        log_area.markdown(f'<div class="log-container">{"".join(log_lines)}</div>',unsafe_allow_html=True)
    t_end=datetime.now()
    durasi=round((t_end-t_start).total_seconds(),1)
    progress.empty()
    log_area.empty()
    st.session_state.df_hasil=pd.DataFrame(hasil_rows)
    st.session_state.audit_done=True
    st.markdown(f'<div class="success-box">✅ <strong>Audit selesai!</strong> &nbsp; {total} data diproses dalam <strong>{durasi} detik</strong> &nbsp;·&nbsp; Auditor: <strong>{nama_auditor}</strong> &nbsp;·&nbsp; {t_end.strftime("%d %b %Y, %H:%M")}</div>',unsafe_allow_html=True)

if st.session_state.audit_done and st.session_state.df_hasil is not None:
    df_hasil=st.session_state.df_hasil
    total=len(df_hasil)
    nama_aud=df_hasil['Diperiksa Oleh'].iloc[0]
    c_app=(df_hasil['AI Status']=='APPROVED').sum()
    c_rej=(df_hasil['AI Status']=='REJECTED').sum()
    c_con=(df_hasil['AI Status']=='CONDITIONAL').sum()
    c_rev=(df_hasil['AI Status']=='REVIEW').sum()
    st.divider()
    st.markdown('<div class="section-title">📊 Ringkasan Hasil Audit</div>',unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
        <div class="mcard app"><div class="mval">{c_app}</div><div class="mlbl">Approved</div><div class="mpct">{c_app/total*100:.1f}% dari total</div></div>
        <div class="mcard rej"><div class="mval">{c_rej}</div><div class="mlbl">Rejected</div><div class="mpct">{c_rej/total*100:.1f}% dari total</div></div>
        <div class="mcard con"><div class="mval">{c_con}</div><div class="mlbl">Conditional</div><div class="mpct">{c_con/total*100:.1f}% dari total</div></div>
        <div class="mcard rev"><div class="mval">{c_rev}</div><div class="mlbl">Perlu Review</div><div class="mpct">{c_rev/total*100:.1f}% dari total</div></div>
    </div>""",unsafe_allow_html=True)
    fig=buat_chart(df_hasil)
    st.pyplot(fig,use_container_width=True)
    plt.close()
    if (c_rev+c_con)>total*0.3:
        st.markdown(f'<div class="warn-box">⚠️ <strong>{c_rev+c_con} kasus</strong> ({(c_rev+c_con)/total*100:.0f}%) memerlukan verifikasi manual. Pertimbangkan untuk memperkaya database referensi.</div>',unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-title">📋 Detail Hasil</div>',unsafe_allow_html=True)
    tab_all,tab_rev,tab_app,tab_rej=st.tabs([f"🗂️ Semua ({total})",f"⚠️ Perlu Review ({c_rev+c_con})",f"✅ Approved ({c_app})",f"❌ Rejected ({c_rej})"])
    with tab_all:st.dataframe(df_hasil,use_container_width=True,height=380)
    with tab_rev:
        df_rv=df_hasil[df_hasil['AI Status'].isin(['REVIEW','CONDITIONAL'])]
        st.dataframe(df_rv,use_container_width=True,height=350) if len(df_rv) else st.success("✅ Tidak ada kasus yang perlu review manual!")
    with tab_app:st.dataframe(df_hasil[df_hasil['AI Status']=='APPROVED'],use_container_width=True,height=350)
    with tab_rej:st.dataframe(df_hasil[df_hasil['AI Status']=='REJECTED'],use_container_width=True,height=350)
    with st.expander("🔍 Detail sumber keputusan"):
        sumber_counts=df_hasil['Sumber Keputusan'].value_counts()
        col_s1,col_s2=st.columns([1,2])
        with col_s1:st.dataframe(sumber_counts.rename("Jumlah").reset_index(),use_container_width=True)
        with col_s2:
            st.markdown("""<div style="font-size:12px;color:#475569;line-height:2.2;padding:8px 0">
            <strong style="color:#1a202c">Keterangan sumber keputusan:</strong><br>
            🟢 <strong>Referensi Kuat</strong> — similarity ≥ 80%, sangat dipercaya<br>
            🟡 <strong>Referensi Sedang</strong> — similarity 60–79%, cukup dipercaya<br>
            🟠 <strong>Referensi Lemah</strong> — similarity 45–59%, perlu dicermati<br>
            🔤 <strong>Keyword</strong> — terdeteksi dari kata kunci rules<br>
            ⚪ <strong>Tidak Cocok</strong> — similarity terlalu rendah, perlu review manual
            </div>""",unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-title">📥 Download Hasil Audit</div>',unsafe_allow_html=True)
    nama_file_out=f"Hasil_Audit_{nama_aud.replace(' ','_')}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    buf=export_excel(df_hasil)
    col_d1,col_d2,col_d3=st.columns([1,2,1])
    with col_d2:
        st.download_button(label=f"📥  Download {nama_file_out}",data=buf,file_name=nama_file_out,
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
    st.markdown(f'<div class="info-box" style="text-align:center;margin-top:12px">📄 4 sheet: <strong>Hasil Audit · Perlu Review · Approved · Rejected</strong><br>👤 Auditor: <strong>{nama_aud}</strong> &nbsp;·&nbsp; 📅 <strong>{datetime.now().strftime("%d %B %Y, %H:%M")}</strong></div>',unsafe_allow_html=True)
    st.divider()
    col_r1,col_r2,col_r3=st.columns([1,2,1])
    with col_r2:
        if st.button("🔁  Audit File Lain",use_container_width=True):
            st.session_state.df_hasil=None
            st.session_state.audit_done=False
            st.rerun()
