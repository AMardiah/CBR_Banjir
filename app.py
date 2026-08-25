import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
 
# =============================================================================
# KONFIGURASI HALAMAN  (HARUS BARIS PERTAMA st.*)
# =============================================================================
st.set_page_config(
    page_title="Flood Aid Rapid Assessment · BPBD Sumbar",
    page_icon="🌊",
    layout="centered",
)
 
# =============================================================================
# INJEKSI CSS — LANGUAGE TOGGLE & HEADER STYLING
# Ditarget ke elemen header paling atas (first stHorizontalBlock).
# Tidak memengaruhi tombol lain (Submit, dsb.) di bawahnya.
# =============================================================================
st.markdown("""
<style>
/* ── 1. Wrapper header row: align items to center vertically ─────────────── */
[data-testid="stHorizontalBlock"]:first-of-type {
    align-items: center !important;
}
 
/* ── 2. Hapus padding berlebih di kolom tombol kanan ────────────────────── */
[data-testid="stHorizontalBlock"]:first-of-type
    > [data-testid="column"]:last-child {
    padding-top: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
}
 
/* ── 3. Container sub-kolom tombol ID/EN ────────────────────────────────── */
[data-testid="stHorizontalBlock"]:first-of-type
    > [data-testid="column"]:last-child
    [data-testid="stHorizontalBlock"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 999px !important;
    padding: 3px 4px !important;
    gap: 2px !important;
    width: fit-content !important;
    flex-wrap: nowrap !important;
}
 
/* ── 4. Semua tombol di dalam pill toggle ───────────────────────────────── */
[data-testid="stHorizontalBlock"]:first-of-type
    > [data-testid="column"]:last-child
    [data-testid="stHorizontalBlock"]
    .stButton > button {
    border-radius: 999px !important;
    border: none !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    padding: 5px 14px !important;
    min-width: 58px !important;
    transition: background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease !important;
    white-space: nowrap !important;
    line-height: 1.4 !important;
}
 
/* ── 5. Tombol AKTIF (primary) — pill bercahaya ─────────────────────────── */
[data-testid="stHorizontalBlock"]:first-of-type
    > [data-testid="column"]:last-child
    [data-testid="stHorizontalBlock"]
    .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1c6ef7 0%, #0ea5e9 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(14, 165, 233, 0.45) !important;
}
 
/* ── 6. Tombol INAKTIF (secondary) — transparan subtle ──────────────────── */
[data-testid="stHorizontalBlock"]:first-of-type
    > [data-testid="column"]:last-child
    [data-testid="stHorizontalBlock"]
    .stButton > button[kind="secondary"] {
    background: transparent !important;
    color: rgba(255,255,255,0.55) !important;
    box-shadow: none !important;
}
 
/* ── 7. Hover pada tombol inaktif ───────────────────────────────────────── */
[data-testid="stHorizontalBlock"]:first-of-type
    > [data-testid="column"]:last-child
    [data-testid="stHorizontalBlock"]
    .stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.10) !important;
    color: rgba(255,255,255,0.85) !important;
}
 
/* ── 8. Hapus border focus bawaan Streamlit pada kedua tombol ───────────── */
[data-testid="stHorizontalBlock"]:first-of-type
    > [data-testid="column"]:last-child
    [data-testid="stHorizontalBlock"]
    .stButton > button:focus:not(:active) {
    box-shadow: 0 0 0 2px rgba(14,165,233,0.5) !important;
    outline: none !important;
}
</style>
""", unsafe_allow_html=True)
 
# =============================================================================
# KONSTANTA & KOLOM FITUR  (tidak berubah dari versi asli)
# =============================================================================
FEATURE_COLS = [
    "Curah Hujan", "Topografi", "Luapan Sungai", "Bendungan",
    "Drainase", "Tanah Longsor", "Fasilitas Umum", "Kepadatan Penduduk",
]
 
MIN_SIMILARITY_THRESHOLD = 50.0
W1_RISK    = 0.6   # bobot risiko fisik  (Rpred vs Rk)  — Eq.(3)
W2_DENSITY = 0.4   # bobot kerentanan    (Dnew  vs Dk)  — Eq.(3)
 
# =============================================================================
# ── BAGIAN BARU: MANAJEMEN BAHASA ────────────────────────────────────────────
# =============================================================================
 
# Inisialisasi bahasa di session_state (default: Indonesia)
if "lang" not in st.session_state:
    st.session_state["lang"] = "id"
 
 
def t(key: str) -> str:
    """Shortcut: ambil terjemahan untuk key tertentu sesuai bahasa aktif."""
    return T[st.session_state["lang"]][key]
 
 
# -----------------------------------------------------------------------------
# KAMUS TERJEMAHAN LENGKAP
# Tambahkan pasangan baru di sini jika ada string UI baru di masa mendatang.
# -----------------------------------------------------------------------------
T = {
    # ── BAHASA INDONESIA ─────────────────────────────────────────────────────
    "id": {
        # Judul & caption
        "app_title": "Kaji Cepat Bantuan Pasca-Bencana Banjir",
        "app_caption": (
            "Badan Penanggulangan Bencana Daerah (BPBD) Provinsi Sumatera Barat "
            "— Prototipe Sistem Hybrid CBR–Ridge Regression"
        ),
        "app_info":         (
            "**Petunjuk pengisian:** pilih opsi yang paling sesuai dengan laporan "
            "visual atau data riil di lokasi kejadian. Setiap parameter memiliki "
            "3 tingkatan: **Ringan, Sedang, Tinggi**."
        ),
        # Form identitas
        "form_identity":    "Identitas Wilayah",
        "label_kecamatan":  "Nama Kecamatan",
        "ph_kecamatan":     "Contoh: Pauh",
        "label_kelurahan":  "Nama Kelurahan / Nagari",
        "ph_kelurahan":     "Contoh: Kapalo Koto",
        # Bagian form
        "sec_weather":      "A. Kondisi Cuaca & Aliran Air",
        "sec_geo":          "B. Karakteristik Geografis Wilayah",
        "sec_infra":        "C. Dampak Infrastruktur & Populasi",
        # Label tier selectbox
        "q1":   "1. Intensitas Curah Hujan Terkini",
        "q2":   "2. Status Luapan Aliran Sungai (Batang Air)",
        "q3":   "3. Kondisi Drainase / Selokan Pemukiman",
        "q4":   "4. Kontur / Topografi Lokasi Terdampak",
        "q5":   "5. Potensi / Longsor Susulan",
        "q6":   "6. Kondisi Fisik Tanggul / Bendungan Terdekat",
        "q7":   "7. Aksesibilitas Transportasi Jalur Logistik",
        "q8":   "8. Estimasi Jumlah Jiwa / Penduduk Terdampak",
        # Pilihan tier
        "tier_low":  "Ringan",
        "tier_mod":  "Sedang",
        "tier_high": "Tinggi",
        # Caption bawah form
        "caption_density": (
            "Catatan: jika Kelurahan yang Anda masukkan sudah tercakup di database "
            "kerentanan BPBD, skor kerentanan kemanusiaan pada hasil analisis akan "
            "memakai data riil tersebut, bukan pilihan di atas — pilihan ini hanya "
            "dipakai sebagai fitur risiko fisik & fallback."
        ),
        "btn_submit":       "Kirim & Analisis Kasus Hibrida",
        # Validasi
        "err_required":     "Nama Kecamatan dan Kelurahan wajib diisi.",
        # Hasil
        "results_header":   "Lembar Hasil Kaji Cepat Bantuan",
        "stage1_title":     "Tahap 1: Tingkat Risiko Fisik Wilayah (Ridge Regression)",
        "metric_rpred":     "Rpred",
        "caption_r2":       (
            "Model ini menjelaskan sekitar 30% variasi risiko banjir (R²≈0.30) — "
            "batas ini disengaja demi formulir yang sederhana untuk petugas, bukan "
            "kegagalan model. Gunakan sebagai salah satu bahan pertimbangan, bukan "
            "satu-satunya dasar keputusan."
        ),
        # Tingkat risiko
        "sev_low":  "RISIKO RENDAH",
        "sev_med":  "RISIKO SEDANG - SIAGA",
        "sev_high": "RISIKO TINGGI - AWAS",
        # Tahap 2
        "stage2_title":     "Tahap 2: Analisis Kemiripan Kasus Historis",
        "sim_label":        "Tingkat Kemiripan",
        "sim_found":        "Kasus serupa ditemukan (kemiripan tertinggi: {sim:.1f}%).",
        "sim_notfound":     (
            "KASUS TIDAK DITEMUKAN: kemiripan tertinggi yang tersedia hanya "
            "{sim:.1f}%, di bawah ambang minimum {thresh:.0f}%. Rekomendasi otomatis "
            "TIDAK diberikan — kasus ini harus dieskalasi ke keputusan manual "
            "petugas/pakar lapangan."
        ),
        "cur_case":         "Kasus Saat Ini (Real-Time)",
        "entry_loc":        "Lokasi Entry:",
        "phys_risk":        "Risiko Fisik (Rpred):",
        "vuln_score":       "Skor Kerentanan Kemanusiaan:",
        "src_real":         "Sumber kerentanan: database registry BPBD (data riil).",
        "src_est":          "Sumber kerentanan: estimasi manual (kelurahan di luar cakupan registry).",
        "nearest_found":    "Kasus Terdekat Terpilih",
        "nearest_below":    "Kasus Terdekat (Belum Memenuhi Ambang)",
        "case_id_hist":     "ID / Riwayat:",
        "hist_risk":        "Risiko Fisik Lama:",
        "hist_vuln":        "Skor Kerentanan Lama:",
        # Tahap 3
        "stage3_title":     "Rekomendasi Manajemen Bantuan Pasca-Bencana",
        "aid_found_intro":  (
            "Berdasarkan pencocokan pola CBR terhadap tingkat kemiripan karakteristik "
            "wilayah di atas, berikut instruksi taktis distribusi bantuan:"
        ),
        "aid_notfound_intro": (
            "Sistem tidak menemukan kasus historis yang cukup mirip untuk "
            "direkomendasikan secara otomatis. Poin di bawah adalah tindak lanjut "
            "yang disarankan:"
        ),
        "aid_manual":       (
            "Tidak ada kasus historis yang cukup mirip — keputusan bantuan harus "
            "ditentukan manual oleh petugas/pakar lapangan."
        ),
        "warn_prefix":      "[Peringatan data]",
        # Expander
        "expander_title":   "Tentang sistem ini / keterbatasan",
        "expander_body":    (
            "- Model Ridge Regression dilatih dari data Kaggle Flood Prediction asli "
            "(1.048.575 baris), R² = 0,2954 — batas struktural dari pembatasan 8/20 "
            "fitur, bukan kegagalan model.\n"
            "- Case base saat ini berisi 9 kasus historis riil dari satu kecamatan "
            "(Kec. Pauh); belum divalidasi lintas kecamatan/kabupaten lain.\n"
            "- Bobot w1=0.6/w2=0.4 dan ambang similarity 50% adalah nilai default, "
            "belum diuji empiris.\n"
            "- Skor kerentanan historis (Rk) merupakan transformasi langsung dari "
            "kepadatan populasi (Eq. 1), bukan estimasi risiko fisik independen — "
            "lihat naskah bagian Limitations."
        ),
        # Peringatan kelurahan tidak ditemukan (dinamis — format() dipanggil di runtime)
        "warn_not_found": (
            "Kelurahan '{kel}' tidak ditemukan di database kerentanan BPBD "
            "(baru mencakup Kec. Pauh). Jumlah jiwa terdampak di bawah ini adalah "
            "ESTIMASI dari tier 'Kepadatan Penduduk' yang dipilih di form, diskalakan "
            "proporsional terhadap kasus historis terparah (Kapalo Koto, {max_d} jiwa) "
            "— BUKAN data registry riil. Perlakukan dengan hati-hati."
        ),
    },
 
    # ── ENGLISH ──────────────────────────────────────────────────────────────
    "en": {
        # Title & caption
        "app_title": "Post-Flood Aid Rapid Assessment",
        "app_caption": (
            "Regional Disaster Management Agency (BPBD) West Sumatra Province "
            "— Hybrid CBR–Ridge Regression System Prototype"
        ),
        "app_info":         (
            "**Instructions:** select the option that best matches the visual "
            "field report or real data at the incident location. Each parameter "
            "has 3 severity tiers: **Low, Moderate, High**."
        ),
        # Form identity
        "form_identity":    "Location Identity",
        "label_kecamatan":  "Sub-District (Kecamatan)",
        "ph_kecamatan":     "e.g.: Pauh",
        "label_kelurahan":  "Village / Nagari (Kelurahan)",
        "ph_kelurahan":     "e.g.: Kapalo Koto",
        # Form sections
        "sec_weather":      "A. Weather & Water Flow Conditions",
        "sec_geo":          "B. Geographic Characteristics",
        "sec_infra":        "C. Infrastructure & Population Impact",
        # Selectbox labels
        "q1":   "1. Current Rainfall Intensity",
        "q2":   "2. River / Waterway Overflow Status",
        "q3":   "3. Residential Drainage / Canal Condition",
        "q4":   "4. Terrain Contour / Topography of Affected Area",
        "q5":   "5. Landslide / Secondary Landslide Potential",
        "q6":   "6. Physical Condition of Nearest Levee / Dam",
        "q7":   "7. Logistics Route / Transport Accessibility",
        "q8":   "8. Estimated Number of Affected Residents",
        # Tier options
        "tier_low":  "Low",
        "tier_mod":  "Moderate",
        "tier_high": "High",
        # Caption below form
        "caption_density": (
            "Note: if the entered Village is already in the BPBD vulnerability "
            "registry, the humanitarian vulnerability score in the output will use "
            "the real registry data instead of the option selected above — the "
            "selection above is only used as a physical risk feature and fallback."
        ),
        "btn_submit":       "Submit & Run Hybrid Case Analysis",
        # Validation
        "err_required":     "Sub-District and Village/Nagari fields are required.",
        # Results
        "results_header":   "Rapid Aid Assessment Output Sheet",
        "stage1_title":     "Stage 1: Physical Flood Risk Score (Ridge Regression)",
        "metric_rpred":     "Rpred",
        "caption_r2":       (
            "This model explains approximately 30% of flood risk variance (R²≈0.30) — "
            "this ceiling is intentional, keeping the form simple for field officers, "
            "not a modeling failure. Use it as one input among several, not the sole "
            "basis for decisions."
        ),
        # Severity labels
        "sev_low":  "LOW RISK",
        "sev_med":  "MODERATE RISK — ALERT",
        "sev_high": "HIGH RISK — WARNING",
        # Stage 2
        "stage2_title":     "Stage 2: Historical Case Similarity Analysis",
        "sim_label":        "Similarity Score",
        "sim_found":        "Similar case found (highest similarity: {sim:.1f}%).",
        "sim_notfound":     (
            "NO MATCHING CASE: highest available similarity is only {sim:.1f}%, "
            "below the minimum threshold of {thresh:.0f}%. Automated recommendation "
            "is NOT provided — this case must be escalated to manual expert/field "
            "officer decision."
        ),
        "cur_case":         "Current Case (Real-Time)",
        "entry_loc":        "Entry Location:",
        "phys_risk":        "Physical Risk Score (Rpred):",
        "vuln_score":       "Humanitarian Vulnerability Score:",
        "src_real":         "Vulnerability source: BPBD registry database (real data).",
        "src_est":          "Vulnerability source: manual estimate (village outside registry coverage).",
        "nearest_found":    "Best-Matched Historical Case",
        "nearest_below":    "Nearest Case (Below Similarity Threshold)",
        "case_id_hist":     "Case ID / History:",
        "hist_risk":        "Historical Physical Risk:",
        "hist_vuln":        "Historical Vulnerability Score:",
        # Stage 3
        "stage3_title":     "Post-Disaster Aid Management Recommendation",
        "aid_found_intro":  (
            "Based on CBR pattern matching against the case similarity score above, "
            "the following tactical aid distribution instructions are recommended:"
        ),
        "aid_notfound_intro": (
            "The system did not find a sufficiently similar historical case for "
            "automated recommendation. The points below are suggested follow-up actions:"
        ),
        "aid_manual":       (
            "No sufficiently similar historical case found — aid decisions must be "
            "determined manually by field officers or domain experts."
        ),
        "warn_prefix":      "[Data Warning]",
        # Expander
        "expander_title":   "About this system / limitations",
        "expander_body":    (
            "- The Ridge Regression model was trained on the original Kaggle Flood "
            "Prediction dataset (1,048,575 rows), R² = 0.2954 — a structural ceiling "
            "from restricting to 8 of 20 available features, not a modeling failure.\n"
            "- The case base currently contains 9 real historical cases from a single "
            "sub-district (Kec. Pauh); not yet validated across other sub-districts or "
            "regencies.\n"
            "- Weights w1=0.6/w2=0.4 and the 50% similarity threshold are expert-informed "
            "defaults, not yet empirically tuned.\n"
            "- The historical vulnerability score (Rk) is a direct linear transformation "
            "of population density (Eq. 1), not an independent physical risk estimate — "
            "see the Limitations section of the paper."
        ),
        # Dynamic warning for unknown village
        "warn_not_found": (
            "Village '{kel}' was not found in the BPBD vulnerability database "
            "(currently covering Kec. Pauh only). The affected population figure "
            "below is an ESTIMATE derived from the 'Population Density' tier "
            "selected in the form, scaled proportionally against the worst historical "
            "case (Kapalo Koto, {max_d} persons) — NOT real registry data. "
            "Treat with caution."
        ),
    },
}
 
# -----------------------------------------------------------------------------
# LABEL TIER BAHASA INGGRIS  (override config.json tanpa mengubah file eksternal)
# Kunci internal tetap "Ringan"/"Sedang"/"Tinggi" agar lookup ke config.json
# tetap valid. Hanya teks TAMPILAN di selectbox yang diubah dalam mode EN.
# -----------------------------------------------------------------------------
TIER_LABELS_EN = {
    "Curah Hujan": {
        "Ringan": "Low — minor puddles, drains absorbing normally",
        "Sedang": "Moderate — localized flooding in low-lying areas",
        "Tinggi": "High — extreme rainfall, area-wide inundation",
    },
    "Luapan Sungai": {
        "Ringan": "Low — water below danger level, banks intact",
        "Sedang": "Moderate — water near or at the bank edge",
        "Tinggi": "High — river overflowing, significant overflow volume",
    },
    "Drainase": {
        "Ringan": "Good — drains flowing, minimal blockage",
        "Sedang": "Partial — partially blocked, slow drainage",
        "Tinggi": "Poor — heavily blocked or completely non-functional",
    },
    "Topografi": {
        "Ringan": "Flat / elevated — low natural accumulation risk",
        "Sedang": "Gentle slope — moderate runoff accumulation",
        "Tinggi": "Valley / basin — high natural water accumulation",
    },
    "Tanah Longsor": {
        "Ringan": "Low — stable slopes, no signs of movement",
        "Sedang": "Moderate — minor cracks or signs of instability",
        "Tinggi": "High — active landslide or imminent collapse risk",
    },
    "Bendungan": {
        "Ringan": "Intact — no visible damage, operating normally",
        "Sedang": "Minor damage — reduced capacity, monitoring needed",
        "Tinggi": "Critical — severely damaged or at risk of failure",
    },
    "Fasilitas Umum": {
        "Ringan": "Accessible — roads and logistics routes open",
        "Sedang": "Limited — some roads flooded or damaged",
        "Tinggi": "Inaccessible — main routes cut off, requires alternative",
    },
    "Kepadatan Penduduk": {
        "Ringan": "Low — few affected residents (<500 people est.)",
        "Sedang": "Moderate — medium-scale impact (500–2,000 est.)",
        "Tinggi": "High — large-scale impact (>2,000 people est.)",
    },
}
 
# =============================================================================
# MUAT ARTEFAK (di-cache agar tidak reload setiap interaksi)
# =============================================================================
@st.cache_resource
def load_model():
    return joblib.load("flood_model.pkl")
 
 
@st.cache_data
def load_config():
    with open("config.json", encoding="utf-8") as f:
        return json.load(f)
 
 
@st.cache_data
def load_vuln_lookup():
    df = pd.read_excel("vulnerability_lookup.xlsx")
    df["kelurahan"] = df["kelurahan"].str.strip().str.upper()
    return df
 
 
@st.cache_data
def load_case_base():
    return pd.read_excel("case_base.xlsx")
 
 
MODEL       = load_model()
TIER_CONFIG = load_config()
VULN_LOOKUP = load_vuln_lookup()
CASE_BASE   = load_case_base()
 
MIN_DENSITY = float(VULN_LOOKUP["population_density"].min())
MAX_DENSITY = float(VULN_LOOKUP["population_density"].max())
 
 
# =============================================================================
# LOGIKA INTI  — identik dengan versi Flask (Eq.1–4 tidak diubah)
# =============================================================================
def normalize_density(raw_value: float) -> float:
    """Eq.(1): Risk_Score = (D - Dmin) / (Dmax - Dmin)."""
    if MAX_DENSITY == MIN_DENSITY:
        return 0.0
    return (raw_value - MIN_DENSITY) / (MAX_DENSITY - MIN_DENSITY)
 
 
def get_population_density(kelurahan_input: str, fallback_tier_value: float,
                            lang: str = "id") -> dict:
    """
    Cari data kerentanan di registry.
    Jika tidak ada, gunakan estimasi dari tier value.
    Pesan peringatan disesuaikan dengan bahasa aktif.
    """
    key   = kelurahan_input.strip().upper()
    match = VULN_LOOKUP[VULN_LOOKUP["kelurahan"] == key]
 
    if len(match) > 0:
        raw = float(match.iloc[0]["population_density"])
        return {"raw": raw, "source": "database_bpbd", "warning": None}
 
    # Fallback: skala proporsional terhadap kasus terparah
    TIER_MAX = 8.0
    raw      = (fallback_tier_value / TIER_MAX) * MAX_DENSITY
    warning  = T[lang]["warn_not_found"].format(
        kel=kelurahan_input, max_d=int(MAX_DENSITY)
    )
    return {"raw": raw, "source": "estimasi_manual", "warning": warning}
 
 
def weighted_distance(r_new, d_new_norm, r_k, d_k_norm,
                      w1: float = W1_RISK, w2: float = W2_DENSITY) -> np.ndarray:
    """Eq.(3): d(Cnew,Ck) = sqrt( w1*(Rpred-Rk)² + w2*(Dnew-Dk)² )."""
    return np.sqrt(w1 * (r_new - r_k) ** 2 + w2 * (d_new_norm - d_k_norm) ** 2)
 
 
def retrieve_case(r_pred: float, d_new_raw: float,
                  lang: str = "id", top_k: int = 1) -> dict:
    """
    CBR Retrieve (Eq.3 & Eq.4).
    Kembalikan kasus terdekat + pesan status dalam bahasa yang dipilih.
    """
    cb              = VULN_LOOKUP.copy()
    d_new_norm      = normalize_density(d_new_raw)
    cb["density_norm"] = cb["population_density"].apply(normalize_density)
    cb["distance"]  = weighted_distance(
        r_pred, d_new_norm, cb["population_density"], cb["density_norm"]
    )
 
    d_max = cb["distance"].max()
    cb["similarity_pct"] = (
        100 * (1 - cb["distance"] / d_max) if d_max > 0 else 100.0
    )
    results = cb.sort_values("distance").head(top_k)
 
    best_sim = float(results.iloc[0]["similarity_pct"])
    found    = best_sim >= MIN_SIMILARITY_THRESHOLD
 
    if found:
        message = T[lang]["sim_found"].format(sim=best_sim)
    else:
        message = T[lang]["sim_notfound"].format(
            sim=best_sim, thresh=MIN_SIMILARITY_THRESHOLD
        )
 
    return {
        "found":      found,
        "message":    message,
        "results":    results,
        "d_new_norm": d_new_norm,
    }
 
 
def severity_label(flood_prob_pct: float, lang: str = "id") -> str:
    """Kembalikan label tingkat kesiagaan dalam bahasa aktif."""
    if flood_prob_pct < 40:
        return T[lang]["sev_low"]
    elif flood_prob_pct < 60:
        return T[lang]["sev_med"]
    else:
        return T[lang]["sev_high"]
 
 
# =============================================================================
# ── UI RENDERING ─────────────────────────────────────────────────────────────
# =============================================================================
 
# ---------------------------------------------------------------------------
# TOMBOL TOGGLE BAHASA — dual-pill 🇮🇩 ID | 🇬🇧 EN di pojok kanan atas
# ---------------------------------------------------------------------------
def set_lang_id():
    """Callback: paksa bahasa Indonesia."""
    st.session_state["lang"] = "id"
 
def set_lang_en():
    """Callback: paksa bahasa English."""
    st.session_state["lang"] = "en"
 
 
lang = st.session_state["lang"]   # shortcut baca sesi saat ini
 
# ── Header: judul (kiri lebar) | pill toggle (kanan sempit) ──────────────
hdr_col, btn_col = st.columns([5, 2])
 
with hdr_col:
    st.title(t("app_title"))
    st.caption(t("app_caption"))
 
with btn_col:
    # Dua tombol dalam sub-baris → dilapisi CSS jadi satu pill tunggal
    pill_id, pill_en = st.columns(2, gap="small")
 
    with pill_id:
        # Tombol aktif = primary (biru glowing), inaktif = secondary (transparan)
        st.button(
            "🇮🇩 ID",
            key      = "btn_lang_id",
            on_click = set_lang_id,
            type     = "primary"    if lang == "id" else "secondary",
            use_container_width = True,
        )
    with pill_en:
        st.button(
            "🇬🇧 EN",
            key      = "btn_lang_en",
            on_click = set_lang_en,
            type     = "primary"    if lang == "en" else "secondary",
            use_container_width = True,
        )
 
st.info(t("app_info"), icon="ℹ️")
 
# ---------------------------------------------------------------------------
# HELPER TIER SELECTBOX — language-aware
# ---------------------------------------------------------------------------
def tier_selectbox(q_key: str, feature_key: str, lang: str):
    """
    Buat selectbox untuk satu parameter.
    - Kunci internal ("Ringan"/"Sedang"/"Tinggi") TIDAK berubah agar
      lookup ke TIER_CONFIG tetap valid.
    - Label tampilan disesuaikan dengan bahasa aktif:
        id  → dari config.json (label asli)
        en  → dari TIER_LABELS_EN (override lokal)
    """
    tiers   = TIER_CONFIG[feature_key]
    options = [t("tier_low"), t("tier_mod"), t("tier_high")]
    # Mapping tampilan → kunci internal config.json
    disp_to_internal = {
        t("tier_low"):  "Ringan",
        t("tier_mod"):  "Sedang",
        t("tier_high"): "Tinggi",
    }
 
    if lang == "id":
        def fmt(disp): return tiers[disp_to_internal[disp]]["label"]
    else:
        def fmt(disp): return TIER_LABELS_EN[feature_key][disp_to_internal[disp]]
 
    selected_disp = st.selectbox(
        t(q_key),
        options     = options,
        format_func = fmt,
        key         = f"select_{feature_key}_{lang}",  # key per-lang cegah konflik
    )
    internal_key = disp_to_internal[selected_disp]
    return internal_key, tiers
 
 
# ---------------------------------------------------------------------------
# FORM INPUT
# ---------------------------------------------------------------------------
with st.form("form_kaji_cepat"):
    st.subheader(t("form_identity"))
    col1, col2 = st.columns(2)
    with col1:
        kecamatan = st.text_input(t("label_kecamatan"), placeholder=t("ph_kecamatan"))
    with col2:
        kelurahan = st.text_input(t("label_kelurahan"), placeholder=t("ph_kelurahan"))
 
    # ── Bagian A
    st.subheader(t("sec_weather"))
    sel_hujan,    tc_hujan    = tier_selectbox("q1", "Curah Hujan",       lang)
    sel_sungai,   tc_sungai   = tier_selectbox("q2", "Luapan Sungai",     lang)
    sel_drainase, tc_drainase = tier_selectbox("q3", "Drainase",          lang)
 
    # ── Bagian B
    st.subheader(t("sec_geo"))
    sel_topo,    tc_topo    = tier_selectbox("q4", "Topografi",     lang)
    sel_longsor, tc_longsor = tier_selectbox("q5", "Tanah Longsor", lang)
 
    # ── Bagian C
    st.subheader(t("sec_infra"))
    sel_bendungan, tc_bendungan = tier_selectbox("q6", "Bendungan",          lang)
    sel_fasum,     tc_fasum     = tier_selectbox("q7", "Fasilitas Umum",     lang)
    sel_kepadatan, tc_kepadatan = tier_selectbox("q8", "Kepadatan Penduduk", lang)
    st.caption(t("caption_density"))
 
    submitted = st.form_submit_button(
        t("btn_submit"), use_container_width=True, type="primary"
    )
 
# ---------------------------------------------------------------------------
# PROSES & TAMPILKAN HASIL
# ---------------------------------------------------------------------------
if submitted:
    # Validasi
    if not kecamatan.strip() or not kelurahan.strip():
        st.error(t("err_required"))
        st.stop()
 
    # Susun vektor fitur (kunci fitur selalu bahasa Indonesia → cocok FEATURE_COLS)
    feature_values = {
        "Curah Hujan":       tc_hujan[sel_hujan]["value"],
        "Luapan Sungai":     tc_sungai[sel_sungai]["value"],
        "Drainase":          tc_drainase[sel_drainase]["value"],
        "Topografi":         tc_topo[sel_topo]["value"],
        "Tanah Longsor":     tc_longsor[sel_longsor]["value"],
        "Bendungan":         tc_bendungan[sel_bendungan]["value"],
        "Fasilitas Umum":    tc_fasum[sel_fasum]["value"],
        "Kepadatan Penduduk": tc_kepadatan[sel_kepadatan]["value"],
    }
    X_new = pd.DataFrame([feature_values])[FEATURE_COLS]
 
    # ── Eq.(2): Ridge Regression → Rpred ────────────────────────────────────
    r_pred        = float(MODEL.predict(X_new)[0])
    r_pred        = min(max(r_pred, 0.0), 1.0)
    flood_prob_pct = r_pred * 100
 
    # ── Eq.(1): Normalisasi kepadatan/kerentanan ─────────────────────────────
    kepadatan_tier_value = feature_values["Kepadatan Penduduk"]
    dens = get_population_density(kelurahan, kepadatan_tier_value, lang)
 
    # ── Eq.(3) & (4): CBR Retrieve ──────────────────────────────────────────
    retrieval = retrieve_case(r_pred, dens["raw"], lang, top_k=1)
    top_case  = retrieval["results"].iloc[0]
 
    # Bangun daftar bantuan
    if retrieval["found"]:
        bantuan_list = [b.strip() for b in str(top_case["aid_package"]).split(";")]
    else:
        bantuan_list = [t("aid_manual")]
 
    if dens["warning"]:
        bantuan_list.append(f"{t('warn_prefix')} {dens['warning']}")
 
    # ── Tampilkan hasil ─────────────────────────────────────────────────────
    st.divider()
    st.header(t("results_header"))
 
    # TAHAP 1 — Tingkat risiko fisik
    st.subheader(t("stage1_title"))
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(t("metric_rpred"), f"{flood_prob_pct:.2f}%")
    with c2:
        sev = severity_label(flood_prob_pct, lang)
        if "RENDAH" in sev or "LOW" in sev:
            st.success(sev)
        elif "SEDANG" in sev or "MODERATE" in sev:
            st.warning(sev)
        else:
            st.error(sev)
    st.caption(t("caption_r2"))
 
    # TAHAP 2 — Kemiripan kasus historis
    st.subheader(t("stage2_title"))
    sim_pct = float(top_case["similarity_pct"])
    if retrieval["found"]:
        st.success(f"{t('sim_label')}: {sim_pct:.2f}% — {retrieval['message']}")
    else:
        st.warning(f"{t('sim_label')}: {sim_pct:.2f}% — {retrieval['message']}")
 
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"**{t('cur_case')}**")
        st.write(f"{t('entry_loc')} **{kelurahan}, {t('label_kecamatan')} {kecamatan}**")
        st.write(f"{t('phys_risk')} **{flood_prob_pct:.2f}%**")
        st.write(f"{t('vuln_score')} **{retrieval['d_new_norm']*100:.1f}%**")
        if dens["source"] == "database_bpbd":
            st.caption(t("src_real"))
        else:
            st.caption(t("src_est"))
 
    with cc2:
        nearest_title = t("nearest_found") if retrieval["found"] else t("nearest_below")
        st.markdown(f"**{nearest_title}**")
        st.write(f"{t('case_id_hist')} **{top_case['case_id']} - {top_case['daerah']}**")
        st.write(f"{t('hist_risk')} **{float(top_case['risk_score'])*100:.2f}%**")
        st.write(f"{t('hist_vuln')} **{float(top_case['density_norm'])*100:.1f}%**")
 
    # TAHAP 3 — Rekomendasi bantuan
    st.subheader(t("stage3_title"))
    st.write(t("aid_found_intro") if retrieval["found"] else t("aid_notfound_intro"))
    for i, b in enumerate(bantuan_list, 1):
        st.markdown(f"{i}. {b}")
 
# ---------------------------------------------------------------------------
# EXPANDER — tentang sistem & keterbatasan
# ---------------------------------------------------------------------------
st.divider()
with st.expander(t("expander_title")):
    st.markdown(t("expander_body"))
 
