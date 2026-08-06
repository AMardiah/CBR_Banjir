import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
 
# ---------------------------------------------------------------------------
# Konfigurasi halaman (harus jadi perintah st. pertama)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kaji Cepat Bantuan Banjir - BPBD Sumbar",
    page_icon="\U0001F30A",
    layout="centered",
)
 
FEATURE_COLS = [
    "Curah Hujan", "Topografi", "Luapan Sungai", "Bendungan",
    "Drainase", "Tanah Longsor", "Fasilitas Umum", "Kepadatan Penduduk",
]
 
MIN_SIMILARITY_THRESHOLD = 50.0
W1_RISK = 0.6      # bobot sumbu risiko (Rpred vs Rk) -- sesuai naskah Eq.(3)
W2_DENSITY = 0.4   # bobot sumbu kepadatan/kerentanan (Dnew vs Dk) -- sesuai naskah Eq.(3)
# Catatan: bobot ini masih nilai default naskah asli, BELUM diuji ulang
# terhadap data historis riil.
 
 
# ---------------------------------------------------------------------------
# Muat artefak sekali saja (di-cache lintas sesi/pengguna oleh Streamlit)
# ---------------------------------------------------------------------------
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
 
 
MODEL = load_model()
TIER_CONFIG = load_config()
VULN_LOOKUP = load_vuln_lookup()
CASE_BASE = load_case_base()
 
# ---------- Referensi normalisasi TUNGGAL (persis definisi Eq. 1 naskah) ----------
MIN_DENSITY = float(VULN_LOOKUP["jumlah_kelompok_rentan"].min())
MAX_DENSITY = float(VULN_LOOKUP["jumlah_kelompok_rentan"].max())
 
 
# ---------------------------------------------------------------------------
# Logika inti (IDENTIK dengan app.py Flask -- tidak ada perubahan rumus)
# ---------------------------------------------------------------------------
def normalize_density(raw_value: float) -> float:
    """Eq. (1) naskah: Risk_Score = (D - Dmin) / (Dmax - Dmin)."""
    if MAX_DENSITY == MIN_DENSITY:
        return 0.0
    return (raw_value - MIN_DENSITY) / (MAX_DENSITY - MIN_DENSITY)
 
 
def get_population_density(kelurahan_input: str, fallback_tier_value: float) -> dict:
    key = kelurahan_input.strip().upper()
    match = VULN_LOOKUP[VULN_LOOKUP["kelurahan"] == key]
 
    if len(match) > 0:
        raw = float(match.iloc[0]["jumlah_kelompok_rentan"])
        return {"raw": raw, "source": "database_bpbd", "warning": None}
    else:
        TIER_MAX = 8.0
        raw = (fallback_tier_value / TIER_MAX) * MAX_DENSITY
        return {
            "raw": raw,
            "source": "estimasi_manual",
            "warning": (
                f"Kelurahan '{kelurahan_input}' tidak ditemukan di database "
                f"kerentanan BPBD (baru mencakup Kec. Pauh). Jumlah jiwa "
                f"terdampak di bawah ini adalah ESTIMASI dari tier "
                f"'Kepadatan Penduduk' yang dipilih di form, diskalakan "
                f"proporsional terhadap kasus historis terparah "
                f"(Kapalo Koto, {int(MAX_DENSITY)} jiwa) -- BUKAN data "
                f"registry riil. Perlakukan dengan hati-hati."
            ),
        }
 
 
def weighted_distance(r_new, d_new_norm, r_k, d_k_norm, w1=W1_RISK, w2=W2_DENSITY):
    """Eq. (3) naskah: d(Cnew,Ck) = sqrt(w1*(Rpred-Rk)^2 + w2*(Dnew-Dk)^2)."""
    return np.sqrt(w1 * (r_new - r_k) ** 2 + w2 * (d_new_norm - d_k_norm) ** 2)
 
 
def retrieve_case(r_pred: float, d_new_raw: float, top_k: int = 1) -> dict:
    cb = CASE_BASE.copy()
    d_new_norm = normalize_density(d_new_raw)
    cb["density_norm"] = cb["population_density"].apply(normalize_density)
 
    cb["distance"] = weighted_distance(r_pred, d_new_norm, cb["risk_score"], cb["density_norm"])
    d_max = cb["distance"].max()
    cb["similarity_pct"] = 100 * (1 - cb["distance"] / d_max) if d_max > 0 else 100.0
    results = cb.sort_values("distance").head(top_k)
 
    best_similarity = float(results.iloc[0]["similarity_pct"])
    found = best_similarity >= MIN_SIMILARITY_THRESHOLD
 
    if found:
        message = f"Kasus serupa ditemukan (kemiripan tertinggi: {best_similarity:.1f}%)."
    else:
        message = (
            f"KASUS TIDAK DITEMUKAN: kemiripan tertinggi yang tersedia hanya "
            f"{best_similarity:.1f}%, di bawah ambang minimum "
            f"{MIN_SIMILARITY_THRESHOLD:.0f}%. Rekomendasi otomatis TIDAK "
            f"diberikan -- kasus ini harus dieskalasi ke keputusan manual "
            f"petugas/pakar lapangan."
        )
    return {"found": found, "message": message, "results": results, "d_new_norm": d_new_norm}
 
 
def severity_label(flood_prob_pct: float) -> str:
    if flood_prob_pct < 40:
        return "RISIKO RENDAH"
    elif flood_prob_pct < 60:
        return "RISIKO SEDANG - SIAGA"
    else:
        return "RISIKO TINGGI - AWAS"
 
 
# ---------------------------------------------------------------------------
# UI - Header
# ---------------------------------------------------------------------------
st.title("\U0001F30A Kaji Cepat Bantuan Pasca-Bencana Banjir")
st.caption("Badan Penanggulangan Bencana Daerah (BPBD) Provinsi Sumatera Barat \u2014 Prototipe Sistem Hybrid CBR\u2013Ridge Regression")
st.info(
    "\u2728 **Petunjuk pengisian:** pilih opsi yang paling sesuai dengan laporan visual atau data riil "
    "di lokasi kejadian. Setiap parameter memiliki 3 tingkatan: Ringan, Sedang, Tinggi.",
    icon="\u2139\ufe0f",
)
 
# ---------------------------------------------------------------------------
# UI - Form input
# ---------------------------------------------------------------------------
with st.form("form_kaji_cepat"):
    st.subheader("Identitas Wilayah")
    col1, col2 = st.columns(2)
    with col1:
        kecamatan = st.text_input("Nama Kecamatan", placeholder="Contoh: Pauh")
    with col2:
        kelurahan = st.text_input("Nama Kelurahan / Nagari", placeholder="Contoh: Kapalo Koto")
 
    def tier_selectbox(label_ui, feature_key):
        tiers = TIER_CONFIG[feature_key]
        options = ["Ringan", "Sedang", "Tinggi"]
        return st.selectbox(
            label_ui,
            options=options,
            format_func=lambda t: tiers[t]["label"],
            key=f"select_{feature_key}",
        ), tiers
 
    st.subheader("A. Kondisi Cuaca & Aliran Air")
    sel_hujan, tc_hujan = tier_selectbox("1. Intensitas Curah Hujan Terkini", "Curah Hujan")
    sel_sungai, tc_sungai = tier_selectbox("2. Status Luapan Aliran Sungai (Batang Air)", "Luapan Sungai")
    sel_drainase, tc_drainase = tier_selectbox("3. Kondisi Drainase / Selokan Pemukiman", "Drainase")
 
    st.subheader("B. Karakteristik Geografis Wilayah")
    sel_topo, tc_topo = tier_selectbox("4. Kontur / Topografi Lokasi Terdampak", "Topografi")
    sel_longsor, tc_longsor = tier_selectbox("5. Potensi / Longsor Susulan", "Tanah Longsor")
 
    st.subheader("C. Dampak Infrastruktur & Populasi")
    sel_bendungan, tc_bendungan = tier_selectbox("6. Kondisi Fisik Tanggul / Bendungan Terdekat", "Bendungan")
    sel_fasum, tc_fasum = tier_selectbox("7. Aksesibilitas Transportasi Jalur Logistik", "Fasilitas Umum")
    sel_kepadatan, tc_kepadatan = tier_selectbox("8. Estimasi Jumlah Jiwa/Penduduk Terdampak", "Kepadatan Penduduk")
    st.caption(
        "Catatan: jika Kelurahan yang Anda masukkan sudah tercakup di database kerentanan BPBD, "
        "skor kerentanan kemanusiaan pada hasil analisis akan memakai data riil tersebut, bukan "
        "pilihan di atas -- pilihan ini hanya dipakai sebagai fitur risiko fisik & fallback."
    )
 
    submitted = st.form_submit_button("Kirim & Analisis Kasus Hibrida", use_container_width=True, type="primary")
 
# ---------------------------------------------------------------------------
# Proses & tampilkan hasil (menggantikan route /proses Flask)
# ---------------------------------------------------------------------------
if submitted:
    if not kecamatan.strip() or not kelurahan.strip():
        st.error("Nama Kecamatan dan Kelurahan wajib diisi.")
        st.stop()
 
    feature_values = {
        "Curah Hujan": tc_hujan[sel_hujan]["value"],
        "Luapan Sungai": tc_sungai[sel_sungai]["value"],
        "Drainase": tc_drainase[sel_drainase]["value"],
        "Topografi": tc_topo[sel_topo]["value"],
        "Tanah Longsor": tc_longsor[sel_longsor]["value"],
        "Bendungan": tc_bendungan[sel_bendungan]["value"],
        "Fasilitas Umum": tc_fasum[sel_fasum]["value"],
        "Kepadatan Penduduk": tc_kepadatan[sel_kepadatan]["value"],
    }
    X_new = pd.DataFrame([feature_values])[FEATURE_COLS]
 
    # ---------- Rpred (Ridge Regression, Eq. 2) ----------
    r_pred = float(MODEL.predict(X_new)[0])
    r_pred = min(max(r_pred, 0.0), 1.0)
    flood_prob_pct = r_pred * 100
 
    # ---------- Dnew (jumlah jiwa/kerentanan, skala SAMA dengan Dk Tabel 1) ----------
    kepadatan_tier_value = feature_values["Kepadatan Penduduk"]
    dens = get_population_density(kelurahan, kepadatan_tier_value)
 
    # ---------- CBR Retrieve (Eq. 3 & 4) ----------
    retrieval = retrieve_case(r_pred, dens["raw"], top_k=1)
    top_case = retrieval["results"].iloc[0]
 
    if retrieval["found"]:
        bantuan_list = [b.strip() for b in str(top_case["aid_package"]).split(";")]
    else:
        bantuan_list = [
            "Tidak ada kasus historis yang cukup mirip -- keputusan bantuan "
            "harus ditentukan manual oleh petugas/pakar lapangan.",
        ]
    if dens["warning"]:
        bantuan_list.append(f"[Peringatan data] {dens['warning']}")
 
    st.divider()
    st.header("Lembar Hasil Kaji Cepat Bantuan")
 
    # ---------- Tahap 1 ----------
    st.subheader("Tahap 1: Tingkat Risiko Fisik Wilayah (Ridge Regression)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Rpred", f"{flood_prob_pct:.2f}%")
    with c2:
        label = severity_label(flood_prob_pct)
        if "RENDAH" in label:
            st.success(label)
        elif "SEDANG" in label:
            st.warning(label)
        else:
            st.error(label)
    st.caption(
        "Model ini menjelaskan sekitar 30% variasi risiko banjir (R\u00b2\u22480.30) \u2014 batas ini "
        "disengaja demi formulir yang sederhana untuk petugas, bukan kegagalan model. Gunakan "
        "sebagai salah satu bahan pertimbangan, bukan satu-satunya dasar keputusan."
    )
 
    # ---------- Tahap 2 ----------
    st.subheader("Tahap 2: Analisis Kemiripan Kasus Historis")
    if retrieval["found"]:
        st.success(f"Tingkat Kemiripan: {top_case['similarity_pct']:.2f}% \u2014 {retrieval['message']}")
    else:
        st.warning(f"Tingkat Kemiripan: {top_case['similarity_pct']:.2f}% \u2014 {retrieval['message']}")
 
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Kasus Saat Ini (Real-Time)**")
        st.write(f"Lokasi Entry: **{kelurahan}, Kec. {kecamatan}**")
        st.write(f"Risiko Fisik (Rpred): **{flood_prob_pct:.2f}%**")
        st.write(f"Skor Kerentanan Kemanusiaan: **{retrieval['d_new_norm']*100:.1f}%**")
        if dens["source"] == "database_bpbd":
            st.caption("Sumber kerentanan: database registry BPBD (data riil).")
        else:
            st.caption("Sumber kerentanan: estimasi manual (kelurahan di luar cakupan registry).")
    with cc2:
        title = "Kasus Terdekat Terpilih" if retrieval["found"] else "Kasus Terdekat (Belum Memenuhi Ambang)"
        st.markdown(f"**{title}**")
        st.write(f"ID / Riwayat: **{top_case['case_id']} - {top_case['daerah']}**")
        st.write(f"Risiko Fisik Lama: **{float(top_case['risk_score'])*100:.2f}%**")
        st.write(f"Skor Kerentanan Lama: **{float(top_case['density_norm'])*100:.1f}%**")
 
    # ---------- Tahap 3 ----------
    st.subheader("Rekomendasi Manajemen Bantuan Pasca-Bencana")
    if retrieval["found"]:
        st.write("Berdasarkan pencocokan pola CBR terhadap tingkat kemiripan karakteristik wilayah di atas, berikut instruksi taktis distribusi bantuan:")
    else:
        st.write("Sistem tidak menemukan kasus historis yang cukup mirip untuk direkomendasikan secara otomatis. Poin di bawah adalah tindak lanjut yang disarankan:")
    for i, b in enumerate(bantuan_list, 1):
        st.markdown(f"{i}. {b}")
 
st.divider()
with st.expander("Tentang sistem ini / keterbatasan"):
    st.markdown(
        "- Model Ridge Regression dilatih dari data Kaggle Flood Prediction asli (1.048.575 baris), "
        "R\u00b2 = 0,2954 \u2014 batas struktural dari pembatasan 8/20 fitur, bukan kegagalan model.\n"
        "- Case base saat ini berisi 9 kasus historis riil dari satu kecamatan (Kec. Pauh); belum "
        "divalidasi lintas kecamatan/kabupaten lain.\n"
        "- Bobot w1=0.6/w2=0.4 dan ambang similarity 50% adalah nilai default, belum diuji empiris.\n"
        "- Skor kerentanan historis (Rk) merupakan transformasi langsung dari kepadatan populasi "
        "(Eq. 1), bukan estimasi risiko fisik independen \u2014 lihat naskah bagian Limitations."
    )
 
