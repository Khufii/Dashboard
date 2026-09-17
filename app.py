"""
app.py
============================================================
Sistem Deteksi Kesegaran & Estimasi Umur Simpan Kuning Telur Puyuh
============================================================
"""

import streamlit as st
from utils import (
    load_models,
    predict,
    MAX_SHELF_LIFE_DAYS,
    CLASS_NAMES
)

# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Pengenalan Kuning Telur Puyuh",
    page_icon="🥚",
    layout="centered"
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

    /* Background */
    .stApp {
        background-color: #FAF9F5;
    }

    /* Container utama */
    .block-container {
        max-width: 800px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Judul */
    h1 {
        text-align: center;
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Label */
    .section-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Preview */
    .preview-box {
        border: 1px dashed #CFCBC0;
        border-radius: 6px;
        min-height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #999999;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Hasil */
    .result-box {
        background-color: #F3F0E8;
        border-radius: 8px;
        padding: 18px;
        margin-top: 1.5rem;
    }

    .result-title {
        font-size: 0.9rem;
        color: #777777;
    }

    .result-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Hilangkan footer */
    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<h1>PENGENALAN KUNING TELUR PUYUH</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Unggah satu citra kuning telur puyuh untuk mengetahui
        tingkat kesegaran dan estimasi umur simpan.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Unggah Citra Kuning Telur Puyuh</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)


# ============================================================
# KONDISI PENYIMPANAN
# ============================================================

storage_condition = st.selectbox(
    "Kondisi penyimpanan",
    options=list(MAX_SHELF_LIFE_DAYS.keys())
)


# ============================================================
# PREVIEW
# ============================================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    st.image(
        image_bytes,
        caption="Citra yang diunggah",
        use_container_width=True
    )

    # ========================================================
    # PROSES
    # ========================================================

    if st.button(
        "PROSES",
        use_container_width=True,
        type="primary"
    ):

        with st.spinner("Memproses citra..."):

            models_dict = get_models()

            result = predict(
                models_dict,
                image_bytes,
                storage_condition
            )

        # ====================================================
        # HASIL
        # ====================================================

        st.markdown("### Hasil Pengenalan")

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-title">
                    Tingkat Kesegaran
                </div>

                <div class="result-value">
                    {result["predicted_class"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Perkiraan Lama Disimpan",
                f'{result["predicted_days_stored"]:.1f} hari'
            )

        with col2:
            st.metric(
                "Perkiraan Sisa Umur Simpan",
                f'{result["estimated_days_remaining"]:.1f} hari'
            )

        # ====================================================
        # PROBABILITAS
        # ====================================================

        st.markdown("#### Probabilitas Kelas")

        for cname in CLASS_NAMES:

            prob = result["class_probs"].get(
                cname,
                0.0
            )

            st.progress(
                prob,
                text=f"{cname} — {prob * 100:.1f}%"
            )

else:

    st.markdown(
        """
        <div class="preview-box">
            Pratinjau citra akan tampil di sini
        </div>
        """,
        unsafe_allow_html=True
    )

    st.button(
        "PROSES",
        disabled=True,
        use_container_width=False
    )


# ============================================================
# RESET
# ============================================================

if uploaded_file is not None:

    st.markdown("---")

    if st.button("RESET"):
        st.rerun()
