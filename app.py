"""
app.py
============================================================
Sistem Deteksi Kesegaran & Estimasi Umur Simpan
Kuning Telur Puyuh

Model:
EfficientNet-B0 + XGBoost + Bayesian Optimization
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
    page_title="Deteksi Kuning Telur Puyuh",
    page_icon="🥚",
    layout="centered"
)


# ============================================================
# WARNA
# ============================================================

CLASS_COLOR = {
    "Segar": "#3F806F",
    "Sedang": "#D39B32",
    "Menurun": "#B85C5C"
}

CLASS_BG = {
    "Segar": "#EAF4EF",
    "Sedang": "#FBF3DD",
    "Menurun": "#F8EAEA"
}


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* ==============================
   BACKGROUND
   ============================== */

.stApp {
    background-color: #FFFDF8;
}

.block-container {
    max-width: 800px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}


/* ==============================
   HEADER
   ============================== */

h1 {
    text-align: center;
    color: #214E4A !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px;
}

.subtitle {
    text-align: center;
    color: #77766F;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 2rem;
}


/* ==============================
   TELUR ANIMASI
   ============================== */

.egg {
    font-size: 60px;
    text-align: center;
    animation: floating 3s ease-in-out infinite;
    margin-bottom: 5px;
}

@keyframes floating {

    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-8px);
    }

    100% {
        transform: translateY(0px);
    }

}


/* ==============================
   SECTION
   ============================== */

.section-title {
    color: #214E4A;
    font-size: 1rem;
    font-weight: 650;
    margin-bottom: 8px;
}


/* ==============================
   UPLOAD BOX
   ============================== */

[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 2px dashed #B8CEC4;
    border-radius: 14px;
    padding: 8px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #4F8A7D;
}


/* ==============================
   BUTTON
   ============================== */

.stButton > button {
    background-color: #214E4A;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #356F65;
    color: white;
}


/* ==============================
   IMAGE
   ============================== */

[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
}


/* ==============================
   RESULT
   ============================== */

.result-card {
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin-top: 15px;
    margin-bottom: 18px;
    animation: resultAppear 0.5s ease;
}

.result-label {
    color: #72746F;
    font-size: 0.9rem;
}

.result-value {
    font-size: 1.9rem;
    font-weight: 750;
}

@keyframes resultAppear {

    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }

}


/* ==============================
   METRIC
   ============================== */

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2DED2;
    border-radius: 12px;
    padding: 15px;
}


/* ==============================
   SCAN ANIMATION
   ============================== */

.scan-box {
    position: relative;
    height: 70px;
    border-radius: 12px;
    background-color: #EAF4EF;
    border: 1px solid #C5D8CC;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #356F65;
    font-weight: 600;
    margin: 15px 0;
}

.scan-line {
    position: absolute;
    width: 100%;
    height: 2px;
    background-color: #4F8A7D;
    box-shadow: 0 0 10px #4F8A7D;
    animation: scan 1.5s linear infinite;
}

@keyframes scan {

    0% {
        top: 0%;
        opacity: 0;
    }

    20% {
        opacity: 1;
    }

    50% {
        opacity: 1;
    }

    80% {
        opacity: 1;
    }

    100% {
        top: 100%;
        opacity: 0;
    }

}


/* ==============================
   DIVIDER
   ============================== */

hr {
    border-color: #DDD9CB;
}


/* ==============================
   FOOTER
   ============================== */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="egg">🥚</div>',
    unsafe_allow_html=True
)

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
    "Upload citra",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)


# ============================================================
# JIKA BELUM UPLOAD
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div style="
            height:160px;
            border:1.5px dashed #CFCBC0;
            border-radius:14px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#999999;
            background:#FFFDF9;
            margin-top:15px;
            margin-bottom:18px;
        ">
            Pratinjau citra akan tampil di sini
        </div>
        """,
        unsafe_allow_html=True
    )

    st.button(
        "🔍 PROSES CITRA",
        disabled=True
    )


# ============================================================
# JIKA SUDAH UPLOAD
# ============================================================

else:

    image_bytes = uploaded_file.getvalue()

    # ========================================================
    # PREVIEW
    # ========================================================

    st.image(
        image_bytes,
        caption="Citra yang diunggah",
        use_container_width=True
    )


    # ========================================================
    # KONDISI PENYIMPANAN
    # ========================================================

    st.markdown(
        '<div class="section-title">Kondisi Penyimpanan</div>',
        unsafe_allow_html=True
    )

    storage_condition = st.selectbox(
        "Kondisi penyimpanan",
        options=list(MAX_SHELF_LIFE_DAYS.keys()),
        label_visibility="collapsed"
    )


    # ========================================================
    # BUTTON PROSES
    # ========================================================

    process = st.button(
        "🔍 PROSES CITRA",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # PROSES
    # ========================================================

    if process:

        scan_placeholder = st.empty()

        scan_placeholder.markdown(
            """
            <div class="scan-box">

                <div class="scan-line"></div>

                🔍 Menganalisis citra...

            </div>
            """,
            unsafe_allow_html=True
        )


        try:

            models_dict = get_models()

            result = predict(
                models_dict,
                image_bytes,
                storage_condition
            )

        except Exception as e:

            scan_placeholder.empty()

            st.error(
                "Terjadi kesalahan saat memproses citra."
            )

            st.exception(e)

            st.stop()


        scan_placeholder.empty()


        # ====================================================
        # HASIL
        # ====================================================

        st.markdown(
            """
            <h3 style="
                color:#214E4A;
                margin-top:25px;
                margin-bottom:10px;
            ">
                Hasil Pengenalan
            </h3>
            """,
            unsafe_allow_html=True
        )


        pred_class = result["predicted_class"]

        color = CLASS_COLOR.get(
            pred_class,
            "#214E4A"
        )

        background = CLASS_BG.get(
            pred_class,
            "#F3F4EF"
        )


        # ====================================================
        # CLASS RESULT
        # ====================================================

        st.markdown(
            f"""
            <div class="result-card"
                 style="
                    background: linear-gradient(
                        135deg,
                        {background},
                        #FFFFFF
                    );
                    border: 1.5px solid {color};
                 ">

                <div class="result-label">
                    Tingkat Kesegaran
                </div>

                <div class="result-value"
                     style="color:{color};">

                    {pred_class}

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # ESTIMASI
        # ====================================================

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

        st.markdown(
            """
            <h4 style="
                color:#214E4A;
                margin-top:25px;
                margin-bottom:10px;
            ">
                Probabilitas Kelas
            </h4>
            """,
            unsafe_allow_html=True
        )


        for cname in CLASS_NAMES:

            prob = result["class_probs"].get(
                cname,
                0.0
            )

            st.progress(
                float(prob),
                text=f"{cname} — {prob * 100:.1f}%"
            )


        # ====================================================
        # INFO
        # ====================================================

        st.caption(
            f"Kondisi penyimpanan: {storage_condition} "
            f"| Umur simpan maksimum: "
            f'{result["max_shelf_life"]} hari'
        )


        # ====================================================
        # KETERANGAN
        # ====================================================

        if pred_class == "Segar":

            st.success(
                "Kuning telur teridentifikasi dalam kondisi segar."
            )

        elif pred_class == "Sedang":

            st.warning(
                "Kuning telur teridentifikasi mengalami "
                "penurunan kesegaran."
            )

        elif pred_class == "Menurun":

            st.error(
                "Kuning telur teridentifikasi dalam kondisi "
                "kesegaran yang menurun."
            )


# ============================================================
# RESET
# ============================================================

if uploaded_file is not None:

    st.markdown("---")

    if st.button("↻ RESET"):

        st.rerun()
