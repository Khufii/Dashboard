"""
app.py
============================================================
Sistem Deteksi Kesegaran & Estimasi Umur Simpan
Kuning Telur Puyuh

Model:
EfficientNet-B0 + XGBoost + Bayesian Optimization

Jalankan:
    streamlit run app.py
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
    layout="centered",
    initial_sidebar_state="collapsed"
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       BACKGROUND
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(79,138,125,0.08),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(211,155,50,0.08),
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #FFFDF8 0%,
                #F7F4E9 100%
            );
    }

    .block-container {
        max-width: 800px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }


    /* ======================================================
       HEADER
       ====================================================== */

    .egg-animation {
        width: 75px;
        height: 92px;
        margin: 0 auto 15px auto;
        position: relative;

        animation:
            floating 3s ease-in-out infinite;
    }

    .egg-body {
        width: 65px;
        height: 82px;

        background:
            radial-gradient(
                circle at 35% 25%,
                #FFFFFF,
                #F4EFE1 70%
            );

        border-radius: 48% 48% 45% 45%;

        border: 2px solid #DDD6C5;

        position: absolute;
        left: 5px;
        top: 5px;

        box-shadow:
            0 10px 25px rgba(33,78,74,0.12);
    }

    .egg-yolk {
        width: 25px;
        height: 25px;

        background:
            radial-gradient(
                circle at 35% 30%,
                #FFE58A,
                #D99B2B
            );

        border-radius: 50%;

        position: absolute;

        left: 20px;
        top: 34px;

        box-shadow:
            0 0 15px rgba(211,155,50,0.25);
    }

    .egg-shine {
        width: 9px;
        height: 14px;

        background: rgba(255,255,255,0.8);

        border-radius: 50%;

        position: absolute;

        left: 17px;
        top: 17px;

        transform: rotate(25deg);
    }

    @keyframes floating {

        0%, 100% {
            transform: translateY(0px) rotate(0deg);
        }

        50% {
            transform: translateY(-8px) rotate(2deg);
        }
    }


    /* ======================================================
       TITLE
       ====================================================== */

    h1 {
        text-align: center;
        color: #214E4A !important;

        font-size: 2rem !important;
        font-weight: 750 !important;

        letter-spacing: 1.5px;

        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;

        color: #77766F;

        font-size: 0.94rem;

        line-height: 1.6;

        max-width: 650px;

        margin:
            0 auto 2.2rem auto;
    }


    /* ======================================================
       SECTION TITLE
       ====================================================== */

    .section-title {
        color: #214E4A;

        font-size: 0.98rem;

        font-weight: 650;

        margin-top: 0.5rem;

        margin-bottom: 0.55rem;
    }


    /* ======================================================
       UPLOAD
       ====================================================== */

    [data-testid="stFileUploader"] {

        background-color: #FFFFFF;

        border:
            2px dashed #B8CEC4;

        border-radius: 14px;

        padding: 8px;

        transition:
            all 0.25s ease;
    }

    [data-testid="stFileUploader"]:hover {

        border-color: #4F8A7D;

        background-color: #F8FCFA;

        box-shadow:
            0 5px 20px
            rgba(63,128,111,0.08);
    }

    [data-testid="stFileUploader"] button {

        border-radius: 8px !important;

        border:
            1px solid #C9D8D1 !important;

        background-color:
            #FFFFFF !important;

        color:
            #214E4A !important;
    }


    /* ======================================================
       SELECTBOX
       ====================================================== */

    div[data-baseweb="select"] > div {

        border-radius: 9px;

        border-color: #C8D8D1;

        background-color: #FFFFFF;
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {

        border-radius: 9px;

        border: none;

        background-color: #214E4A;

        color: #FFFFFF;

        font-weight: 600;

        min-height: 42px;

        transition:
            all 0.2s ease;

        box-shadow:
            0 4px 12px
            rgba(33,78,74,0.12);
    }

    .stButton > button:hover {

        background-color: #356F65;

        color: #FFFFFF;

        transform: translateY(-1px);

        box-shadow:
            0 6px 16px
            rgba(33,78,74,0.18);
    }


    /* ======================================================
       IMAGE
       ====================================================== */

    [data-testid="stImage"] {

        border-radius: 14px;

        overflow: hidden;

        margin-top: 1rem;

        margin-bottom: 1rem;

        box-shadow:
            0 5px 20px
            rgba(33,78,74,0.07);
    }


    /* ======================================================
       RESULT CARD
       ====================================================== */

    .result-card {

        border-radius: 14px;

        padding: 23px;

        text-align: center;

        margin-top: 1.3rem;

        margin-bottom: 1.3rem;

        animation:
            resultAppear 0.5s ease;
    }

    @keyframes resultAppear {

        from {
            opacity: 0;
            transform: translateY(10px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .result-label {

        color: #72746F;

        font-size: 0.9rem;

        margin-bottom: 5px;
    }

    .result-value {

        font-size: 1.9rem;

        font-weight: 750;
    }


    /* ======================================================
       METRIC
       ====================================================== */

    [data-testid="stMetric"] {

        background-color: #FFFFFF;

        border:
            1px solid #E2DED2;

        border-radius: 12px;

        padding: 15px;

        min-height: 105px;

        box-shadow:
            0 3px 12px
            rgba(33,78,74,0.04);
    }

    [data-testid="stMetricLabel"] {

        color: #6F756F;
    }

    [data-testid="stMetricValue"] {

        color: #214E4A;

        font-size: 1.35rem;
    }


    /* ======================================================
       PROGRESS
       ====================================================== */

    [data-testid="stProgressBar"] {

        margin-bottom: 5px;
    }

    [data-testid="stProgressBar"] > div > div {

        background-color: #4F8A7D;
    }


    /* ======================================================
       ALERT
       ====================================================== */

    [data-testid="stAlert"] {

        border-radius: 10px;
    }


    /* ======================================================
       SCANNING ANIMATION
       ====================================================== */

    .scan-box {

        position: relative;

        height: 80px;

        border-radius: 12px;

        background:
            linear-gradient(
                135deg,
                #EAF4EF,
                #F8F4DF
            );

        border:
            1px solid #C5D8CC;

        overflow: hidden;

        display: flex;

        align-items: center;

        justify-content: center;

        color: #356F65;

        font-weight: 600;

        margin:
            1rem 0;
    }

    .scan-line {

        position: absolute;

        width: 100%;

        height: 2px;

        background: #4F8A7D;

        box-shadow:
            0 0 10px #4F8A7D;

        animation:
            scan 1.5s linear infinite;
    }

    @keyframes scan {

        0% {
            top: 0%;
            opacity: 0;
        }

        15% {
            opacity: 1;
        }

        50% {
            opacity: 1;
        }

        85% {
            opacity: 1;
        }

        100% {
            top: 100%;
            opacity: 0;
        }
    }


    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {

        border-color:
            #DDD9CB;

        margin-top: 1.5rem;

        margin-bottom: 1.5rem;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


# ============================================================
# ANIMATED EGG HEADER
# ============================================================

st.markdown(
    """
    <div class="egg-animation">

        <div class="egg-body">

            <div class="egg-shine"></div>

            <div class="egg-yolk"></div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


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
    """
    <div class="section-title">
        Unggah Citra Kuning Telur Puyuh
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload citra",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
    help="Format yang didukung: PNG, JPG, JPEG"
)


# ============================================================
# KONDISI PENYIMPANAN
# ============================================================

if uploaded_file is not None:

    st.markdown(
        """
        <div class="section-title">
            Kondisi Penyimpanan
        </div>
        """,
        unsafe_allow_html=True
    )

    storage_condition = st.selectbox(
        "Pilih kondisi penyimpanan",
        options=list(MAX_SHELF_LIFE_DAYS.keys()),
        label_visibility="collapsed"
    )

else:

    storage_condition = list(
        MAX_SHELF_LIFE_DAYS.keys()
    )[0]


# ============================================================
# PREVIEW & PROCESS
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

    process = st.button(
        "🔍  PROSES CITRA",
        type="primary",
        use_container_width=True
    )


    if process:

        # ====================================================
        # ANIMASI SCANNING
        # ====================================================

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


        # ====================================================
        # PREDIKSI
        # ====================================================

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
                margin-top:1.8rem;
                margin-bottom:0.8rem;
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
        # HASIL KELAS
        # ====================================================

        st.markdown(
            f"""
            <div class="result-card"
                 style="
                    background:
                        linear-gradient(
                            135deg,
                            {background},
                            #FFFFFF
                        );

                    border:
                        1.5px solid {color};
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
        # ESTIMASI UMUR SIMPAN
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
                margin-top:1.5rem;
                margin-bottom:0.8rem;
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
        # INFORMASI
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
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        """
        <div style="
            height:170px;

            border:
                1.5px dashed #CFCBC0;

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
        "🔍  PROSES CITRA",
        disabled=True
    )


# ============================================================
# RESET
# ============================================================

if uploaded_file is not None:

    st.markdown("---")

    if st.button("↻  RESET"):

        st.rerun()
