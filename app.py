"""
app.py
============================================================
Tampilan awal (GUI) sistem deteksi kesegaran & estimasi umur
simpan kuning telur puyuh - produk luaran skripsi (3.3 Desain Sistem).

Jalankan lokal:
    streamlit run app.py

Deploy via GitHub + Streamlit Community Cloud: lihat README.md
============================================================
"""

import streamlit as st
from utils import load_models, models_available, predict, MAX_SHELF_LIFE_DAYS, CLASS_NAMES

st.set_page_config(
    page_title="Deteksi Kesegaran Kuning Telur Puyuh",
    page_icon="🥚",
    layout="centered",
)

CLASS_COLOR = {
    "Segar": "#2E7D32",     # hijau
    "Sedang": "#F9A825",    # kuning/oranye
    "Menurun": "#C62828",   # merah
}
CLASS_EMOJI = {
    "Segar": "🟢",
    "Sedang": "🟡",
    "Menurun": "🔴",
}


@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


def render_header():
    st.title("🥚 Deteksi Kesegaran Kuning Telur Puyuh")
    st.caption(
        "Unggah citra kuning telur puyuh untuk mengetahui kelas kualitas "
        "(Segar / Sedang / Menurun) dan estimasi sisa umur simpan, "
        "menggunakan model EfficientNet-B0 + XGBoost + Bayesian Optimization."
    )


def render_sidebar():
    st.sidebar.header("Pengaturan")
    storage_condition = st.sidebar.radio(
        "Kondisi penyimpanan telur",
        options=list(MAX_SHELF_LIFE_DAYS.keys()),
        index=0,
        help="Digunakan untuk memperkirakan sisa umur simpan, "
             "berdasarkan rata-rata umur simpan maksimum pada kondisi tersebut.",
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tentang Sistem")
    st.sidebar.markdown(
        """

        **"Klasifikasi dan Estimasi Umur Simpan Kuning Telur Puyuh
        Menggunakan EfficientNet-XGBoost dengan Bayesian Optimization"**

        Alur prediksi:
        1. Citra kuning telur → *resize* 224×224 & normalisasi
        2. EfficientNet-B0 → ekstraksi *deep feature*
        3. XGBoost (hasil tuning Bayesian Optimization) →
           klasifikasi kelas kualitas & estimasi lama penyimpanan
        """
    )
    return storage_condition


def render_result(result: dict):
    pred_class = result["predicted_class"]
    color = CLASS_COLOR[pred_class]
    emoji = CLASS_EMOJI[pred_class]

    st.markdown("### Hasil Analisis")
    st.markdown(
        f"""
        <div style="padding:16px;border-radius:12px;background-color:{color}20;
                    border:2px solid {color};text-align:center;">
            <span style="font-size:2.2rem;">{emoji}</span>
            <div style="font-size:1.6rem;font-weight:700;color:{color};">
                Kualitas: {pred_class}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Perkiraan lama disimpan",
            f"{result['predicted_days_stored']:.1f} hari",
        )
    with col2:
        st.metric(
            "Perkiraan masih bertahan",
            f"{result['estimated_days_remaining']:.1f} hari",
            help=f"Asumsi umur simpan maksimum {result['max_shelf_life']} hari "
                 f"untuk kondisi penyimpanan yang dipilih.",
        )

    st.markdown("#### Probabilitas tiap kelas")
    for cname in CLASS_NAMES:
        prob = result["class_probs"].get(cname, 0.0)
        st.progress(prob, text=f"{cname}: {prob*100:.1f}%")

    if pred_class == "Menurun":
        st.error("Kualitas kuning telur sudah menurun — disarankan segera digunakan atau tidak dikonsumsi.")
    elif pred_class == "Sedang":
        st.warning("Kualitas mulai menurun — sebaiknya segera digunakan dalam waktu dekat.")
    else:
        st.success("Kualitas masih segar dan layak disimpan lebih lanjut.")


def main():
    render_header()
    storage_condition = render_sidebar()

    uploaded_file = st.file_uploader(
        "Unggah citra kuning telur puyuh (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption="Citra yang diunggah", use_container_width=True)

        if st.button("🔍 Analisis Citra", type="primary", use_container_width=True):
            with st.spinner("Menganalisis citra..."):
                models_dict = get_models()
                result = predict(models_dict, image_bytes, storage_condition)
            render_result(result)
    else:
        st.info("Silakan unggah citra kuning telur untuk memulai analisis.")


if __name__ == "__main__":
    main()
