"""
utils.py
============================================================
Fungsi bantu untuk memuat model dan melakukan prediksi pada
aplikasi Streamlit deteksi kesegaran kuning telur puyuh.

Mengikuti alur Bab III skripsi:
  Citra -> EfficientNet-B0 (feature extractor) -> XGBoost
  (Classifier untuk kelas kualitas, Regressor untuk lama penyimpanan)

Skrip ini didesain agar TETAP BISA DIJALANKAN walau file model
belum ada (mode DEMO dengan prediksi acak/heuristik), supaya
tampilan GUI bisa langsung dites sebelum model hasil training
(pipeline.py) tersedia. Setelah training selesai, cukup taruh
file model di folder models/ sesuai nama di bawah, aplikasi akan
otomatis memakai model asli.
============================================================
"""

import os
import io
import numpy as np
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "efficientnet_feature_extractor.h5")
XGB_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "xgb_classifier.json")
XGB_REGRESSOR_PATH = os.path.join(MODEL_DIR, "xgb_regressor.json")

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Menurun", "Sedang", "Segar"]

# Aturan bisnis sederhana: perkiraan total umur simpan (hari) sampai
# kondisi "Menurun", tergantung kondisi penyimpanan yang dipilih user.
# SESUAIKAN angka ini dengan hasil penelitian kamu (mis. hari saat rata-rata
# YI sampel menyentuh ambang batas "Menurun" pada tiap kondisi).
MAX_SHELF_LIFE_DAYS = {
    "Suhu ruang": 10,
    "Refrigerator": 30,
}


def _lazy_import_tf():
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return tf, preprocess_input


def models_available() -> bool:
    """Cek apakah ketiga file model hasil training sudah tersedia."""
    return (
        os.path.exists(FEATURE_EXTRACTOR_PATH)
        and os.path.exists(XGB_CLASSIFIER_PATH)
        and os.path.exists(XGB_REGRESSOR_PATH)
    )


def load_models():
    """
    Memuat feature extractor (EfficientNet-B0) + XGBoost classifier &
    regressor hasil training (lihat pipeline.py, Skenario 3).
    Mengembalikan None jika model belum tersedia -> aplikasi otomatis
    jalan di mode DEMO.
    """
    if not models_available():
        return None

    tf, _ = _lazy_import_tf()
    from xgboost import XGBClassifier, XGBRegressor

    feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH, compile=False)

    clf = XGBClassifier()
    clf.load_model(XGB_CLASSIFIER_PATH)

    reg = XGBRegressor()
    reg.load_model(XGB_REGRESSOR_PATH)

    return {"extractor": feature_extractor, "clf": clf, "reg": reg}


def preprocess_uploaded_image(uploaded_bytes: bytes) -> np.ndarray:
    """Resize (224x224) + normalisasi sesuai bobot pretrained EfficientNet."""
    img = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype(np.float32)

    try:
        _, preprocess_input = _lazy_import_tf()
        arr = preprocess_input(arr)
    except Exception:
        arr = (arr / 127.5) - 1.0  # fallback normalisasi sederhana

    return np.expand_dims(arr, axis=0)


def predict(models_dict, uploaded_bytes: bytes, storage_condition: str):
    """
    Mengembalikan dict:
      {
        "predicted_class": str,
        "class_probs": {kelas: prob},
        "predicted_days_stored": float,   # estimasi telur SUDAH disimpan berapa hari
        "estimated_days_remaining": float # estimasi MASIH BERTAHAN berapa hari
      }
    """
    max_shelf_life = MAX_SHELF_LIFE_DAYS.get(storage_condition, 10)

    if models_dict is None:
        # ---------- MODE DEMO (belum ada model asli) ----------
        # Heuristik sederhana berbasis kecerahan & saturasi rata-rata citra,
        # HANYA untuk keperluan pratinjau UI, BUKAN hasil ilmiah.
        img = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        brightness = arr.mean()
        seed = int(brightness * 1000) % 1000
        rng = np.random.RandomState(seed)

        probs = rng.dirichlet(alpha=[2, 2, 2])
        pred_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[pred_idx]
        predicted_days_stored = float(rng.uniform(0, max_shelf_life))
    else:
        x = preprocess_uploaded_image(uploaded_bytes)
        features = models_dict["extractor"].predict(x, verbose=0)

        probs_arr = models_dict["clf"].predict_proba(features)[0]
        pred_idx = int(np.argmax(probs_arr))
        predicted_class = CLASS_NAMES[pred_idx]
        probs = probs_arr

        predicted_days_stored = float(models_dict["reg"].predict(features)[0])
        predicted_days_stored = max(0.0, predicted_days_stored)

    class_probs = {c: float(p) for c, p in zip(CLASS_NAMES, probs)}
    estimated_days_remaining = max(0.0, max_shelf_life - predicted_days_stored)

    return {
        "predicted_class": predicted_class,
        "class_probs": class_probs,
        "predicted_days_stored": predicted_days_stored,
        "estimated_days_remaining": estimated_days_remaining,
        "max_shelf_life": max_shelf_life,
    }
