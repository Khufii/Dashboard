"""
pipeline.py
============================================================
Pipeline analisis untuk skripsi:
"Klasifikasi dan Estimasi Umur Simpan Kuning Telur Puyuh
Menggunakan EfficientNet-XGBoost dengan Bayesian Optimization"

Mengikuti persis Bab III (3.2.5 - 3.2.8):
  - Skenario 1: EfficientNet-B0 baseline (end-to-end)
  - Skenario 2: EfficientNet-B0 (feature extractor) + XGBoost baseline
  - Skenario 3: EfficientNet-B0 (feature extractor) + XGBoost + Bayesian Optimization

Dua target:
  - Klasifikasi kelas kualitas/kesegaran (dari Yolk Index)
  - Regresi lama penyimpanan (hari)

------------------------------------------------------------
FORMAT DATA YANG DIHARAPKAN
------------------------------------------------------------
1) Folder citra kuning telur (hasil akuisisi, sudah di-crop ke ROI kuning
   telur atau masih penuh - resize otomatis ditangani kode ini).

2) File metadata CSV, contoh kolom:

   egg_id, image_path, storage_condition, storage_day, yolk_height_mm, yolk_diameter_mm

   - egg_id            : ID unik telur (WAJIB, dipakai untuk group split
                          agar tidak ada leakage - satu telur tidak boleh
                          muncul di train & test sekaligus)
   - image_path        : path menuju file citra kuning telur
   - storage_condition : "ruang" atau "refrigerator"
   - storage_day       : jumlah hari penyimpanan saat citra diambil (target regresi)
   - yolk_height_mm     : tinggi kuning telur (mm)
   - yolk_diameter_mm   : diameter kuning telur (mm)

   Yolk Index (YI = tinggi / diameter) dihitung otomatis oleh skrip ini,
   lalu dipetakan ke kelas kualitas (Segar / Sedang / Menurun) memakai
   ambang batas yang bisa kamu atur di CONFIG di bawah.

------------------------------------------------------------
INSTALASI
------------------------------------------------------------
pip install tensorflow xgboost scikit-learn scikit-optimize pandas numpy pillow matplotlib --break-system-packages

------------------------------------------------------------
CARA PAKAI
------------------------------------------------------------
python pipeline.py --metadata metadata.csv --image_root ./images --outdir ./hasil
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers

from xgboost import XGBClassifier, XGBRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Integer


# ============================================================
# 1. KONFIGURASI (silakan sesuaikan)
# ============================================================
CONFIG = {
    "image_size": (224, 224),          # sesuai 3.2.6.C
    "batch_size": 16,
    "epochs_baseline": 30,             # Skenario 1
    "epochs_feature_extractor": 15,    # fine-tuning ringan sebelum ekstraksi fitur
    "random_state": 42,
    "test_size": 0.15,
    "val_size": 0.15,
    # Ambang batas Yolk Index -> kelas kualitas (SESUAIKAN dengan literatur/protokol Anda)
    # YI >= yi_segar_min          -> "Segar"
    # yi_menurun_max <= YI < yi_segar_min -> "Sedang"
    # YI < yi_menurun_max         -> "Menurun"
    "yi_segar_min": 0.40,
    "yi_menurun_max": 0.30,
    "class_names": ["Menurun", "Sedang", "Segar"],
    # ruang pencarian Bayesian Optimization untuk XGBoost (3.2.7 Skenario 3)
    "bo_n_iter": 30,
    "bo_cv": 3,
}


# ============================================================
# 2. LABELING: Yolk Index -> kelas kualitas (3.2.4 & 3.2.5)
# ============================================================
def compute_yolk_index_and_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["yolk_index"] = df["yolk_height_mm"] / df["yolk_diameter_mm"]

    def to_class(yi):
        if yi >= cfg["yi_segar_min"]:
            return "Segar"
        elif yi < cfg["yi_menurun_max"]:
            return "Menurun"
        else:
            return "Sedang"

    df["quality_class"] = df["yolk_index"].apply(to_class)
    return df


# ============================================================
# 3. SPLIT DATA (group-aware, egg_id tidak boleh bocor antar split)
# ============================================================
def group_train_val_test_split(df: pd.DataFrame, cfg: dict):
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )
    trainval_idx, test_idx = next(gss1.split(df, groups=df["egg_id"]))
    df_trainval, df_test = df.iloc[trainval_idx], df.iloc[test_idx]

    val_ratio_within_trainval = cfg["val_size"] / (1 - cfg["test_size"])
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_ratio_within_trainval, random_state=cfg["random_state"]
    )
    train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval["egg_id"]))
    df_train = df_trainval.iloc[train_idx]
    df_val = df_trainval.iloc[val_idx]

    print(f"[SPLIT] train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


# ============================================================
# 4. DATA PIPELINE CITRA (3.2.6: ROI asumsi sudah dilakukan saat akuisisi/crop manual,
#    resize + normalisasi ditangani di sini)
# ============================================================
def load_and_preprocess_image(path, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = preprocess_input(img)  # normalisasi sesuai bobot pretrained EfficientNet
    return img


def make_tf_dataset(df, image_root, cfg, class_to_idx, training=False):
    paths = df["image_path"].apply(lambda p: os.path.join(image_root, p)).values
    class_idx = df["quality_class"].map(class_to_idx).values.astype(np.int32)
    storage_day = df["storage_day"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, class_idx, storage_day))

    def _load(path, cls, day):
        img = load_and_preprocess_image(path, cfg["image_size"])
        return img, {"cls_out": cls, "reg_out": day}

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=cfg["random_state"])
    ds = ds.batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 5A. SKENARIO 1: EfficientNet-B0 BASELINE (end-to-end, multi-output)
# ============================================================
def build_baseline_model(num_classes, image_size):
    base = EfficientNetB0(include_top=False, weights="imagenet",
                           input_shape=(*image_size, 3), pooling="avg")
    inputs = base.input
    x = base.output
    x = layers.Dropout(0.3)(x)

    cls_out = layers.Dense(num_classes, activation="softmax", name="cls_out")(x)
    reg_out = layers.Dense(1, activation="linear", name="reg_out")(x)

    model = models.Model(inputs=inputs, outputs=[cls_out, reg_out])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss={"cls_out": "sparse_categorical_crossentropy", "reg_out": "mse"},
        loss_weights={"cls_out": 1.0, "reg_out": 0.05},  # skala loss regresi vs klasifikasi
        metrics={"cls_out": "accuracy", "reg_out": "mae"},
    )
    return model, base


def run_scenario1_baseline(train_ds, val_ds, test_ds, num_classes, cfg):
    print("\n[SKENARIO 1] Training EfficientNet-B0 baseline ...")
    model, backbone = build_baseline_model(num_classes, cfg["image_size"])
    backbone.trainable = False  # transfer learning: bekukan backbone dulu

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs_baseline"], callbacks=callbacks)

    preds = model.predict(test_ds)
    cls_pred = np.argmax(preds[0], axis=1)
    reg_pred = preds[1].ravel()
    return model, cls_pred, reg_pred


# ============================================================
# 5B. FEATURE EXTRACTOR (dipakai Skenario 2 & 3)
# ============================================================
def build_feature_extractor(image_size):
    base = EfficientNetB0(include_top=False, weights="imagenet",
                           input_shape=(*image_size, 3), pooling="avg")
    return base


def extract_deep_features(extractor, df, image_root, cfg):
    paths = df["image_path"].apply(lambda p: os.path.join(image_root, p)).values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: load_and_preprocess_image(p, cfg["image_size"]),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)
    features = extractor.predict(ds, verbose=0)
    return features  # shape: (N, 1280) untuk EfficientNet-B0


# ============================================================
# 5C. SKENARIO 2: XGBoost BASELINE di atas deep features
# ============================================================
def run_scenario2_xgb_baseline(X_train, y_cls_train, y_reg_train, X_test, cfg):
    print("\n[SKENARIO 2] Training XGBoost baseline (default hyperparameter) ...")
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=cfg["random_state"],
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_cls_train)
    cls_pred = clf.predict(X_test)

    reg = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=cfg["random_state"],
    )
    reg.fit(X_train, y_reg_train)
    reg_pred = reg.predict(X_test)

    return clf, reg, cls_pred, reg_pred


# ============================================================
# 5D. SKENARIO 3: XGBoost + BAYESIAN OPTIMIZATION
# (parameter yang dioptimasi sesuai 3.2.7 Skenario 3: learning_rate,
#  max_depth, n_estimators, subsample)
# ============================================================
def run_scenario3_xgb_bayesopt(X_train, y_cls_train, y_reg_train, X_test, cfg):
    print("\n[SKENARIO 3] Bayesian Optimization tuning XGBoost ...")
    search_space = {
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "max_depth": Integer(3, 10),
        "n_estimators": Integer(100, 600),
        "subsample": Real(0.5, 1.0),
    }

    clf_base = XGBClassifier(random_state=cfg["random_state"], eval_metric="mlogloss")
    clf_search = BayesSearchCV(
        clf_base, search_space, n_iter=cfg["bo_n_iter"], cv=cfg["bo_cv"],
        scoring="f1_macro", random_state=cfg["random_state"], n_jobs=-1,
    )
    clf_search.fit(X_train, y_cls_train)
    print("  Best classifier params:", clf_search.best_params_)
    cls_pred = clf_search.predict(X_test)

    reg_base = XGBRegressor(random_state=cfg["random_state"])
    reg_search = BayesSearchCV(
        reg_base, search_space, n_iter=cfg["bo_n_iter"], cv=cfg["bo_cv"],
        scoring="neg_mean_absolute_error", random_state=cfg["random_state"], n_jobs=-1,
    )
    reg_search.fit(X_train, y_reg_train)
    print("  Best regressor params:", reg_search.best_params_)
    reg_pred = reg_search.predict(X_test)

    return clf_search, reg_search, cls_pred, reg_pred


# ============================================================
# 6. EVALUASI (3.2.8) -> mengisi Tabel 3.3 (klasifikasi) & Tabel 3.4 (regresi)
# ============================================================
def evaluate_classification(y_true, y_pred, class_names):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
    }


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }


# ============================================================
# 7. MAIN
# ============================================================
def main(args):
    cfg = CONFIG
    os.makedirs(args.outdir, exist_ok=True)

    # --- load & label ---
    df = pd.read_csv(args.metadata)
    df = compute_yolk_index_and_labels(df, cfg)
    class_to_idx = {c: i for i, c in enumerate(cfg["class_names"])}

    print("[INFO] Distribusi kelas kualitas:")
    print(df["quality_class"].value_counts())

    # --- split (group-aware by egg_id, sesuai 3.1) ---
    df_train, df_val, df_test = group_train_val_test_split(df, cfg)

    y_cls_test_true = df_test["quality_class"].map(class_to_idx).values
    y_reg_test_true = df_test["storage_day"].values.astype(np.float32)

    results = {}

    # ================= SKENARIO 1 =================
    train_ds = make_tf_dataset(df_train, args.image_root, cfg, class_to_idx, training=True)
    val_ds = make_tf_dataset(df_val, args.image_root, cfg, class_to_idx, training=False)
    test_ds = make_tf_dataset(df_test, args.image_root, cfg, class_to_idx, training=False)

    model_s1, cls_pred_s1, reg_pred_s1 = run_scenario1_baseline(
        train_ds, val_ds, test_ds, len(cfg["class_names"]), cfg
    )
    results["Skenario 1 - EfficientNet-B0 baseline"] = {
        "classification": evaluate_classification(y_cls_test_true, cls_pred_s1, cfg["class_names"]),
        "regression": evaluate_regression(y_reg_test_true, reg_pred_s1),
    }

    # ================= FEATURE EXTRACTION (untuk skenario 2 & 3) =================
    print("\n[INFO] Mengekstraksi deep features (EfficientNet-B0, global average pooling) ...")
    extractor = build_feature_extractor(cfg["image_size"])
    X_train = extract_deep_features(extractor, df_train, args.image_root, cfg)
    X_val = extract_deep_features(extractor, df_val, args.image_root, cfg)
    X_test = extract_deep_features(extractor, df_test, args.image_root, cfg)

    # gabungkan train+val untuk fitting XGBoost (val sudah tidak diperlukan terpisah
    # karena BayesSearchCV melakukan cross-validation sendiri)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_cls_trainval = np.concatenate([
        df_train["quality_class"].map(class_to_idx).values,
        df_val["quality_class"].map(class_to_idx).values,
    ])
    y_reg_trainval = np.concatenate([
        df_train["storage_day"].values.astype(np.float32),
        df_val["storage_day"].values.astype(np.float32),
    ])

    # ================= SKENARIO 2 =================
    _, _, cls_pred_s2, reg_pred_s2 = run_scenario2_xgb_baseline(
        X_trainval, y_cls_trainval, y_reg_trainval, X_test, cfg
    )
    results["Skenario 2 - EfficientNet-B0 + XGBoost baseline"] = {
        "classification": evaluate_classification(y_cls_test_true, cls_pred_s2, cfg["class_names"]),
        "regression": evaluate_regression(y_reg_test_true, reg_pred_s2),
    }

    # ================= SKENARIO 3 =================
    _, _, cls_pred_s3, reg_pred_s3 = run_scenario3_xgb_bayesopt(
        X_trainval, y_cls_trainval, y_reg_trainval, X_test, cfg
    )
    results["Skenario 3 - EfficientNet-B0 + XGBoost + Bayesian Optimization"] = {
        "classification": evaluate_classification(y_cls_test_true, cls_pred_s3, cfg["class_names"]),
        "regression": evaluate_regression(y_reg_test_true, reg_pred_s3),
    }

    # ================= SIMPAN HASIL (untuk Tabel 3.3 & 3.4 / Bab IV) =================
    summary_rows = []
    for scenario, r in results.items():
        c, g = r["classification"], r["regression"]
        summary_rows.append({
            "Skenario": scenario,
            "Accuracy": c["accuracy"],
            "Precision (macro)": c["precision_macro"],
            "Recall (macro)": c["recall_macro"],
            "F1 (macro)": c["f1_macro"],
            "MAE (hari)": g["mae"],
            "RMSE (hari)": g["rmse"],
            "R2": g["r2"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "ringkasan_evaluasi_tabel_3_3_3_4.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SELESAI] Ringkasan evaluasi -> {summary_path}")
    print(summary_df.to_string(index=False))

    detail_path = os.path.join(args.outdir, "hasil_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SELESAI] Detail lengkap (confusion matrix, classification report) -> {detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline analisis skripsi kuning telur puyuh")
    parser.add_argument("--metadata", required=True, help="Path ke file metadata.csv")
    parser.add_argument("--image_root", required=True, help="Folder root berisi citra")
    parser.add_argument("--outdir", default="./hasil", help="Folder untuk menyimpan hasil")
    args = parser.parse_args()
    main(args)
