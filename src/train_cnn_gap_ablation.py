# -*- coding: utf-8 -*-
"""
train_cnn_gap_ablation.py - Ablation : meme CNN 15 categories mais avec
GlobalAveragePooling2D au lieu de Flatten (recommandation du mentor).
Reutilise le cache d'images de train_cnn_all_categories.py (meme split
train/test, random_state=42, pour une comparaison equitable).
"""
import json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

tf.get_logger().setLevel("ERROR")

REPO_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_DIR / "data" / "cnn_cache"
MODELS_DIR = REPO_DIR / "models"
FIGS_DIR = REPO_DIR / "figures"
RESULTS_DIR = REPO_DIR / "results"

IMG_SIZE = 64
TOTAL_EPOCHS = 30
CACHE_FILE = CACHE_DIR / f"all_categories_{IMG_SIZE}.npz"

d = np.load(CACHE_FILE)
X, y = d["X"], d["y"]
print(f"Images chargees depuis le cache ({len(X)} images).", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"train={len(X_train)} test={len(X_test)}", flush=True)

raw_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
class_weight = {0: 1.0, 1: float(raw_ratio)}
print("class_weight:", class_weight, flush=True)

def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = Conv2D(16, (3, 3), padding="same", name="Conv_1")(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (3, 3), padding="same", name="Conv_2")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), padding="same", name="Conv_3")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    m = Model(inputs, outputs, name="cnn_15cat_gap")
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

MODEL_PATH = MODELS_DIR / "cnn_binary_15cat_gap.keras"
HIST_PATH = RESULTS_DIR / "cnn_15cat_gap_history.json"

model = build_model()
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ModelCheckpoint(MODELS_DIR / "cnn_binary_15cat_gap_best.keras", monitor="val_loss", save_best_only=True),
]
t0 = time.time()
history = model.fit(
    X_train, y_train, epochs=TOTAL_EPOCHS, batch_size=32,
    validation_split=0.15, class_weight=class_weight, verbose=2, callbacks=callbacks,
)
print(f"Entrainement termine en {time.time()-t0:.1f}s", flush=True)
model.save(MODEL_PATH)
json.dump(history.history, open(HIST_PATH, "w"))

y_proba = model.predict(X_test, verbose=0).ravel()
y_pred = (y_proba > 0.5).astype(int)
auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred, target_names=["Conforme", "Defectueux"])
cm_ = confusion_matrix(y_test, y_pred)

print(f"\nROC-AUC={auc:.4f}", flush=True)
print(report, flush=True)
print("Matrice de confusion:\n", cm_, flush=True)

with open(RESULTS_DIR / "cnn_15cat_gap_metrics.json", "w") as f:
    json.dump({
        "reduction": "gap", "img_size": IMG_SIZE,
        "n_train": len(X_train), "n_test": len(X_test),
        "roc_auc": auc, "confusion_matrix": cm_.tolist(),
        "classification_report": report,
    }, f, indent=2, ensure_ascii=False)

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch"); plt.legend(); plt.title("CNN 15 categories (GAP) - accuracy")
plt.tight_layout()
plt.savefig(FIGS_DIR / "cnn_15cat_gap_accuracy.png", dpi=120)
plt.close()

print("\nAblation GAP terminee.", flush=True)