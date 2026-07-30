# -*- coding: utf-8 -*-
"""
train_cnn_128_local.py - CNN supervise (conforme/defectueux), 15 categories
MVTec AD, resolution 128x128. Meme architecture que la version 64x64
(Flatten, pas GAP) pour isoler l'effet de la resolution seule.
Genere aussi les heatmaps de matrice de confusion.
"""
import json, time
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

tf.get_logger().setLevel("ERROR")

# ── Chemins ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(r"C:\Users\paulf\avr26_bmle_ds_anomalies")
ARCHIVE_ROOT = Path(r"C:\Users\paulf\OneDrive\Documents\Claude\Projects\Formation Machine Learning Engineer\03_Projets\DS\01_Data\archive")
CACHE_DIR = ARCHIVE_ROOT.parent / "cnn_cache_local"   # hors du depot git
MODELS_DIR = SCRIPT_DIR / "models"
FIGS_DIR = SCRIPT_DIR / "figures"
RESULTS_DIR = SCRIPT_DIR / "results"
for d in (CACHE_DIR, MODELS_DIR, FIGS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
TOTAL_EPOCHS = 30

CATEGORIES = sorted(d.name for d in ARCHIVE_ROOT.iterdir() if d.is_dir() and (d / "train").exists())
print(f"{len(CATEGORIES)} categories detectees : {CATEGORIES}", flush=True)

# ── Chargement des images, avec cache disque (attention : ~1 Go en 128x128) ─
CACHE_FILE = CACHE_DIR / f"all_categories_{IMG_SIZE}.npz"

if CACHE_FILE.exists():
    d = np.load(CACHE_FILE)
    X, y = d["X"], d["y"]
    print(f"Images chargees depuis le cache ({len(X)} images).", flush=True)
else:
    paths, labels = [], []
    for cat in CATEGORIES:
        for split in ["train", "test"]:
            for sub in sorted((ARCHIVE_ROOT / cat / split).iterdir()):
                lab = 0 if sub.name == "good" else 1
                for p in sorted(sub.glob("*.png")):
                    paths.append(p)
                    labels.append(lab)
    print(f"{len(paths)} images a charger ({sum(labels)} defauts)...", flush=True)

    t0 = time.time()
    imgs = []
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(paths)} images redimensionnees...", flush=True)
    X = np.stack(imgs).astype("float32") / 255.0
    y = np.array(labels)
    np.savez(CACHE_FILE, X=X, y=y)
    print(f"Chargement termine en {time.time()-t0:.1f}s, cache sauvegarde.", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"train={len(X_train)} test={len(X_test)}", flush=True)

raw_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
class_weight = {0: 1.0, 1: float(raw_ratio)}
print("class_weight:", class_weight, flush=True)

# ── Architecture (identique a la version 64x64) ─────────────────────────────
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
    x = Flatten()(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    m = Model(inputs, outputs, name="cnn_15cat_128")
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

MODEL_PATH = MODELS_DIR / "cnn_binary_15cat_128.keras"
HIST_PATH = RESULTS_DIR / "cnn_15cat_128_history.json"

model = build_model()
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ModelCheckpoint(MODELS_DIR / "cnn_binary_15cat_128_best.keras", monitor="val_loss", save_best_only=True),
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

with open(RESULTS_DIR / "cnn_15cat_128_metrics.json", "w") as f:
    json.dump({
        "categories": CATEGORIES, "img_size": IMG_SIZE,
        "n_train": len(X_train), "n_test": len(X_test),
        "roc_auc": auc, "confusion_matrix": cm_.tolist(),
        "classification_report": report,
    }, f, indent=2, ensure_ascii=False)

# Courbe accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch"); plt.legend(); plt.title("CNN 15 categories (128x128) - accuracy")
plt.tight_layout()
plt.savefig(FIGS_DIR / "cnn_15cat_128_accuracy.png", dpi=120)
plt.close()

# Heatmap matrice de confusion
plt.figure(figsize=(5, 4))
sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Conforme", "Defectueux"], yticklabels=["Conforme", "Defectueux"])
plt.xlabel("Predit"); plt.ylabel("Reel")
plt.title(f"CNN 15 categories (128x128)\nROC-AUC={auc:.3f}")
plt.tight_layout()
plt.savefig(FIGS_DIR / "cnn_15cat_128_confusion_heatmap.png", dpi=120)
plt.close()

# ── Grad-CAM ─────────────────────────────────────────────────────────────
def grad_cam(image, model, layer_name):
    layer = model.get_layer(layer_name)
    grad_model = Model(inputs=model.input, outputs=[layer.output, model.output])
    image_b = tf.expand_dims(image, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_b)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap_resized = tf.image.resize(heatmap[..., None], (IMG_SIZE, IMG_SIZE)).numpy()
    heatmap_resized = np.squeeze(heatmap_resized, axis=-1)
    heatmap_colored = cm.jet(heatmap_resized)[..., :3]
    superimposed = heatmap_colored * 0.5 + image
    return np.clip(superimposed, 0, 1), float(predictions[0, 0])

defect_idxs = np.where(y_test == 1)[0][:4]
good_idxs = np.where(y_test == 0)[0][:2]
show_idxs = np.concatenate([good_idxs, defect_idxs])

fig, axes = plt.subplots(2, len(show_idxs), figsize=(3.5 * len(show_idxs), 7))
for i, idx in enumerate(show_idxs):
    gradcam_img, pred = grad_cam(X_test[idx], model, "Conv_3")
    axes[0, i].imshow(X_test[idx])
    axes[0, i].set_title(f"Original ({'defaut' if y_test[idx] else 'ok'})")
    axes[0, i].axis("off")
    axes[1, i].imshow(gradcam_img)
    axes[1, i].set_title(f"Grad-CAM (pred={pred:.2f})")
    axes[1, i].axis("off")
plt.tight_layout()
plt.savefig(FIGS_DIR / "cnn_15cat_128_gradcam_examples.png", dpi=120)
plt.close()

print("\nTermine : modele, historique, metriques, figures sauvegardes.", flush=True)