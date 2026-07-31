# -*- coding: utf-8 -*-
# Identification du TYPE de defaut par transfer learning sur MVTec AD
# Ici on ne demande plus juste "defaut oui/non" mais "quel type de defaut ?".
# C'est donc de la classification multi-classe (good + les differents types de defaut).
#
# Difference avec la detection :
#   - detection      = binaire   -> 1 neurone sigmoid
#   - identification = multi-classe -> N neurones softmax
#
# Comme pour la detection, j'entraine UN modele par categorie (bottle, screw, carpet...).
# Fichier en cellules "# %%".
#
# Remarque importante : il y a tres peu d'images par type de defaut (environ 20).
# Par categorie ca reste jouable, mais les scores par classe reposent sur peu d'images
# de test -> toujours afficher le "support" (n) a cote du F1.

# %% Imports et reglages de depart
import json
import collections
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)

np.random.seed(42)
tf.random.set_seed(42)
plt.rcParams["figure.dpi"] = 110

# chemin vers les donnees (a changer si besoin)
DATA_ROOT = Path("../dataBase_mvtec")

# Mode rapide :
#   True  = test rapide sur 1 modele (peu d'epochs), benchmark saute
#   False = benchmark complet (4 backbones x categories)
MODE_RAPIDE = True

CATEGORIE    = "bottle"        # bottle (3 defauts) ; screw / carpet (5 defauts)
BACKBONE     = "resnet50"      # resnet50 / efficientnetb0 / vgg16 / mobilenetv2
INCLURE_GOOD = True            # True = good + les types de defaut ; False = types seulement
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
LR_P1        = 1e-3
LR_P2        = 1e-5           # 100x plus petit (important)
CAP_GOOD     = 80            # nombre max d'images good ; None = pas de limite
UNFREEZE     = 0.20
MIN_PAR_CLASSE = 6           # une classe avec moins d'images que ca est ecartee

# moins d'epochs en mode rapide
if MODE_RAPIDE:
    EPOCHS_P1, EPOCHS_P2 = 10, 8
else:
    EPOCHS_P1, EPOCHS_P2 = 25, 20

OUT_DIR = Path(__file__).parent / "resultats_identification" if "__file__" in dir() else Path("resultats_identification")
OUT_DIR.mkdir(exist_ok=True)

print("TensorFlow", tf.__version__)
print("GPU :", tf.config.list_physical_devices("GPU") or "aucun (CPU)")
assert DATA_ROOT.exists(), f"Chemin introuvable : {DATA_ROOT}"


# %% [markdown]
# ## 0. Fonctions communes (backbone + augmentation)
#
# Je remets ici les memes fonctions que dans le script de detection, pour que ce
# fichier soit autonome (pas besoin d'importer un autre module).

# %% Backbone et couche d'augmentation
def get_backbone(nom, input_shape):
    nom = nom.lower()
    if nom == "resnet50":
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        base = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    elif nom == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    elif nom == "vgg16":
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input
        base = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    elif nom == "mobilenetv2":
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        raise ValueError(f"Backbone inconnu : {nom}")
    return base, preprocess_input


def couche_augmentation(seed=42):
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.03, fill_mode="reflect", seed=seed),
        layers.RandomZoom(0.10, fill_mode="reflect", seed=seed),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed),
    ], name="augmentation")


# %% [markdown]
# ## 1. Liste des images multi-classe + decoupage
#
# Les classes sont les sous-dossiers de test/ (un par type de defaut), plus good si
# INCLURE_GOOD. On ecarte les classes trop petites, et on plafonne good (sinon il
# ecrase les types de defaut). Puis decoupage stratifie 70/15/15.

# %% Construction du manifeste multi-classe
def construit_manifeste_multi(data_root, categorie, inclure_good=True,
                              cap_good=None, min_par_classe=6, seed=42):
    cat = Path(data_root) / categorie
    par_classe = {}   # nom de classe -> liste de chemins

    # les defauts : chaque sous-dossier de test/ sauf good
    for sous in sorted((cat / "test").iterdir()):
        if not sous.is_dir() or sous.name == "good":
            continue
        imgs = [str(p) for p in sorted(sous.glob("*.png"))]
        if imgs:
            par_classe[sous.name] = imgs

    # good : on prend train/good + test/good, puis on plafonne
    if inclure_good:
        good = [str(p) for p in sorted((cat / "train" / "good").glob("*.png"))]
        good += [str(p) for p in sorted((cat / "test" / "good").glob("*.png"))]
        rng = np.random.default_rng(seed)
        if cap_good is not None and len(good) > cap_good:
            good = list(rng.choice(good, cap_good, replace=False))
        par_classe["good"] = good

    # on enleve les classes trop petites (impossible a decouper)
    ecartees = [c for c, v in par_classe.items() if len(v) < min_par_classe]
    for c in ecartees:
        print(f"  [!] classe '{c}' ecartee ({len(par_classe[c])} images < {min_par_classe})")
        del par_classe[c]

    # on transforme les noms de classe en numeros
    class_names = sorted(par_classe.keys())
    cls2idx = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []
    for c in class_names:
        paths += par_classe[c]
        labels += [cls2idx[c]] * len(par_classe[c])
    return np.array(paths), np.array(labels, int), class_names


def split_stratifie(paths, labels, test_size=0.15, val_size=0.15, seed=42):
    # 1) on met de cote le test
    p_rest, p_te, y_rest, y_te = train_test_split(
        paths, labels, test_size=test_size, random_state=seed, stratify=labels)
    # 2) on separe train / val sur le reste
    val_ratio = val_size / (1.0 - test_size)
    p_tr, p_val, y_tr, y_val = train_test_split(
        p_rest, y_rest, test_size=val_ratio, random_state=seed, stratify=y_rest)
    return (p_tr, y_tr), (p_val, y_val), (p_te, y_te)


paths, labels, CLASSES = construit_manifeste_multi(
    DATA_ROOT, CATEGORIE, INCLURE_GOOD, CAP_GOOD, MIN_PAR_CLASSE)
N_CLASSES = len(CLASSES)
(p_tr, y_tr), (p_val, y_val), (p_te, y_te) = split_stratifie(paths, labels)

print(f"Categorie : {CATEGORIE}  |  {N_CLASSES} classes : {CLASSES}")
print("Repartition :", dict(collections.Counter(labels)))
for nom, y in [("train", y_tr), ("val", y_val), ("test", y_te)]:
    print(f"  {nom:5} : {len(y):4}")


# %% [markdown]
# ## 2. Charger les images avec tf.data (labels one-hot)
#
# Difference avec la detection : le label devient un vecteur one-hot de taille
# N_CLASSES (au lieu d'un simple 0 ou 1).

# %% Dataset multi-classe
base_model, preprocess_input = get_backbone(BACKBONE, IMG_SIZE + (3,))

def make_ds_multi(paths, labels, shuffle=False, batch=BATCH_SIZE):
    def charge_une(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        img = preprocess_input(img)
        return img, tf.one_hot(label, N_CLASSES)          # label one-hot
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(charge_une, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

ds_tr  = make_ds_multi(p_tr, y_tr, shuffle=True)
ds_val = make_ds_multi(p_val, y_val)
ds_te  = make_ds_multi(p_te, y_te)

# poids des classes (il y a souvent plus de good que de defauts)
poids = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_tr)
class_weight = {i: float(w) for i, w in enumerate(poids)}
print("class_weight :", {CLASSES[i]: round(w, 2) for i, w in class_weight.items()})


# %% [markdown]
# ## 3. Modele avec une tete softmax a N classes
#
# backbone gele -> GlobalAveragePooling -> Dropout -> Dense(N_CLASSES, softmax)
# Perte : categorical_crossentropy (labels one-hot).

# %% Construction du modele + phase 1
# Comme pour la detection, je cree la couche d'augmentation UNE SEULE FOIS et je la
# reutilise. Sinon le benchmark devient tres lent sur CPU (TensorFlow recompile a chaque
# nouveau modele).
augmentation = couche_augmentation()

def construit_modele_multi(base, n_classes, dropout=0.3):
    inp = keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = augmentation(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="pred")(x)   # softmax
    return Model(inp, out, name=f"{BACKBONE}_{CATEGORIE}_multi")

# La demo sur une seule categorie ne tourne qu'en mode rapide.
model, hist1, hist2, f1m = None, None, None, None
if MODE_RAPIDE:
    base_model.trainable = False
    model = construit_modele_multi(base_model, N_CLASSES)
    model.compile(optimizer=keras.optimizers.Adam(LR_P1),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    cb1 = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)]
    hist1 = model.fit(ds_tr, validation_data=ds_val, epochs=EPOCHS_P1,
                      class_weight=class_weight, callbacks=cb1, verbose=1)
else:
    print("[MODE COMPLET] Demo mono-categorie sautee -> voir le benchmark plus bas.")


# %% Phase 2 : fine-tuning (LR / 100)
if MODE_RAPIDE:
    base_model.trainable = True
    n_gel = int(len(base_model.layers) * (1 - UNFREEZE))
    for couche in base_model.layers[:n_gel]:
        couche.trainable = False
    for couche in base_model.layers:
        if isinstance(couche, layers.BatchNormalization):
            couche.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(LR_P2),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"Couches degelees : {len(base_model.layers) - n_gel}/{len(base_model.layers)}")

    cb2 = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)]
    hist2 = model.fit(ds_tr, validation_data=ds_val, epochs=EPOCHS_P2,
                      class_weight=class_weight, callbacks=cb2, verbose=1)


# %% Courbes d'apprentissage
def trace_courbes(h1, h2):
    def concat(cle):
        return list(h1.history.get(cle, [])) + list(h2.history.get(cle, []))
    n1 = len(h1.history["loss"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    a_tracer = [("loss", "val_loss", "Perte"),
                ("accuracy", "val_accuracy", "Accuracy")]
    for ax, (k, kv, titre) in zip(axes, a_tracer):
        ax.plot(concat(k), label="train", lw=1.8)
        ax.plot(concat(kv), label="val", lw=1.8)
        ax.axvline(n1 - 1, color="grey", ls="--", lw=1, label="-> fine-tuning")
        ax.set_title(titre)
        ax.set_xlabel("epoque")
        ax.legend()
        ax.grid(alpha=.3)
    plt.suptitle(f"{CATEGORIE} - {BACKBONE} - identification ({N_CLASSES} classes)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"courbes_{CATEGORIE}_{BACKBONE}.png", dpi=140, bbox_inches="tight")
    plt.show()

if MODE_RAPIDE:
    trace_courbes(hist1, hist2)


# %% [markdown]
# ## 4. Evaluation multi-classe
#
# On regarde surtout le F1 macro (chaque type de defaut compte pareil) et la matrice
# de confusion N×N : quels defauts sont confondus entre eux ?

# %% Metriques (mode rapide seulement)
if MODE_RAPIDE:
    y_prob = model.predict(ds_te, verbose=0)
    y_pred = y_prob.argmax(1)      # la classe avec la plus forte proba

    f1m = f1_score(y_te, y_pred, average="macro")
    print(f"F1 macro : {f1m:.3f}   <- metrique principale")
    print(f"Accuracy : {(y_pred == y_te).mean():.3f}\n")
    print(classification_report(y_te, y_pred, target_names=CLASSES, digits=3, zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, norm, titre in zip(axes, [None, "true"],
                               ["Effectifs", "Normalisee (rappel par classe)"]):
        cm = confusion_matrix(y_te, y_pred, labels=np.arange(N_CLASSES), normalize=norm)
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
            ax=ax, cmap="Blues", colorbar=False,
            values_format=".2f" if norm else "d", xticks_rotation=45)
        ax.set_title(titre)
    plt.suptitle(f"{CATEGORIE} - {BACKBONE} - identification", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"confusion_{CATEGORIE}_{BACKBONE}.png", dpi=140, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 5. Grad-CAM : ou le modele regarde pour choisir le type de defaut

# %% Grad-CAM multi-classe
def derniere_conv(base):
    for couche in reversed(base.layers):
        if isinstance(couche, layers.Conv2D):
            return couche.name
    return None

def gradcam(model, base, img_batch, class_idx):
    grad_model = keras.models.Model(base.input,
                                    [base.get_layer(derniere_conv(base)).output, base.output])
    with tf.GradientTape() as tape:
        conv_out, _ = grad_model(img_batch)
        tape.watch(conv_out)
        x = layers.GlobalAveragePooling2D()(conv_out)
        score = model.get_layer("pred")(x)[:, class_idx]   # score de la classe predite
    grads = tape.gradient(score, conv_out)
    poids = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * poids, axis=-1)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()

def montre_gradcam(model, base, paths, y_true, n=6):
    import matplotlib.cm as cm
    idx = np.random.default_rng(42).choice(len(paths), min(n, len(paths)), replace=False)
    fig, axes = plt.subplots(2, len(idx), figsize=(2.3 * len(idx), 4.8))
    for c, i in enumerate(idx):
        raw = tf.image.resize(tf.image.decode_png(tf.io.read_file(paths[i]), channels=3),
                              IMG_SIZE).numpy().astype("uint8")
        arr = preprocess_input(tf.cast(raw, tf.float32)[None, ...])
        pred = int(model.predict(arr, verbose=0).argmax(1)[0])
        heat = gradcam(model, base, arr, pred)
        heat_up = np.array(keras.utils.array_to_img(heat[..., None]).resize(IMG_SIZE))
        overlay = np.uint8(0.6 * raw + 0.4 * 255 * cm.jet(heat_up / 255.)[..., :3])
        vrai = CLASSES[y_true[i]]
        predit = CLASSES[pred]
        axes[0, c].imshow(raw)
        axes[0, c].axis("off")
        axes[0, c].set_title(f"vrai: {vrai}", fontsize=8,
                             color="seagreen" if vrai == predit else "crimson")
        axes[1, c].imshow(overlay)
        axes[1, c].axis("off")
        axes[1, c].set_title(f"predit: {predit}", fontsize=8)
    plt.suptitle(f"{CATEGORIE} - {BACKBONE} - GradCAM", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"gradcam_{CATEGORIE}_{BACKBONE}.png", dpi=140, bbox_inches="tight")
    plt.show()

if MODE_RAPIDE:
    montre_gradcam(model, base_model, p_te, y_te)


# %% Sauvegarde (mode rapide seulement)
if MODE_RAPIDE:
    model.save(OUT_DIR / f"{BACKBONE}_{CATEGORIE}_identification.keras")
    resume = {"categorie": CATEGORIE, "backbone": BACKBONE, "tache": "identification",
              "classes": CLASSES, "inclure_good": INCLURE_GOOD,
              "f1_macro": round(float(f1m), 3),
              "accuracy": round(float((y_pred == y_te).mean()), 3),
              "n_train": len(y_tr), "n_test": len(y_te)}
    (OUT_DIR / f"resume_{BACKBONE}_{CATEGORIE}.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resume, indent=2, ensure_ascii=False))


# %% [markdown]
# ## 6. Benchmark : comparaison des 4 backbones (identification)
#
# Memes types de graphiques que pour la detection, mais adaptes au multi-classe :
#  - metrique de comparaison = F1 macro (l'AUC-ROC ne marche que pour le binaire) ;
#  - matrice de confusion N×N par categorie (les classes changent d'une categorie a l'autre) ;
#  - barres de F1 macro par categorie, puis comparaison des backbones.
#
# Attention : 4 backbones x N categories = 4N entrainements. Commencer avec 2-3 categories.

# %% Fonction qui entraine et evalue une categorie
def pipeline_identification(categorie, backbone, inclure_good=INCLURE_GOOD, verbose=0):
    # ces variables globales servent au nom du modele et au pre-traitement
    global preprocess_input, N_CLASSES, CLASSES, BACKBONE, CATEGORIE

    paths, labels, classes = construit_manifeste_multi(
        DATA_ROOT, categorie, inclure_good, CAP_GOOD, MIN_PAR_CLASSE)
    CLASSES, N_CLASSES, BACKBONE, CATEGORIE = classes, len(classes), backbone, categorie
    (ptr, ytr), (pval, yval), (pte, yte) = split_stratifie(paths, labels)
    base, prep = get_backbone(backbone, IMG_SIZE + (3,))
    preprocess_input = prep
    # on reutilise la couche augmentation partagee (creee plus haut), on n'en refait pas une neuve
    tr = make_ds_multi(ptr, ytr, shuffle=True)
    val = make_ds_multi(pval, yval)
    te = make_ds_multi(pte, yte)
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=ytr)
    cw = {i: float(w) for i, w in enumerate(cw)}

    # phase 1 : backbone gele
    base.trainable = False
    m = construit_modele_multi(base, len(classes))
    m.compile(optimizer=keras.optimizers.Adam(LR_P1), loss="categorical_crossentropy",
              metrics=["accuracy"])
    m.fit(tr, validation_data=val, epochs=EPOCHS_P1, class_weight=cw, verbose=verbose,
          callbacks=[keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)])

    # phase 2 : fine-tuning
    base.trainable = True
    ng = int(len(base.layers) * (1 - UNFREEZE))
    for couche in base.layers[:ng]:
        couche.trainable = False
    for couche in base.layers:
        if isinstance(couche, layers.BatchNormalization):
            couche.trainable = False
    m.compile(optimizer=keras.optimizers.Adam(LR_P2), loss="categorical_crossentropy",
              metrics=["accuracy"])
    m.fit(tr, validation_data=val, epochs=EPOCHS_P2, class_weight=cw, verbose=verbose,
          callbacks=[keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)])

    yp = m.predict(te, verbose=0).argmax(1)
    return {"categorie": categorie, "backbone": backbone, "n_classes": len(classes),
            "classes": classes,
            "f1_macro": float(f1_score(yte, yp, average="macro", zero_division=0)),
            "accuracy": float((yp == yte).mean()),
            "n_test": len(yte), "y_test": yte, "y_pred": yp, "paths_test": pte}


# %% Boucle sur les backbones et les categories
# Ne tourne qu'en mode complet (MODE_RAPIDE = False).
BACKBONES = ["resnet50", "efficientnetb0", "vgg16", "mobilenetv2"]
CATS = ["bottle", "screw", "carpet"]        # a reduire / etendre ici
# Liste complete (attention : 4 x 15 = 60 entrainements, tres long) :
# CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
#         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

results, resume_bb = {}, {}     # results[backbone][cat] = infos (y_test, y_pred, classes...)
import pandas as pd

if MODE_RAPIDE:
    print(">>> [MODE_RAPIDE] Benchmark saute. Mets MODE_RAPIDE = False pour le lancer.")
else:
    n_runs = len(BACKBONES) * len(CATS)
    print(f"\n>>> Benchmark : {len(BACKBONES)} backbone(s) x "
          f"{len(CATS)} categorie(s) = {n_runs} entrainement(s)")
    import time
    t0 = time.time()
    for bb in BACKBONES:
        results[bb] = {}
        for cat in CATS:
            print(f"\n{'='*56}\n  {bb} - {cat}\n{'='*56}")
            info = pipeline_identification(cat, bb, verbose=0)
            results[bb][cat] = info
            print(f"  F1 macro={info['f1_macro']:.3f} | accuracy={info['accuracy']:.3f} "
                  f"| {info['n_classes']} classes")
    print(f"\n{'='*56}\n  Termine en {time.time()-t0:.0f}s\n{'='*56}")

    # moyenne par backbone (+ ecart-type sur les categories)
    for bb in BACKBONES:
        f1s = [results[bb][c]["f1_macro"] for c in CATS]
        accs = [results[bb][c]["accuracy"] for c in CATS]
        resume_bb[bb] = {"f1_macro_moy": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
                         "accuracy_moy": float(np.mean(accs)), "accuracy_std": float(np.std(accs))}
    df_bb = pd.DataFrame(resume_bb).T.round(3)
    df_bb.to_csv(OUT_DIR / "identification_benchmark_backbones.csv")
    print("\nResume par backbone :")
    print(df_bb.to_string())


# %% [markdown]
# ### 6a. Matrices de confusion N×N + barres F1 macro, pour chaque backbone

# %% Graphiques multi-classe
def grille_confusion_multi(res_bb, titre="backbone"):
    cats = list(res_bb.keys())
    ncol = min(3, len(cats))
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.8, nrow * 4.3))
    axes = np.array(axes).ravel()
    for i, cat in enumerate(cats):
        r = res_bb[cat]
        cm = confusion_matrix(r["y_test"], r["y_pred"], labels=np.arange(r["n_classes"]))
        ConfusionMatrixDisplay(cm, display_labels=r["classes"]).plot(
            ax=axes[i], cmap="Blues", colorbar=False, values_format="d", xticks_rotation=45)
        axes[i].set_title(f"{cat}\nF1 macro={r['f1_macro']:.3f}", fontsize=9, fontweight="bold")
    for j in range(len(cats), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{titre} - matrices de confusion N×N (identification)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ident_confusion_{titre}.png", dpi=130, bbox_inches="tight")
    plt.show()


def barres_f1_multi(res_bb, titre="backbone"):
    cats = list(res_bb.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, metric, lib, col in [(axes[0], "f1_macro", "F1 macro", "#3498db"),
                                 (axes[1], "accuracy", "Accuracy", "#2ecc71")]:
        vals = np.array([res_bb[c][metric] for c in cats])
        order = np.argsort(vals)
        cc = [cats[i] for i in order]
        v = vals[order]
        colors = [col if x >= vals.mean() else "#e74c3c" for x in v]
        ax.barh(cc, v, color=colors, edgecolor="white", height=.7)
        ax.axvline(vals.mean(), color="gray", ls="--", lw=1.4, label=f"Moyenne = {vals.mean():.3f}")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(lib)
        ax.set_title(f"{lib} par categorie", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
    plt.suptitle(f"{titre} - identification, {len(cats)} categories", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ident_barres_{titre}.png", dpi=140, bbox_inches="tight")
    plt.show()

if not MODE_RAPIDE:
    for bb in BACKBONES:
        grille_confusion_multi(results[bb], titre=bb)
        barres_f1_multi(results[bb], titre=bb)


# %% [markdown]
# ## 7. Comparaison des 4 backbones (identification) - le graphique du rapport
#
# Metrique commune = F1 macro. Barres groupees F1 macro + accuracy, avec ecart-type.

# %% Barres comparatives des backbones
def compare_backbones_ident(resume_bb, titre="Identification - comparaison des 4 backbones"):
    bbs = list(resume_bb.keys())
    f1 = [resume_bb[b]["f1_macro_moy"] for b in bbs]
    f1_std = [resume_bb[b]["f1_macro_std"] for b in bbs]
    acc = [resume_bb[b]["accuracy_moy"] for b in bbs]
    acc_std = [resume_bb[b]["accuracy_std"] for b in bbs]
    x = np.arange(len(bbs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.9 * len(bbs) + 3, 5))
    b1 = ax.bar(x - w / 2, f1, w, yerr=f1_std, capsize=4, label="F1 macro",
                color="#3498db", edgecolor="white")
    b2 = ax.bar(x + w / 2, acc, w, yerr=acc_std, capsize=4, label="Accuracy",
                color="#2ecc71", edgecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bbs)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score moyen (+/- ecart-type)")
    ax.set_title(titre, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ident_comparaison_backbones.png", dpi=140, bbox_inches="tight")
    plt.show()

    print("\nClassement (F1 macro moyen) :")
    for b in sorted(bbs, key=lambda n: resume_bb[n]["f1_macro_moy"], reverse=True):
        print(f"  {resume_bb[b]['f1_macro_moy']:.3f}  {b}")

if not MODE_RAPIDE:
    compare_backbones_ident(resume_bb)


# %% F1 macro par categorie ET par backbone (barres groupees)
def compare_backbones_par_cat_ident(results, metric="f1_macro"):
    bbs = list(results.keys())
    cats = list(results[bbs[0]].keys())
    x = np.arange(len(cats))
    w = 0.8 / len(bbs)
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(cats) + 3), 5))
    for i, bb in enumerate(bbs):
        vals = [results[bb][c][metric] for c in cats]
        ax.bar(x + (i - (len(bbs) - 1) / 2) * w, vals, w, label=bb, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"{metric} par categorie et par backbone (identification)", fontweight="bold")
    ax.legend(fontsize=8, ncol=len(bbs))
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ident_comparaison_par_cat_{metric}.png", dpi=140, bbox_inches="tight")
    plt.show()

if not MODE_RAPIDE:
    compare_backbones_par_cat_ident(results, "f1_macro")


# %% [markdown]
# ## 8. Graphiques qui montrent vraiment l'identification par type de defaut
#
#  - 8a. F1 par type de defaut : une barre par classe, avec le support (n) annote.
#  - 8b. Galerie de predictions : images avec vrai / predit (vert = correct, rouge = erreur).

# %% 8a. F1 par TYPE de defaut, pour chaque backbone
def f1_par_type(res_bb, titre="backbone"):
    from sklearn.metrics import precision_recall_fscore_support
    cats = list(res_bb.keys())
    ncol = min(3, len(cats))
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.7, nrow * 4.0))
    axes = np.array(axes).ravel()
    for i, cat in enumerate(cats):
        r = res_bb[cat]
        _, _, f1c, sup = precision_recall_fscore_support(
            r["y_test"], r["y_pred"], labels=np.arange(r["n_classes"]), zero_division=0)
        ax = axes[i]
        # good en gris, les defauts en bleu
        colors = ["#95a5a6" if c == "good" else "#3498db" for c in r["classes"]]
        ax.barh(r["classes"], f1c, color=colors, edgecolor="white")
        # on annote le support (nombre d'images de test) a cote de chaque barre
        for j, (fv, s) in enumerate(zip(f1c, sup)):
            ax.text(min(fv + 0.02, 0.99), j, f"n={s}", va="center", fontsize=8, color="#555")
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("F1")
        ax.set_title(f"{cat}  (F1 macro={r['f1_macro']:.3f})", fontsize=10, fontweight="bold")
    for j in range(len(cats), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{titre} - F1 par TYPE de defaut (support annote)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ident_f1_par_type_{titre}.png", dpi=130, bbox_inches="tight")
    plt.show()

if not MODE_RAPIDE:
    for bb in BACKBONES:
        f1_par_type(results[bb], titre=bb)


# %% 8b. Galerie de predictions, pour le meilleur backbone seulement
def galerie_predictions(res_bb, titre="backbone", n_par_cat=8, seed=42):
    for cat, r in res_bb.items():
        paths = r["paths_test"]
        yt = r["y_test"]
        yp = r["y_pred"]
        cl = r["classes"]
        idx = np.random.default_rng(seed).choice(len(paths), min(n_par_cat, len(paths)), replace=False)
        ncol = min(4, len(idx))
        nrow = int(np.ceil(len(idx) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 3.0 * nrow))
        axes = np.array(axes).ravel()
        for a, k in enumerate(idx):
            img = keras.utils.load_img(paths[k], target_size=IMG_SIZE)
            axes[a].imshow(img)
            axes[a].axis("off")
            ok = (yt[k] == yp[k])
            axes[a].set_title(f"vrai : {cl[yt[k]]}\npredit : {cl[yp[k]]}",
                              fontsize=8, color="seagreen" if ok else "crimson")
        for a in range(len(idx), len(axes)):
            axes[a].axis("off")
        plt.suptitle(f"{titre} - {cat} : predictions (vert = correct, rouge = erreur)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"ident_galerie_{titre}_{cat}.png", dpi=120, bbox_inches="tight")
        plt.show()

if not MODE_RAPIDE:
    # on prend le backbone avec le meilleur F1 macro moyen (pour ne pas faire 4 galeries)
    BACKBONE_DETAIL = max(BACKBONES, key=lambda b: resume_bb[b]["f1_macro_moy"])
    print(f"Galerie generee pour le meilleur backbone : {BACKBONE_DETAIL}")
    galerie_predictions(results[BACKBONE_DETAIL], titre=BACKBONE_DETAIL)


# %% [markdown]
# ## Ce que je retiens - detection vs identification
#
# - Detection = binaire (1 neurone sigmoid) ; identification = multi-classe (N neurones softmax).
# - Toujours un modele par categorie (un modele global melangerait des defauts differents
#   qui portent parfois le meme nom).
# - Metrique principale de l'identification = F1 macro, et on regarde la matrice de confusion
#   N×N pour voir quels defauts se confondent.
# - Il y a peu d'images par type de defaut : toujours afficher le support (n) a cote du F1.
