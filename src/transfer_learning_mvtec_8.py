# -*- coding: utf-8 -*-
# Detection d'anomalie par transfer learning sur MVTec AD
# Objectif : pour une categorie (bottle, screw, carpet...), dire si une image
# est conforme (0) ou defectueuse (1), en partant d'un CNN deja entraine sur ImageNet.
#
# J'entraine UN modele par categorie (une bouteille et une vis n'ont rien a voir).
# Entrainement en 2 temps : d'abord on gele le backbone, ensuite on degele un peu (fine-tuning).
#
# Le fichier est decoupe en cellules "# %%" (bouton "Run Cell" dans VSCode).

# %% Imports et reglages de depart
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve)

# on fixe les graines aleatoires pour avoir toujours le meme resultat
np.random.seed(42)
tf.random.set_seed(42)
plt.rcParams["figure.dpi"] = 110

# chemin vers les donnees (a changer si besoin)
DATA_ROOT = Path("../Data_Projet_DS")

# Mode rapide :
#   True  = petit test rapide (1 backbone x 1 categorie, peu d'epochs)
#   False = benchmark complet (voir la liste BACKBONES / CATEGORIES plus bas)
MODE_RAPIDE = True

CATEGORIE  = "bottle"          # je commence par bottle, puis screw, carpet...
BACKBONE   = "resnet50"        # resnet50 / efficientnetb0 / vgg16 / mobilenetv2
IMG_SIZE   = (224, 224)        # taille attendue par les modeles ImageNet
BATCH_SIZE = 16
LR_P1      = 1e-3              # learning rate phase 1
LR_P2      = 1e-5             # learning rate phase 2 : 100x plus petit (important)
CAP_GOOD   = 120              # nombre max d'images conformes (pour equilibrer) ; None = pas de limite
UNFREEZE   = 0.20             # part du backbone qu'on degele en phase 2

# moins d'epochs en mode rapide (de toute facon l'EarlyStopping coupe souvent avant)
if MODE_RAPIDE:
    EPOCHS_P1, EPOCHS_P2 = 8, 6
else:
    EPOCHS_P1, EPOCHS_P2 = 20, 15

# dossier ou je sauvegarde les figures et les modeles
OUT_DIR = Path(__file__).parent / "resultats" if "__file__" in dir() else Path("resultats")
OUT_DIR.mkdir(exist_ok=True)

print("TensorFlow", tf.__version__)
print("GPU :", tf.config.list_physical_devices("GPU") or "aucun (CPU)")
assert DATA_ROOT.exists(), f"Chemin introuvable : {DATA_ROOT}"


# %% [markdown]
# ## 1. Preparer les images d'une categorie et faire le decoupage
#
# Dans MVTec, le decoupage d'origine ne met aucun defaut dans le train (train/good
# uniquement), donc il n'est pas utilisable en supervise. Je remets tout ensemble
# puis je refais un decoupage stratifie 70/15/15 (train/val/test).
#
# - conformes (0) = train/good + test/good
# - defauts (1)   = tous les sous-dossiers de test/ sauf good

# %% Liste des images + labels
def construit_manifeste(data_root, categorie, cap_good=None, seed=42):
    cat = Path(data_root) / categorie
    conformes = []
    defauts = []

    # images conformes du dossier train/good
    for p in sorted((cat / "train" / "good").glob("*.png")):
        conformes.append(str(p))

    # dossier test/ : good = conforme, le reste = defaut
    for sous in sorted((cat / "test").iterdir()):
        if not sous.is_dir():
            continue
        for p in sorted(sous.glob("*.png")):
            if sous.name == "good":
                conformes.append(str(p))
            else:
                defauts.append(str(p))

    # si trop de conformes, on en garde seulement cap_good (tirage au hasard)
    rng = np.random.default_rng(seed)
    conformes = np.array(conformes)
    if cap_good is not None and len(conformes) > cap_good:
        conformes = rng.choice(conformes, cap_good, replace=False)

    paths = np.concatenate([conformes, np.array(defauts)])
    labels = np.concatenate([np.zeros(len(conformes), int), np.ones(len(defauts), int)])
    return paths, labels


def split_stratifie(paths, labels, seed=42):
    # 1) on met de cote le test (15%)
    p_rest, p_test, y_rest, y_test = train_test_split(
        paths, labels, test_size=0.15, random_state=seed, stratify=labels)
    # 2) on separe train / val sur le reste (val = 15% du total, soit ~0.176 du reste)
    p_tr, p_val, y_tr, y_val = train_test_split(
        p_rest, y_rest, test_size=0.176, random_state=seed, stratify=y_rest)
    return (p_tr, y_tr), (p_val, y_val), (p_test, y_test)


paths, labels = construit_manifeste(DATA_ROOT, CATEGORIE, CAP_GOOD)
(p_tr, y_tr), (p_val, y_val), (p_te, y_te) = split_stratifie(paths, labels)

print(f"Categorie : {CATEGORIE}")
print(f"  total    : {len(paths)}  (conformes {int((labels==0).sum())}, defauts {int((labels==1).sum())})")
for nom, y in [("train", y_tr), ("val", y_val), ("test", y_te)]:
    print(f"  {nom:5} : {len(y):4}  (conformes {int((y==0).sum())}, defauts {int((y==1).sum())})")


# %% [markdown]
# ## 2. Charger les images avec tf.data
#
# Chaque backbone a son propre pre-traitement des pixels, donc je le passe en
# parametre. Le redimensionnement en 224x224 est fait pour les 3 jeux.

# %% Choix du backbone + fonction de chargement
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


def make_ds(paths, labels, preprocess_fn, shuffle=False, batch=BATCH_SIZE):
    # petite fonction qui lit une image et applique le pre-traitement
    def charge_une(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        img = preprocess_fn(img)
        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(charge_une, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


base_model, preprocess_input = get_backbone(BACKBONE, IMG_SIZE + (3,))
ds_tr  = make_ds(p_tr, y_tr, preprocess_input, shuffle=True)
ds_val = make_ds(p_val, y_val, preprocess_input)
ds_te  = make_ds(p_te, y_te, preprocess_input)

# petit controle : on regarde un batch
for x, y in ds_tr.take(1):
    print("batch images :", x.shape, f"[{float(tf.reduce_min(x)):.1f}, {float(tf.reduce_max(x)):.1f}]")


# %% Poids des classes (il y a plus de conformes que de defauts)
poids = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
class_weight = {0: float(poids[0]), 1: float(poids[1])}
print("class_weight :", {k: round(v, 2) for k, v in class_weight.items()})


# %% [markdown]
# ## 3. Augmentation des donnees (seulement a l'entrainement)
#
# Je limite la rotation a environ 10 degres : sur des pieces comme screw ou metal_nut
# l'orientation est fixe. Je ne touche pas a la couleur car parfois le defaut EST une
# couleur anormale.

# %% La couche d'augmentation
def couche_augmentation(seed=42):
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.03, fill_mode="reflect", seed=seed),     # ~ +/- 11 degres
        layers.RandomZoom(0.10, fill_mode="reflect", seed=seed),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed),
    ], name="augmentation")

# Astuce vitesse : je cree l'augmentation UNE SEULE FOIS et je la reutilise pour tous
# les modeles. Si j'en recree une neuve a chaque modele, le benchmark devient tres lent
# sur CPU (TensorFlow doit tout recompiler a chaque fois).
augmentation = couche_augmentation()


# %% [markdown]
# ## 4. Phase 1 : on gele le backbone, on entraine juste la tete
#
# backbone ImageNet (gele) -> GlobalAveragePooling -> Dropout -> Dense(1, sigmoid)
#
# On gele d'abord pour ne pas abimer les poids ImageNet avec les gradients de la tete
# qui est initialisee au hasard.

# %% Construction du modele + phase 1
def construit_modele(base, augment=True, dropout=0.3):
    inp = keras.Input(shape=IMG_SIZE + (3,), name="image")
    if augment:
        x = augmentation(inp)          # on reutilise la couche partagee
    else:
        x = inp
    x = base(x, training=False)        # training=False pour figer les BatchNorm
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="pred")(x)   # 1 neurone = binaire
    return Model(inp, out, name=f"{BACKBONE}_{CATEGORIE}")


# La demo sur un seul modele (parties 4 a 7) ne tourne qu'en mode rapide. En mode complet
# on saute directement au benchmark pour ne pas entrainer bottle/resnet50 deux fois.
model, hist1, hist2 = None, None, None
if MODE_RAPIDE:
    base_model.trainable = False
    model = construit_modele(base_model)
    model.compile(optimizer=keras.optimizers.Adam(LR_P1),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    cb1 = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                         restore_best_weights=True)]
    hist1 = model.fit(ds_tr, validation_data=ds_val, epochs=EPOCHS_P1,
                      class_weight=class_weight, callbacks=cb1, verbose=1)
else:
    print("[MODE COMPLET] Demo mono-modele sautee -> voir le benchmark plus bas.")


# %% [markdown]
# ## 5. Phase 2 : fine-tuning (on degele les derniers blocs, LR / 100)
#
# On divise le learning rate par 100. Si on le laisse trop grand, le modele "oublie"
# ce qu'il avait appris sur ImageNet. On garde les BatchNorm gelees (les lots sont petits).

# %% Fine-tuning
if MODE_RAPIDE:
    base_model.trainable = True
    n_gel = int(len(base_model.layers) * (1 - UNFREEZE))
    # on regele les premieres couches
    for couche in base_model.layers[:n_gel]:
        couche.trainable = False
    # et on garde toutes les BatchNorm gelees
    for couche in base_model.layers:
        if isinstance(couche, layers.BatchNormalization):
            couche.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(LR_P2),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    print(f"Couches degelees : {len(base_model.layers) - n_gel}/{len(base_model.layers)}")

    cb2 = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                         restore_best_weights=True)]
    hist2 = model.fit(ds_tr, validation_data=ds_val, epochs=EPOCHS_P2,
                      class_weight=class_weight, callbacks=cb2, verbose=1)


# %% Courbes d'apprentissage (perte et AUC)
def trace_courbes(h1, h2, metrique="auc"):
    # on colle les 2 phases bout a bout
    def concat(cle):
        return list(h1.history.get(cle, [])) + list(h2.history.get(cle, []))

    n1 = len(h1.history["loss"])   # ou commence la phase 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    a_tracer = [("loss", "val_loss", "Perte"),
                (metrique, f"val_{metrique}", metrique.upper())]
    for ax, (k, kv, titre) in zip(axes, a_tracer):
        ax.plot(concat(k), label="train", lw=1.8)
        ax.plot(concat(kv), label="val", lw=1.8)
        ax.axvline(n1 - 1, color="grey", ls="--", lw=1, label="-> fine-tuning")
        ax.set_title(titre)
        ax.set_xlabel("epoque")
        ax.legend()
        ax.grid(alpha=.3)
    plt.suptitle(f"{CATEGORIE} - {BACKBONE}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"courbes_{CATEGORIE}_{BACKBONE}.png", dpi=140, bbox_inches="tight")
    plt.show()

if MODE_RAPIDE:
    trace_courbes(hist1, hist2)


# %% [markdown]
# ## 6. Evaluation : F1 macro, matrice de confusion, courbe ROC
#
# Je ne regarde pas juste l'accuracy (les classes sont desequilibrees). Le jeu de test
# ne sert qu'ici, a la toute fin.

# %% Metriques (seulement en mode rapide)
if MODE_RAPIDE:
    y_prob = model.predict(ds_te, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    auc = roc_auc_score(y_te, y_prob)
    f1m = f1_score(y_te, y_pred, average="macro")
    print(f"ROC-AUC  : {auc:.3f}")
    print(f"F1 macro : {f1m:.3f}   <- metrique principale")
    print(f"Accuracy : {(y_pred == y_te).mean():.3f}\n")
    print(classification_report(y_te, y_pred, target_names=["Conforme", "Defaut"],
                                digits=3, zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # matrice de confusion
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                           display_labels=["Conforme", "Defaut"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Matrice de confusion")
    # courbe ROC
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    axes[1].plot(fpr, tpr, lw=2.5, color="#3498db", label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="hasard")
    axes[1].set_xlabel("faux positifs")
    axes[1].set_ylabel("vrais positifs")
    axes[1].set_title("Courbe ROC")
    axes[1].legend()
    plt.suptitle(f"{CATEGORIE} - {BACKBONE}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"eval_{CATEGORIE}_{BACKBONE}.png", dpi=140, bbox_inches="tight")
    plt.show()

    # faux negatifs = defauts classes conformes (c'est le pire cas en industrie)
    fn = ((y_te == 1) & (y_pred == 0)).sum()
    print(f"Faux negatifs (defauts rates) : {fn}/{int((y_te==1).sum())}")


# %% [markdown]
# ## 7. Grad-CAM : voir ou le modele regarde
#
# On affiche une carte de chaleur des zones qui ont compte dans la decision. Ca permet
# de verifier que le modele regarde bien le defaut et pas le fond de l'image.

# %% Grad-CAM
def derniere_conv(base):
    # on cherche la derniere couche de convolution du backbone
    for couche in reversed(base.layers):
        if isinstance(couche, layers.Conv2D):
            return couche.name
    return None


def gradcam(model, img_batch, base):
    layer_name = derniere_conv(base)
    grad_model = keras.models.Model(base.input,
                                    [base.get_layer(layer_name).output, base.output])
    with tf.GradientTape() as tape:
        conv_out, _ = grad_model(img_batch)
        tape.watch(conv_out)
        x = layers.GlobalAveragePooling2D()(conv_out)
        score = model.get_layer("pred")(x)
    grads = tape.gradient(score, conv_out)
    poids = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * poids, axis=-1)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)   # normalise entre 0 et 1
    return heat.numpy()


def montre_gradcam(model, base, paths, y_true, n=6):
    import matplotlib.cm as cm
    idx = np.where(y_true == 1)[0][:n]        # on prend des defauts
    fig, axes = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.6))
    for c, i in enumerate(idx):
        raw = tf.image.resize(tf.image.decode_png(tf.io.read_file(paths[i]), channels=3),
                              IMG_SIZE).numpy().astype("uint8")
        arr = preprocess_input(tf.cast(raw, tf.float32)[None, ...])
        heat = gradcam(model, arr, base)
        heat_up = np.array(keras.utils.array_to_img(heat[..., None]).resize(IMG_SIZE))
        overlay = np.uint8(0.6 * raw + 0.4 * 255 * cm.jet(heat_up / 255.)[..., :3])
        axes[0, c].imshow(raw)
        axes[0, c].axis("off")
        axes[1, c].imshow(overlay)
        axes[1, c].axis("off")
    axes[0, 0].set_title("original", loc="left", fontsize=9)
    axes[1, 0].set_title("GradCAM", loc="left", fontsize=9)
    plt.suptitle(f"{CATEGORIE} - {BACKBONE} - zones decisives", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"gradcam_{CATEGORIE}_{BACKBONE}.png", dpi=140, bbox_inches="tight")
    plt.show()

if MODE_RAPIDE:
    montre_gradcam(model, base_model, p_te, y_te)


# %% Sauvegarde du modele et des chiffres (mode rapide seulement)
if MODE_RAPIDE:
    model.save(OUT_DIR / f"{BACKBONE}_{CATEGORIE}.keras")
    resume = {"categorie": CATEGORIE, "backbone": BACKBONE,
              "roc_auc": round(float(auc), 3), "f1_macro": round(float(f1m), 3),
              "n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
              "lr_p1": LR_P1, "lr_p2": LR_P2, "cap_good": CAP_GOOD}
    (OUT_DIR / f"resume_{BACKBONE}_{CATEGORIE}.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resume, indent=2, ensure_ascii=False))


# %% [markdown]
# ## 8. Benchmark : entrainer un modele par categorie et comparer les backbones
#
# Ici je fais une boucle : pour chaque backbone et chaque categorie, j'entraine un
# modele, je choisis le seuil optimal de Youden (celui qui maximise TPR - FPR sur la
# courbe ROC), et je garde les metriques + les predictions pour tracer les graphiques.
#
# Attention : chaque categorie = un entrainement complet. Sur 15 categories ca peut
# etre long, surtout sur CPU. Je conseille de commencer avec 2-3 categories.

# %% Fonction qui entraine et evalue une categorie
def seuil_youden(y_true, y_prob):
    # seuil qui maximise TPR - FPR sur la courbe ROC
    fpr, tpr, seuils = roc_curve(y_true, y_prob)
    return float(seuils[np.argmax(tpr - fpr)])


def entraine_evalue(categorie, backbone=BACKBONE, cap_good=CAP_GOOD, verbose=0):
    from sklearn.metrics import precision_score, recall_score

    paths, labels = construit_manifeste(DATA_ROOT, categorie, cap_good)
    (ptr, ytr), (pval, yval), (pte, yte) = split_stratifie(paths, labels)
    base, prep = get_backbone(backbone, IMG_SIZE + (3,))
    tr = make_ds(ptr, ytr, prep, shuffle=True)
    val = make_ds(pval, yval, prep)
    te = make_ds(pte, yte, prep)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    cw = {0: float(cw[0]), 1: float(cw[1])}

    # construit_modele utilise les variables globales BACKBONE/CATEGORIE/preprocess_input
    # pour le nom du modele et le pre-traitement, donc je les mets a jour ici
    global preprocess_input, BACKBONE, CATEGORIE
    preprocess_input, BACKBONE, CATEGORIE = prep, backbone, categorie

    # phase 1 : backbone gele
    base.trainable = False
    m = construit_modele(base)
    m.compile(optimizer=keras.optimizers.Adam(LR_P1), loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc")])
    m.fit(tr, validation_data=val, epochs=EPOCHS_P1, class_weight=cw, verbose=verbose,
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                   patience=6, restore_best_weights=True)])

    # phase 2 : fine-tuning
    base.trainable = True
    ng = int(len(base.layers) * (1 - UNFREEZE))
    for couche in base.layers[:ng]:
        couche.trainable = False
    for couche in base.layers:
        if isinstance(couche, layers.BatchNormalization):
            couche.trainable = False
    m.compile(optimizer=keras.optimizers.Adam(LR_P2), loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc")])
    m.fit(tr, validation_data=val, epochs=EPOCHS_P2, class_weight=cw, verbose=verbose,
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                   patience=6, restore_best_weights=True)])

    # evaluation avec le seuil de Youden
    prob = m.predict(te, verbose=0).ravel()
    auc = roc_auc_score(yte, prob)
    thr = seuil_youden(yte, prob)
    pred = (prob >= thr).astype(int)
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {"categorie": categorie, "backbone": backbone,
            "auc": float(auc), "threshold": thr,
            "accuracy": float((pred == yte).mean()),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "f1": float(f1_score(yte, pred, zero_division=0)),
            "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "y_test": yte, "y_prob": prob}


# %% Boucle sur les backbones et les categories
# Ne tourne qu'en mode complet (MODE_RAPIDE = False).
BACKBONES  = ["resnet50", "efficientnetb0", "vgg16", "mobilenetv2"]
CATEGORIES = ["bottle", "screw", "carpet"]        # a reduire / etendre ici
# Liste complete (attention : 4 x 15 = 60 entrainements, tres long) :
# CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid",
#               "hazelnut", "leather", "metal_nut", "pill", "screw",
#               "tile", "toothbrush", "transistor", "wood", "zipper"]

# results[backbone][cat] = metriques ; all_preds[backbone][cat] = predictions
results, all_preds, resume_bb = {}, {}, {}
import pandas as pd

if MODE_RAPIDE:
    print(">>> [MODE_RAPIDE] Benchmark saute. Mets MODE_RAPIDE = False pour le lancer.")
else:
    n_runs = len(BACKBONES) * len(CATEGORIES)
    print(f"\n>>> Benchmark : {len(BACKBONES)} backbone(s) x "
          f"{len(CATEGORIES)} categorie(s) = {n_runs} entrainement(s)")
    import time
    t0 = time.time()
    for bb in BACKBONES:
        results[bb], all_preds[bb] = {}, {}
        for cat in CATEGORIES:
            print(f"\n{'='*56}\n  {bb} - {cat}\n{'='*56}")
            info = entraine_evalue(cat, backbone=bb, verbose=0)
            results[bb][cat] = {k: info[k] for k in
                                ["auc", "threshold", "accuracy", "precision", "recall",
                                 "f1", "f1_macro", "tp", "fp", "fn", "tn"]}
            all_preds[bb][cat] = {"y_test": info["y_test"], "y_prob": info["y_prob"]}
            print(f"  AUC={info['auc']:.3f} | F1={info['f1']:.3f} | seuil={info['threshold']:.3f}")
    print(f"\n{'='*56}\n  Termine en {time.time()-t0:.0f}s\n{'='*56}")

    # moyenne par backbone (+ ecart-type sur les categories)
    for bb in BACKBONES:
        cats = list(results[bb].keys())
        resume_bb[bb] = {
            "auc_moy": float(np.mean([results[bb][c]["auc"] for c in cats])),
            "auc_std": float(np.std([results[bb][c]["auc"] for c in cats])),
            "f1_moy":  float(np.mean([results[bb][c]["f1"] for c in cats])),
            "f1_std":  float(np.std([results[bb][c]["f1"] for c in cats])),
        }
    df_bb = pd.DataFrame(resume_bb).T.round(3)
    df_bb.to_csv(OUT_DIR / "benchmark_backbones.csv")
    print("\nResume par backbone :")
    print(df_bb.to_string())


# %% [markdown]
# ### 8a. Grille des matrices de confusion + barres AUC/F1, pour chaque backbone

# %% Graphiques : grille de confusion et barres AUC/F1
def grille_confusion(res_bb, titre="backbone"):
    cats = list(res_bb.keys())
    ncol = 3
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.0, nrow * 3.6))
    axes = np.array(axes).ravel()
    for i, cat in enumerate(cats):
        r = res_bb[cat]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax = axes[i]
        ax.imshow(cm, cmap="Blues")
        # on ecrit les chiffres dans les cases
        for a in range(2):
            for b in range(2):
                ax.text(b, a, cm[a, b], ha="center", va="center",
                        color="white" if cm[a, b] > cm.max() / 2 else "black",
                        fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["G", "D"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["G", "D"])
        ax.set_xlabel("Predit")
        ax.set_ylabel("Reel")
        ax.set_title(f"{cat}\nAUC={r['auc']:.3f}", fontsize=9, fontweight="bold")
    # on cache les cases vides
    for j in range(len(cats), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{titre} - matrices de confusion (seuil de Youden)",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"grille_confusion_{titre}.png", dpi=130, bbox_inches="tight")
    plt.show()


def barres_auc_f1(res_bb, titre="backbone"):
    cats = list(res_bb.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, metric, lib, col in [(axes[0], "auc", "AUC-ROC", "#2ecc71"),
                                 (axes[1], "f1", "F1-score", "#3498db")]:
        vals = np.array([res_bb[c][metric] for c in cats])
        order = np.argsort(vals)
        cc = [cats[i] for i in order]
        v = vals[order]
        # rouge si en dessous de la moyenne, sinon couleur normale
        colors = [col if x >= vals.mean() else "#e74c3c" for x in v]
        ax.barh(cc, v, color=colors, edgecolor="white", height=.72)
        ax.axvline(vals.mean(), color="gray", ls="--", lw=1.4,
                   label=f"Moyenne = {vals.mean():.3f}")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(lib)
        ax.set_title(f"{lib} par categorie", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
    plt.suptitle(f"{titre} - {len(cats)} categories", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"barres_auc_f1_{titre}.png", dpi=140, bbox_inches="tight")
    plt.show()

# une grille + un jeu de barres par backbone (mode complet seulement)
if not MODE_RAPIDE:
    for bb in BACKBONES:
        grille_confusion(results[bb], titre=bb)
        barres_auc_f1(results[bb], titre=bb)


# %% [markdown]
# ## 9. Comparaison des 4 backbones (le graphique pour le rapport)
#
# A metrique commune (AUC-ROC et F1 moyens sur les categories), on compare les 4 modeles.

# %% Barres comparatives des backbones
def compare_backbones(resume_bb, titre="Comparaison des 4 backbones"):
    bbs = list(resume_bb.keys())
    auc = [resume_bb[b]["auc_moy"] for b in bbs]
    auc_std = [resume_bb[b]["auc_std"] for b in bbs]
    f1 = [resume_bb[b]["f1_moy"] for b in bbs]
    f1_std = [resume_bb[b]["f1_std"] for b in bbs]
    x = np.arange(len(bbs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.9 * len(bbs) + 3, 5))
    b1 = ax.bar(x - w / 2, auc, w, yerr=auc_std, capsize=4, label="AUC-ROC",
                color="#2ecc71", edgecolor="white")
    b2 = ax.bar(x + w / 2, f1, w, yerr=f1_std, capsize=4, label="F1-score",
                color="#3498db", edgecolor="white")
    # on ecrit la valeur au dessus de chaque barre
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bbs)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score moyen (+/- ecart-type)")
    ax.axhline(0.5, color="red", ls=":", lw=1, label="hasard (0.5)")
    ax.set_title(titre, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparaison_backbones.png", dpi=140, bbox_inches="tight")
    plt.show()

    print("\nClassement (AUC-ROC moyen) :")
    for b in sorted(bbs, key=lambda n: resume_bb[n]["auc_moy"], reverse=True):
        print(f"  {resume_bb[b]['auc_moy']:.3f}  {b}")

if not MODE_RAPIDE:
    compare_backbones(resume_bb)


# %% AUC par categorie ET par backbone (barres groupees)
def compare_backbones_par_categorie(results, metric="auc"):
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
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} par categorie et par backbone", fontweight="bold")
    ax.legend(fontsize=8, ncol=len(bbs))
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"comparaison_backbones_par_cat_{metric}.png", dpi=140, bbox_inches="tight")
    plt.show()

if not MODE_RAPIDE:
    compare_backbones_par_categorie(results, "auc")


# %% [markdown]
# ## Ce que je retiens
#
# - Un modele par categorie, classification binaire conforme / defaut.
# - Seuil de decision = seuil de Youden (pas 0.5 par defaut).
# - On regarde le F1 macro + la matrice de confusion + la ROC, jamais l'accuracy seule.
# - Le benchmark compare les 4 backbones (resnet50, efficientnetb0, vgg16, mobilenetv2).
# - C'est long : commencer avec 2-3 categories avant de lancer les 15.
