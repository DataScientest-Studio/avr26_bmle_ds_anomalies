# -*- coding: utf-8 -*-
# Fonctions de chargement + prediction + Grad-CAM pour la demo d'identification.
# (equivalent du inference.py de l'app de Paul, mais pour l'identification multi-classe)
#
# Aucun reentrainement : on charge des artefacts prepares par prepare_identification_assets.py.

import json
from pathlib import Path

import numpy as np
import streamlit as st

import config as C

BASE = Path(__file__).resolve().parent
DATA_DIR   = BASE / "data" / "identification"       # index.json + <cat>_confusion.png
MODELS_DIR = BASE / "models" / "identification"     # <cat>/model.keras (ou morceaux .partNN)
ASSETS_DIR = BASE / "assets" / "identification"     # <cat>/ images d'exemple


def get_preprocess(backbone):
    if backbone == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif backbone == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif backbone == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif backbone == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    else:
        raise ValueError(f"backbone inconnu : {backbone}")
    return preprocess_input


@st.cache_data
def charge_index():
    """index.json : { categorie: {backbone, f1_macro, accuracy, classes, n_test} }."""
    f = DATA_DIR / "index.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def categories_disponibles():
    """Categories reellement chargeables (presentes dans l'index ET avec un fichier modele)."""
    index = charge_index()
    dispo = []
    for cat in index:
        d = MODELS_DIR / cat
        if (d / "model.keras").exists() or list(d.glob("model.keras.part*")):
            dispo.append(cat)
    return dispo


def _recolle_si_besoin(dossier):
    """Reassemble model.keras.part00, part01... en un seul fichier si necessaire."""
    entier = dossier / "model.keras"
    if entier.exists():
        return entier
    morceaux = sorted(dossier.glob("model.keras.part*"))
    if not morceaux:
        raise FileNotFoundError(f"Aucun modele dans {dossier}")
    with open(entier, "wb") as sortie:
        for m in morceaux:
            sortie.write(m.read_bytes())
    return entier


@st.cache_resource
def charge_modele(categorie):
    from tensorflow import keras
    chemin = _recolle_si_besoin(MODELS_DIR / categorie)
    return keras.models.load_model(chemin)


# ─── Grad-CAM ────────────────────────────────────────────────────────────────
def _dernier_conv_et_base(modele):
    """
    Cherche la DERNIERE couche de convolution et le (sous-)modele qui la contient.
    Un modele de transfer learning contient souvent des sous-modeles imbriques :
    la couche d'augmentation (sans conv) ET le backbone (avec les conv). On doit
    descendre dedans. Renvoie (base, couche_conv, est_plat).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    def derniere_conv(m):
        for couche in reversed(m.layers):
            if isinstance(couche, layers.Conv2D):
                return couche
        return None

    # cas imbrique : un sous-modele (le backbone) contient les convolutions
    for c in modele.layers:
        if isinstance(c, keras.Model):
            conv = derniere_conv(c)
            if conv is not None:
                return c, conv, False
    # cas plat : les convolutions sont directement dans le modele
    conv = derniere_conv(modele)
    if conv is not None:
        return modele, conv, True
    return None, None, True


def calcule_gradcam(modele, image_pretraitee, classe):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    base, conv, est_plat = _dernier_conv_et_base(modele)
    if conv is None:
        raise ValueError("aucune couche de convolution (Conv2D) trouvee dans le modele")

    if est_plat:
        # la conv est reliee directement a l'entree et a la sortie du modele
        grad_model = keras.models.Model(modele.inputs, [conv.output, modele.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(image_pretraitee)
            tape.watch(conv_out)
            score = preds[:, classe]
    else:
        # base = backbone : on rejoue la tete (GAP -> Dense 'pred') apres la conv
        grad_model = keras.models.Model(base.input, conv.output)
        with tf.GradientTape() as tape:
            conv_out = grad_model(image_pretraitee)
            tape.watch(conv_out)
            h = layers.GlobalAveragePooling2D()(conv_out)
            score = modele.get_layer("pred")(h)[:, classe]

    grads = tape.gradient(score, conv_out)
    poids = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * poids, axis=-1)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()

def superpose_gradcam(image_rgb, heat):
    import matplotlib.cm as cm
    from tensorflow import keras
    heat_img = np.array(keras.utils.array_to_img(heat[..., None]).resize(C.IMG_SIZE))
    couleur = cm.jet(heat_img / 255.0)[..., :3]
    return np.uint8(0.6 * image_rgb + 0.4 * 255 * couleur)


def images_exemple(categorie):
    d = ASSETS_DIR / categorie
    if not d.exists():
        return []
    return sorted(str(p) for p in d.glob("*.png"))
