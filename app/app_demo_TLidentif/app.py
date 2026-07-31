# -*- coding: utf-8 -*-
# Demo Streamlit : identification du type de defaut par transfer learning.
#
# L'utilisateur choisit une categorie et un backbone (un BON et un FAIBLE, pour
# montrer l'impact du choix du modele), fournit une image (upload ou exemple du
# jeu de test), et l'app affiche :
#   - le type de defaut predit + la confiance,
#   - les probabilites de toutes les classes,
#   - une carte Grad-CAM (ou le modele regarde),
#   - le F1 macro du modele (pour rappeler sa qualite).
#
# Prerequis : avoir lance `python entrainer_modeles_demo.py` pour creer les
# modeles dans models_demo/.
#
# Lancement :  streamlit run app.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import streamlit as st
from tensorflow import keras
from tensorflow.keras import layers

# ─── chemins ─────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models_demo")
DATA_ROOT  = Path("../dataBase_mvtec")     # pour les images d'exemple
IMG_SIZE   = (224, 224)

# preprocessing propre a chaque backbone (doit etre IDENTIQUE a l'entrainement)
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
        raise ValueError(backbone)
    return preprocess_input


# ─── chargements (mis en cache pour ne pas recharger a chaque clic) ──────────
@st.cache_data
def charge_index():
    fichier = MODELS_DIR / "demo_index.json"
    if not fichier.exists():
        return []
    return json.loads(fichier.read_text(encoding="utf-8"))["modeles"]


@st.cache_resource
def charge_modele(fichier):
    return keras.models.load_model(MODELS_DIR / fichier)


# ─── Grad-CAM ────────────────────────────────────────────────────────────────
def trouve_backbone(modele):
    # dans notre modele, le backbone est un sous-modele imbrique
    for couche in modele.layers:
        if isinstance(couche, keras.Model):
            return couche
    return None

def derniere_conv(base):
    for couche in reversed(base.layers):
        if isinstance(couche, layers.Conv2D):
            return couche.name
    return None

def calcule_gradcam(modele, image_pretraitee, classe):
    import tensorflow as tf
    base = trouve_backbone(modele)
    grad_model = keras.models.Model(
        base.input, [base.get_layer(derniere_conv(base)).output, base.output])
    with tf.GradientTape() as tape:
        conv_out, _ = grad_model(image_pretraitee)
        tape.watch(conv_out)
        x = layers.GlobalAveragePooling2D()(conv_out)
        score = modele.get_layer("pred")(x)[:, classe]
    grads = tape.gradient(score, conv_out)
    poids = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * poids, axis=-1)
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()

def superpose_gradcam(image_rgb, heat):
    heat_img = np.array(keras.utils.array_to_img(heat[..., None]).resize(IMG_SIZE))
    couleur = cm.jet(heat_img / 255.0)[..., :3]
    overlay = np.uint8(0.6 * image_rgb + 0.4 * 255 * couleur)
    return overlay


# ─── liste d'images d'exemple du jeu de test ────────────────────────────────
def images_exemple(categorie, n=12):
    dossier = DATA_ROOT / categorie / "test"
    if not dossier.exists():
        return []
    images = []
    for sous in sorted(dossier.iterdir()):
        if sous.is_dir():
            for p in sorted(sous.glob("*.png"))[:3]:      # 3 par type de defaut
                images.append((f"{sous.name} / {p.name}", str(p)))
    return images[:n]


# ═══════════════════════════════════════════════════════════════════════════
#  INTERFACE
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Identification de defauts", page_icon="🔍", layout="wide")
st.title("🔍 Identification du type de defaut par transfer learning")
st.caption("MVTec AD — un modele par categorie, classification multi-classe (good + types de defaut)")

index = charge_index()
if not index:
    st.error("Aucun modele trouve. Lance d'abord :  python entrainer_modeles_demo.py")
    st.stop()

# ─── barre laterale : choix categorie + backbone ─────────────────────────────
with st.sidebar:
    st.header("Reglages")
    categories = sorted({m["categorie"] for m in index})
    categorie = st.selectbox("Categorie", categories)

    # backbones disponibles pour cette categorie, avec leur niveau (bon / faible)
    dispo = [m for m in index if m["categorie"] == categorie]
    def libelle(m):
        etoile = "✅ bon" if m["niveau"] == "bon" else "⚠️ faible"
        return f"{m['backbone']}  ({etoile}, F1={m['f1_macro']:.2f})"
    choix = st.radio("Modele", dispo, format_func=libelle)

    st.divider()
    st.markdown(
        "**Astuce demo :** compare le meme cas avec le modele **bon** puis "
        "le modele **faible** — l'ecart se voit surtout sur les defauts fins (ex. `screw`)."
    )

modele = charge_modele(choix["fichier"])
classes = choix["classes"]
preprocess = get_preprocess(choix["backbone"])

# ─── choix de l'image ────────────────────────────────────────────────────────
st.subheader(f"Categorie : {categorie}  —  modele : {choix['backbone']}")

col_gauche, col_droite = st.columns([1, 1])
with col_gauche:
    source = st.radio("Image a analyser", ["Exemple du jeu de test", "Uploader une image"],
                      horizontal=True)
    chemin_image = None
    if source == "Exemple du jeu de test":
        exemples = images_exemple(categorie)
        if exemples:
            noms = [e[0] for e in exemples]
            i = st.selectbox("Choisis une image", range(len(noms)), format_func=lambda k: noms[k])
            chemin_image = exemples[i][1]
    else:
        up = st.file_uploader("Image (.png / .jpg)", type=["png", "jpg", "jpeg"])
        if up is not None:
            chemin_image = up

# ─── prediction ──────────────────────────────────────────────────────────────
if chemin_image is not None:
    img = keras.utils.load_img(chemin_image, target_size=IMG_SIZE)
    image_rgb = np.array(img).astype("uint8")
    x = preprocess(np.array(img, dtype="float32")[None, ...])

    proba = modele.predict(x, verbose=0)[0]
    i_pred = int(proba.argmax())
    classe_pred = classes[i_pred]
    confiance = float(proba[i_pred])

    with col_gauche:
        st.image(image_rgb, caption="Image analysee", width=300)

    with col_droite:
        if classe_pred == "good":
            st.success(f"### Prediction : **{classe_pred}** (conforme)")
        else:
            st.warning(f"### Prediction : **{classe_pred}**")
        st.metric("Confiance", f"{confiance*100:.1f} %")

        # probabilites de toutes les classes
        st.markdown("**Probabilites par classe**")
        ordre = np.argsort(proba)[::-1]
        df_proba = pd.DataFrame({"probabilite": [float(proba[k]) for k in ordre]},
                                index=[classes[k] for k in ordre])
        st.bar_chart(df_proba)

    # Grad-CAM
    st.divider()
    st.markdown("### Grad-CAM — ou le modele regarde")
    try:
        heat = calcule_gradcam(modele, x, i_pred)
        overlay = superpose_gradcam(image_rgb, heat)
        c1, c2 = st.columns(2)
        c1.image(image_rgb, caption="Original", width=320)
        c2.image(overlay, caption=f"Zones decisives pour « {classe_pred} »", width=320)
    except Exception as e:
        st.info(f"Grad-CAM indisponible pour ce modele ({e}).")

    st.caption(
        f"Rappel : ce modele ({choix['backbone']}, niveau « {choix['niveau']} ») a un "
        f"F1 macro de {choix['f1_macro']:.2f} sur {choix['n_test']} images de test. "
        "Les scores par classe reposent sur peu d'images — a interpreter avec prudence."
    )
else:
    st.info("Choisis une image d'exemple ou uploade-en une pour lancer l'identification.")
