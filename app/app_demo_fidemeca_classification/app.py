# -*- coding: utf-8 -*-
"""
app.py — Application Streamlit dediee : IDENTIFICATION du type d'anomalie (Fidemeca / MVTec AD).

App autonome pour la partie identification par transfer learning.
Meme charte visuelle que l'app de Paul. Aucun réentrainement : on charge des artefacts
prepares par prepare_identification_assets.py.

Lancement :  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config as C
import identification as I

# ──────────────────────────────────────────────────────────────────────────
# CONFIG PAGE + STYLE (identique a l'app d'equipe)
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fidemeca — Identification", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .main .block-container {{ padding-top: 2rem; }}
  h1, h2, h3 {{ color: {C.COLOR_PRIMARY}; }}
</style>
""", unsafe_allow_html=True)

dispo = I.categories_disponibles()
index = I.charge_index()

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Identification d'anomalies")
    st.caption("Projet fil rouge — Bootcamp MLE, cohorte Avril 2026")
    st.markdown(f"**Équipe :** {C.EQUIPE}")
    st.markdown(f"**Entreprise (cas) :** {C.ENTREPRISE}")
    st.markdown(f"**Dataset :** {C.DATASET}")
    st.divider()
    st.caption("Modèles d'identification chargés (aucun réentrainement) :")
    if dispo:
        for cat in dispo:
            st.write(f"• {cat} — EfficientNetB0 (F1 {index[cat]['f1_macro']:.2f})")
    else:
        st.write("aucun (lance prepare_identification_assets.py)")
    st.caption("Backbone : EfficientNetB0 (leger, deployable). Les autres catégories "
               "sont listées mais non prises en charge à ce stade.")


# ──────────────────────────────────────────────────────────────────────────
# CONTENU PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────
st.title("Identification du type de défaut — Transfer Learning")
st.caption("Un modèle EfficientNetB0 par catégorie (classification multi-classe : good + types de défaut). "
           "Complément de la partie détection ; voir le rapport final pour le détail.")

t_demo, t_perf = st.tabs(["Démo — identification", "Performances"])

# ══════════════════════════════════════════════════════════════════════════
# ONGLET DEMO
# ══════════════════════════════════════════════════════════════════════════
with t_demo:
    # colonne gauche = controles (categorie + metriques + choix image)
    # colonne droite = matrice de confusion
    col_gauche, col_droite = st.columns([1, 1.3])

    image_choisie = None
    with col_gauche:
        categorie = st.selectbox("Catégorie de la pièce", C.CATEGORIES)
        supporte = categorie in dispo

        if not supporte:
            st.info(f"⚠️ Catégorie « {categorie} » non prise en charge à ce stade.")
            if dispo:
                st.caption("Catégories disponibles en démo : " + ", ".join(dispo))
        else:
            meta = index[categorie]
            classes = meta["classes"]
            backbone = meta.get("backbone", "efficientnetb0")
            preprocess = I.get_preprocess(backbone)

            st.metric("F1 macro (EfficientNetB0)", f"{meta['f1_macro']:.3f}")
            st.metric("Nombre de classes", len(classes))
            st.caption(f"{meta.get('n_test', '?')} images de test")

            st.markdown("#### Choisir une image")
            source = st.radio("Source", ["Exemple embarqué", "Charger une image"],
                              horizontal=True)
            if source == "Exemple embarqué":
                exemples = I.images_exemple(categorie)
                if exemples:
                    noms = [Path(p).name for p in exemples]
                    i = st.selectbox("Image", range(len(noms)), format_func=lambda k: noms[k])
                    image_choisie = exemples[i]
                else:
                    st.caption("(pas d'image d'exemple embarqué pour cette catégorie)")
            else:
                up = st.file_uploader("Image (.png / .jpg)", type=["png", "jpg", "jpeg"])
                if up is not None:
                    image_choisie = up

    with col_droite:
        if supporte:
            png_conf = I.DATA_DIR / f"{categorie}_confusion.png"
            if png_conf.exists():
                st.image(str(png_conf),
                         caption=f"{categorie} — matrice de confusion (jeu de test)",
                         use_container_width=True)
            else:
                st.caption("(matrice de confusion non disponible)")

    # ─── resultat : pleine largeur sous les deux colonnes ────────────────────
    if supporte and image_choisie is not None:
        from tensorflow import keras
        img = keras.utils.load_img(image_choisie, target_size=C.IMG_SIZE)
        image_rgb = np.array(img).astype("uint8")
        x = preprocess(np.array(img, dtype="float32")[None, ...])

        with st.spinner("Chargement du modele et prediction..."):
            modele = I.charge_modele(categorie)
            proba = modele.predict(x, verbose=0)[0]

        i_pred = int(proba.argmax())
        classe_pred = classes[i_pred]
        confiance = float(proba[i_pred])

        st.divider()
        st.markdown("#### Resultat")
        if classe_pred == "good":
            st.success(f"### ✅ {classe_pred} (conforme) — confiance {confiance*100:.1f} %")
        else:
            st.error(f"### ❌ Defaut : {classe_pred} — confiance {confiance*100:.1f} %")

        # 3 colonnes : image | probabilites | Grad-CAM
        c_img, c_proba, c_gcam = st.columns([1, 2, 1])
        with c_img:
            st.markdown("**Image analysee**")
            st.image(image_rgb, use_container_width=True)
        with c_proba:
            st.markdown("**Probabilites par classe**")
            ordre = np.argsort(proba)[::-1]
            df_proba = pd.DataFrame({"probabilite": [float(proba[k]) for k in ordre]},
                                    index=[classes[k] for k in ordre])
            st.bar_chart(df_proba)
        with c_gcam:
            st.markdown("**Grad-CAM**")
            try:
                heat = I.calcule_gradcam(modele, x, i_pred)
                overlay = I.superpose_gradcam(image_rgb, heat)
                st.image(overlay, caption=f"Zones decisives pour « {classe_pred} »",
                         use_container_width=True)
            except Exception as e:
                st.info(f"Grad-CAM indisponible ({e}).")

        st.caption(
            f"Modele : {backbone} — F1 macro {meta['f1_macro']:.3f} sur "
            f"{meta.get('n_test', '?')} images de test. Les scores par classe reposent "
            "sur peu d'images : a interpreter avec prudence.")
    elif supporte:
        st.info("Choisis une image d'exemple ou uploade-en une pour lancer l'identification.")

# ══════════════════════════════════════════════════════════════════════════
# ONGLET PERFORMANCES
# ══════════════════════════════════════════════════════════════════════════
with t_perf:
    st.header("Performances de l'identification (EfficientNetB0, 15 catégories)")
    st.caption("F1 macro par catégorie (benchmark complet). Les catégories chargées en demo "
               "live sont marquées. Métrique principale : F1 macro (pas l'accuracy), "
               "car les classes sont déséquilibrées.")

    # on prepare le tableau AVANT les colonnes (les deux colonnes s'en servent)
    df = pd.DataFrame({
        "Categorie": list(C.F1_EFFICIENTNET.keys()),
        "F1 macro (EfficientNetB0)": list(C.F1_EFFICIENTNET.values()),
    }).sort_values("F1 macro (EfficientNetB0)", ascending=False)
    df["Demo live"] = df["Categorie"].apply(lambda c: "✅ oui" if c in dispo else "—")

    perf_tab, perf_graph = st.columns([1, 2])
    with perf_tab:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with perf_graph:
        import matplotlib.pyplot as plt
        d = df.sort_values("F1 macro (EfficientNetB0)")
        couleurs = [C.COLOR_ACCENT if c in dispo else C.COLOR_PRIMARY for c in d["Categorie"]]
        fig, ax = plt.subplots(figsize=(9, 5.4))
        ax.barh(d["Categorie"], d["F1 macro (EfficientNetB0)"], color=couleurs)
        for i, v in enumerate(d["F1 macro (EfficientNetB0)"]):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
        ax.axvline(0.599, color="#555", ls="--", lw=1)
        ax.text(0.599, -1.2, "Moyenne = 0.599", fontsize=8, color="#555", ha="center")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("F1 macro")
        st.pyplot(fig)

    st.info("Lecture : les textures a defauts francs (leather, tile, wood) et bottle sont bien "
            "identifiees ; grid, capsule, pill, cable s'effondrent (defauts fins, nombreuses "
            "classes, peu d'images). Le facteur limitant est la volumetrie par type de defaut, "
            "pas le backbone.")
    st.caption("Barres oranges = categories disponibles en demo live dans l'onglet Demo. "
               "Le reste est documente ici mais non charge (app legere).")
