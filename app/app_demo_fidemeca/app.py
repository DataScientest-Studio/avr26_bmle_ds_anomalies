# -*- coding: utf-8 -*-
"""
app.py — Application Streamlit — Détection d'anomalies (Projet DS Fidémeca / MVTec AD).

Lancement :  streamlit run app.py

Application recentrée sur la Partie IV du rapport (Modélisation). Le contexte, l'exploration,
le pré-processing et la conclusion sont présentés dans le support de soutenance (PowerPoint) ;
cette appli se concentre sur la démonstration des modèles.

Structure :
  1. Démo interactive       — upload d'une image → score d'anomalie (baseline / RF / AutoEncoder / CNN)
  2. Comparatif des modèles — tous les modèles testés par l'équipe (voir rapport final), avec ou
     sans démo live selon la disponibilité des poids sur GitHub.

Aucun modèle n'est réentraîné : on charge des artefacts déjà produits (models/...), qu'ils
viennent du dépôt Streamlit lui-même ou du dépôt GitHub d'équipe (CNN 15 catégories).
"""
import numpy as np
import pandas as pd
import streamlit as st

import config as C

# ──────────────────────────────────────────────────────────────────────────
# CONFIG PAGE + STYLE
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fidémeca — Modélisation", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .main .block-container {{ padding-top: 2rem; }}
  h1, h2, h3 {{ color: {C.COLOR_PRIMARY}; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Détection d'anomalies")
    st.caption("Projet fil rouge — Bootcamp MLE, cohorte Avril 2026")
    st.markdown("**Équipe :** Paul Fournel · Alex Mac-Kame · Ludovic Marquant · Fabrice Masola")
    st.markdown("**Entreprise (cas) :** Fidémeca — mécanique de précision")
    st.markdown("**Dataset :** MVTec AD (15 catégories, 5 354 images)")
    st.divider()
    st.caption("Modèles chargés (aucun réentraînement) :")
    from inference import (available_templates, load_feature_model, available_autoencoders,
                           available_cnn_variants, available_transfer_learning)
    _tpls = available_templates()
    _fm, _sc = load_feature_model()
    _ae_cats = available_autoencoders()
    _cnn_variants = available_cnn_variants()
    _tl_cats = available_transfer_learning()
    st.write("Templates :", f"{len(_tpls)} disponibles" if _tpls else "aucun")
    st.write("Modèle features (RF) :", "disponible" if _fm else "absent")
    st.write("Autoencodeurs :", ', '.join(_ae_cats) if _ae_cats else "aucun")
    st.write("CNN 15 catégories :", f"{len(_cnn_variants)} variante(s)" if _cnn_variants else "absent")
    st.write("Transfer Learning (ResNet50) :", ', '.join(_tl_cats) if _tl_cats else "absent")
    st.caption("Détail de tous les modèles testés par l'équipe dans l'onglet Comparatif.")


# ──────────────────────────────────────────────────────────────────────────
# CONTENU PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────
st.title("Modélisation — détection d'anomalies")
st.caption("Partie IV du rapport final. Contexte, exploration, pré-processing et conclusion : "
           "voir le support de soutenance (PowerPoint).")

t_model, t_cmp = st.tabs(["Démo interactive", "Comparatif des modèles"])

# ══════════════════════════════════════════════════════════════════════════
# DÉMO INTERACTIVE
# ══════════════════════════════════════════════════════════════════════════
with t_model:
    st.markdown("""
La démo **charge un modèle déjà entraîné** (aucun réentraînement dans l'appli, conformément
aux consignes du mentor). Quatre modes selon ce qui est disponible dans `models/` :
""")
    from inference import (load_template, to_rgb_array, score_baseline, extract_features,
                           load_feature_model, load_autoencoder, score_autoencoder,
                           available_templates, available_autoencoders, available_cnn_variants,
                           load_cnn, score_cnn, available_transfer_learning,
                           load_transfer_learning, score_transfer_learning)

    ae_cats = available_autoencoders()
    cnn_variants = available_cnn_variants()
    tl_cats = available_transfer_learning()
    mode = st.radio("Mode de détection", [
        "Baseline — différence au template (toujours dispo)",
        "Modèle sur features — Random Forest (joblib)",
        f"AutoEncodeur (Keras) — {', '.join(ae_cats) if ae_cats else 'aucun disponible'}",
        f"CNN supervisé — 15 catégories (Keras) — {len(cnn_variants)} variante(s)",
        f"Transfer Learning — ResNet50 (Keras) — {', '.join(tl_cats) if tl_cats else 'aucun disponible'}",
    ], horizontal=False)

    tpls = available_templates()
    colL, colR = st.columns([1, 2])
    cat = colL.selectbox("Catégorie de la pièce", tpls or C.CATEGORIES)
    variant = None
    if mode.startswith("CNN"):
        variant = colL.selectbox("Variante du CNN (ablation — voir Partie IV du rapport)", cnn_variants)
        seuil = 50.0
    else:
        seuil = colL.slider("Seuil de décision (anomalie si score >)", 0.0, 60.0, 12.0, 0.5)
    up = colR.file_uploader("Déposer une image de pièce à contrôler", type=["png", "jpg", "jpeg"])

    if up is not None:
        from PIL import Image
        img = to_rgb_array(Image.open(up))
        tpl = load_template(cat)
        st.image(img, width=220, caption="Pièce à contrôler")

        try:
            if mode.startswith("Baseline"):
                if tpl is None:
                    st.warning("Template manquant pour cette catégorie (voir prepare_assets.py).")
                else:
                    res = score_baseline(img, tpl)
                    score = res["score"]
                    verdict = score > seuil
                    st.metric("Score d'anomalie (diff. moyenne)", f"{score:.2f}",
                              delta="ANOMALIE" if verdict else "conforme",
                              delta_color="inverse" if verdict else "normal")
                    st.image(res["mask"], caption="Zones suspectes", width=220, clamp=True)
                    (st.error if verdict else st.success)(
                        "Pièce probablement DÉFECTUEUSE" if verdict else "Pièce conforme")

            elif mode.startswith("Modèle sur features"):
                if cat not in C.FEATURE_MODEL_CATEGORIES:
                    st.warning(f"Le modèle Random Forest a été entraîné sur les catégories pilotes "
                               f"**{', '.join(C.FEATURE_MODEL_CATEGORIES)}** uniquement. Le résultat sur "
                               f"« {cat} » n'a pas été validé par l'équipe — à titre indicatif.")
                model, scaler = load_feature_model()
                if model is None or tpl is None:
                    st.warning("Modèle joblib (`models/anomaly_model.joblib`) et/ou template absent.")
                else:
                    feats = extract_features(img, tpl)
                    X = np.array([[feats[c] for c in C.FEATURE_COLS]])
                    if scaler is not None:
                        X = scaler.transform(X)
                    pred = model.predict(X)[0]
                    verdict = int(pred) in (1, -1)  # -1 = anomalie pour IsolationForest/OneClassSVM
                    st.write("**Features extraites :**", {k: round(feats[k], 3) for k in C.FEATURE_COLS})
                    (st.error if verdict else st.success)(
                        "DÉFECTUEUSE" if verdict else "Conforme")

            elif mode.startswith("AutoEncodeur"):
                if cat not in ae_cats:
                    st.warning(f"Aucun autoencodeur entraîné pour « {cat} ». Catégories disponibles : "
                               f"{', '.join(ae_cats) if ae_cats else 'aucune'}.")
                else:
                    ae = load_autoencoder(cat)
                    res = score_autoencoder(img, ae)
                    verdict = res["score"] > (seuil / 1000)  # échelle MSE
                    m1, m2, m3 = st.columns(3)
                    m1.image(img, caption="Entrée", use_container_width=True)
                    m2.image(res["recon"], caption="Reconstruction", use_container_width=True)
                    m3.image(res["err_map"], caption="Carte d'erreur", use_container_width=True, clamp=True)
                    st.metric("Erreur de reconstruction (MSE)", f"{res['score']:.5f}")
                    (st.error if verdict else st.success)(
                        "DÉFECTUEUSE" if verdict else "Conforme")

            elif mode.startswith("CNN"):
                cnn = load_cnn(variant) if variant else None
                if cnn is None:
                    st.warning("Aucune variante de CNN disponible dans `models/`.")
                else:
                    res = score_cnn(img, cnn)
                    st.caption("Modèle entraîné sur les 15 catégories confondues (pas de template requis) "
                               "— voir l'ablation Flatten vs GAP et 64×64 vs 128×128, Partie IV du rapport.")
                    st.metric("Probabilité de défaut (sortie sigmoïde)", f"{res['score']:.3f}")
                    st.progress(min(max(res["score"], 0.0), 1.0))
                    (st.error if res["verdict"] else st.success)(
                        "Pièce probablement DÉFECTUEUSE" if res["verdict"] else "Pièce conforme")

            else:  # Transfer Learning ResNet50
                if cat not in tl_cats:
                    st.warning(f"Aucun modèle ResNet50 entraîné pour « {cat} ». Catégorie(s) "
                               f"disponible(s) : {', '.join(tl_cats) if tl_cats else 'aucune'}.")
                else:
                    tl_model = load_transfer_learning(cat)
                    if tl_model is None:
                        st.warning("Modèle ResNet50 introuvable ou erreur au chargement.")
                    else:
                        res = score_transfer_learning(img, tl_model)
                        st.caption("Backbone ResNet50 pré-entraîné sur ImageNet, tête de classification "
                                   "fine-tunée (exploration de Ludovic — voir dépôt GitHub d'équipe, "
                                   "`transfer_learning_mvtec_8.py`).")
                        st.metric("Probabilité de défaut (sortie sigmoïde)", f"{res['score']:.3f}")
                        st.progress(min(max(res["score"], 0.0), 1.0))
                        (st.error if res["verdict"] else st.success)(
                            "Pièce probablement DÉFECTUEUSE" if res["verdict"] else "Pièce conforme")
        except Exception as e:
            st.error(f"Erreur pendant le scoring : {e}")
    else:
        st.info("Dépose une image ci-dessus pour lancer la détection.")

    st.caption("Le mode baseline fonctionne dès qu'un template existe. Les autres modes s'appuient "
               "sur des modèles réellement entraînés par l'équipe et versionnés sur GitHub "
               "(DataScientest-Studio/avr26_bmle_ds_anomalies).")

# ══════════════════════════════════════════════════════════════════════════
# COMPARATIF DE TOUS LES MODÈLES DE L'ÉQUIPE
# ══════════════════════════════════════════════════════════════════════════
with t_cmp:
    st.header("Comparatif de tous les modèles testés par l'équipe")
    st.caption("Synthèse du rapport final — code complet sur le dépôt GitHub "
               "DataScientest-Studio/avr26_bmle_ds_anomalies (branches de chaque membre, fusionnées sur main).")

    st.subheader("Partie III — Reconnaissance de la catégorie d'objet")
    st.caption("Jeu de 7 590 fichiers (391 images/catégorie après augmentation, 128×128×3).")
    reco_df = pd.DataFrame({
        "Modèle": ["Random Forest", "RNN Dense", "CNN", "LeNet", "AutoEncoder (test sur bottle)"],
        "Auteur(s)": ["Fabrice", "Fabrice", "Fabrice / Paul", "Fabrice", "Fabrice"],
        "Principe": ["Sur features engineered", "Réseau dense", "Convolution + pooling",
                     "Architecture LeNet", "Reconstruction"],
        "Erreurs de classement": ["47 images", "< 47 (non chiffré)", "4 images", "0", "0 (83/83 bottles)"],
    })
    st.dataframe(reco_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Partie IV — Détection d'anomalies (métrique : AUC-ROC)")
    st.caption("Moyenne sur les 15 catégories, sauf mention contraire. Source : rapport final de l'équipe.")

    modele_df = pd.DataFrame({
        "Modèle": ["AutoEncoder convolutif", "CNN scratch — 64×64 (Flatten)",
                   "CNN scratch — 128×128 (Flatten)", "CNN scratch — 64×64 (GAP, ablation)",
                   "Transfer Learning — EfficientNetB0", "Transfer Learning — ResNet50",
                   "PaDiM (features pré-entraînées)"],
        "Auteur(s)": ["Fabrice", "Paul (VM Liora)", "Ludovic", "Paul (ablation)", "Alex", "Alex", "Alex"],
        "Protocole": ["Non supervisé (one-class)", "Supervisé", "Supervisé", "Supervisé",
                      "Supervisé", "Supervisé", "Non supervisé (one-class)"],
        "AUC-ROC moyen": [0.718, 0.748, 0.7295, 0.542, 0.937, 0.930, None],
        "Démo live": ["Oui (bottle, carpet, screw)", "Oui", "Oui", "Oui", "Non — poids non versionnés",
                      "Non — poids entraînés (215 Mo), trop volumineux pour GitHub", "Non — non finalisé"],
    })
    st.dataframe(modele_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Benchmark complémentaire — 4 backbones sur les 15 catégories (Ludovic)")
    st.caption("Exploration indépendante de Ludovic (transfer_learning_mvtec_8.py, dépôt GitHub d'équipe) : "
               "comparaison de 4 backbones ImageNet sur la détection binaire, toutes catégories confondues. "
               "Protocole d'entraînement propre à cette exploration — chiffres à lire séparément de ceux "
               "retenus dans le rapport final (tableau ci-dessus, protocole d'Alex).")
    backbone_df = pd.DataFrame({
        "Backbone": ["ResNet50", "VGG16", "EfficientNetB0", "MobileNetV2"],
        "AUC-ROC moyen": [0.984, 0.981, 0.945, 0.891],
        "AUC-ROC écart-type": [0.013, 0.026, 0.077, 0.154],
        "F1 moyen": [0.963, 0.967, 0.941, 0.867],
    }).sort_values("AUC-ROC moyen", ascending=False)
    st.dataframe(backbone_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Comparaison globale — tous les modèles testés (AUC-ROC)")
    st.caption("Les deux tableaux ci-dessus sont fusionnés dans un seul graphique, à titre indicatif : "
               "barres bleues = protocole du rapport final (Alex/Fabrice/Paul), barres oranges = "
               "benchmark complémentaire de Ludovic. Deux protocoles distincts — ne pas comparer les "
               "valeurs terme à terme (ex. les deux lignes ResNet50).")

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    combined_df = pd.concat([
        modele_df.dropna(subset=["AUC-ROC moyen"])[["Modèle", "AUC-ROC moyen"]]
            .assign(Source="Rapport final", Modèle=lambda d: d["Modèle"] + " (rapport)"),
        backbone_df[["Backbone", "AUC-ROC moyen"]]
            .rename(columns={"Backbone": "Modèle"})
            .assign(Source="Benchmark Ludovic", Modèle=lambda d: d["Modèle"] + " (Ludovic, 15cat)"),
    ], ignore_index=True).sort_values("AUC-ROC moyen")

    source_colors = {"Rapport final": C.COLOR_PRIMARY, "Benchmark Ludovic": "#E07B00"}
    colors = [source_colors[s] for s in combined_df["Source"]]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(combined_df["Modèle"], combined_df["AUC-ROC moyen"], color=colors)
    for i, v in enumerate(combined_df["AUC-ROC moyen"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC-ROC moyen (15 catégories)")
    ax.axvline(0.99, color="#555", linestyle="--", linewidth=1)
    ax.text(0.99, -1.3, "État de l'art\n(PatchCore) ≈ 0.99", fontsize=8, color="#555", ha="center")
    ax.legend(handles=[mpatches.Patch(color=c, label=s) for s, c in source_colors.items()],
              loc="lower right", fontsize=8)
    st.pyplot(fig)

    st.info("**Ce qui a le plus fait progresser la performance :** le passage d'une approche non "
            "supervisée pure (AutoEncoder) au transfer learning supervisé (EfficientNetB0, ResNet50) — "
            "les features génériques ImageNet généralisent nettement mieux, en particulier sur les "
            "textures répétitives où l'AutoEncoder et le CNN from scratch échouaient le plus.")
    st.caption("EfficientNetB0/ResNet50 (rapport)/PaDiM ne sont pas chargés en démo live : leurs poids "
               "n'ont pas été versionnés sur GitHub — voir le rapport final pour le détail complet des "
               "courbes ROC et matrices de confusion par catégorie.")
    st.success("Le modèle ResNet50 de Ludovic (catégorie bottle) est disponible en démo live dans l'onglet "
               "Démo interactive : ses poids (205 Mo) dépassaient la limite GitHub de 100 Mo/fichier, ils "
               "ont été fractionnés en morceaux puis reconstitués automatiquement au chargement. VGG16, "
               "EfficientNetB0 et MobileNetV2 (benchmark Ludovic) restent sans démo live (poids non "
               "récupérés).")
