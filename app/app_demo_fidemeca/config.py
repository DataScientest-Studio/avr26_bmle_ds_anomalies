# -*- coding: utf-8 -*-
"""
config.py — Chemins et constantes de l'application Streamlit.
⚠️ SEULE SECTION À ADAPTER selon l'environnement (VM Liora, local, etc.).
"""
from pathlib import Path

# Racine de l'application (dossier 05_Streamlit)
APP_ROOT = Path(__file__).resolve().parent

# --- Données ---------------------------------------------------------------
# Résumé du dataset (bundlé dans le repo → l'onglet Exploration marche sans le dataset).
SUMMARY_CSV = APP_ROOT / "data" / "mvtec_summary.csv"

# CSV des features déjà calculées par feature_engineering.py (optionnel).
FEATURES_CSV = APP_ROOT / "data" / "features_engineered.csv"

# Racine des images redimensionnées (optionnel — pour la démo sur images réelles).
# Sur la VM : adapter au dossier dataBase_resized_128 / Resized_128.
DATASET_ROOT = APP_ROOT / "data" / "dataBase_resized_128"

# --- Modèles ---------------------------------------------------------------
# Templates moyens par catégorie (fichiers .npy), générés par prepare_assets.py.
TEMPLATES_DIR = APP_ROOT / "models" / "templates"

# Modèle de classification sur features (joblib/pickle) — à déposer par l'équipe.
FEATURE_MODEL_PATH = APP_ROOT / "models" / "anomaly_model.joblib"
SCALER_PATH        = APP_ROOT / "models" / "scaler.joblib"

# Autoencodeur Keras (.keras ou .h5) — à déposer par l'équipe.
AUTOENCODER_PATH = APP_ROOT / "models" / "autoencoder.keras"

# Autoencodeurs entraînés par catégorie (un fichier models/autoencoder_<categorie>.keras par catégorie).
AUTOENCODER_DIR = APP_ROOT / "models"

# CNN supervisé (15 catégories) entraîné par l'équipe sur GitHub (Paul/Ludovic) — 3 variantes
# comparées lors de l'ablation (voir rapport, Partie IV). Aucune n'est réentraînée ici.
CNN_MODELS = {
    "CNN 64×64 — Flatten (production, AUC 0.748)": APP_ROOT / "models" / "cnn_binary_15cat_best.keras",
    "CNN 128×128 — Flatten (AUC 0.7295)": APP_ROOT / "models" / "cnn_binary_15cat_128_best.keras",
    "CNN 64×64 — GlobalAveragePooling, ablation (AUC 0.542)": APP_ROOT / "models" / "cnn_binary_15cat_gap_best.keras",
}

# Transfer Learning ResNet50 (détection binaire, catégorie "bottle" uniquement — Ludovic).
# Poids de ~205 Mo : fractionnés en morceaux <50 Mo (resnet50_bottle_detection.keras.part-*)
# pour respecter la limite GitHub de 100 Mo/fichier, reconstitués automatiquement au 1er chargement.
TRANSFER_LEARNING_PATH = APP_ROOT / "models" / "resnet50_bottle_detection.keras"
TRANSFER_LEARNING_CATEGORIES = ["bottle"]

# --- Paramètres image ------------------------------------------------------
DIMENSION = 128
TARGET_SIZE = (DIMENSION, DIMENSION)   # (largeur, hauteur)

# 12 features utilisées par le modèle (identiques à FEATURE_COLS de feature_engineering.py)
FEATURE_COLS = [
    "diff_area_ratio", "diff_n_components", "diff_max_blob_area", "diff_mean_intensity",
    "lap_var", "sobel_var", "edge_density",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "lbp_entropy",
]

GLCM_LEVELS = 24

# Les 15 catégories du dataset MVTec AD (templates disponibles pour les 15).
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Catégories pilotes couvertes par le modèle sur features (RF) et par les autoencodeurs
# par catégorie (les autres modèles de l'équipe — EfficientNet/ResNet/PaDiM — n'ont pas
# leurs poids versionnés sur GitHub : trop volumineux pour être ré-entraînés/committés).
FEATURE_MODEL_CATEGORIES = ["bottle", "carpet", "screw"]
AUTOENCODER_CATEGORIES = ["bottle", "carpet", "screw"]

# Palette
COLOR_GOOD = "#2E7D32"
COLOR_ANOMALY = "#C62828"
COLOR_PRIMARY = "#1F3864"
