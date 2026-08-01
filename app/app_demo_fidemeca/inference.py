# -*- coding: utf-8 -*-
"""
inference.py — Extraction de features et scoring d'anomalie pour la démo Streamlit.

Principe : on réutilise EXACTEMENT la logique de feature_engineering.py (Ludovic).
- Si feature_engineering.py est importable (présent sur la VM), on importe ses fonctions.
- Sinon, on utilise les ré-implémentations locales ci-dessous (mêmes formules).

⚠️ Aucune fonction ici ne réentraîne un modèle : on charge des artefacts déjà produits
(templates .npy, modèle joblib, autoencodeur Keras) ou on calcule un score déterministe
basé sur la différence au template (baseline).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

import config as C

# --- Dépendances image optionnelles (présentes sur la VM, pas garanties en local) ---
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# On tente d'importer le module des collègues (source de vérité des features).
try:
    import feature_engineering as FE   # nécessite feature_engineering.py dans le PYTHONPATH
    _HAS_FE = True
except Exception:
    FE = None
    _HAS_FE = False


# ════════════════════════════════════════════════════════════════════════
# PRÉTRAITEMENT
# ════════════════════════════════════════════════════════════════════════
def to_rgb_array(pil_img, size=C.TARGET_SIZE) -> np.ndarray:
    """Convertit une image PIL en array RGB uint8 redimensionné (HxWx3)."""
    img = pil_img.convert("RGB").resize(size)
    return np.asarray(img).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════
# TEMPLATES (moyenne des images train/good d'une catégorie)
# ════════════════════════════════════════════════════════════════════════
def load_template(category: str):
    """Charge le template .npy d'une catégorie s'il existe (généré par prepare_assets.py)."""
    p = C.TEMPLATES_DIR / f"{category}.npy"
    if p.exists():
        return np.load(p)
    return None


def available_templates() -> list[str]:
    if not C.TEMPLATES_DIR.exists():
        return []
    return sorted(p.stem for p in C.TEMPLATES_DIR.glob("*.npy"))


# ════════════════════════════════════════════════════════════════════════
# CARTE DE DIFFÉRENCE AU TEMPLATE (cœur de la détection non-supervisée)
# ════════════════════════════════════════════════════════════════════════
def diff_map(img_rgb: np.ndarray, template_rgb: np.ndarray):
    """
    Renvoie (carte_diff_gris, masque_binaire, dict_features_diff).
    Reproduit diff_features() de feature_engineering.py :
    absdiff -> GaussianBlur -> seuillage Otsu -> érosion/dilatation -> composantes connexes.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) requis pour la carte de différence.")
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_tpl = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray_img, gray_tpl)
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    n_comp, _, stats_cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    feats = {
        "diff_area_ratio":    float(mask.sum() / 255) / mask.size,
        "diff_n_components":  max(n_comp - 1, 0),
        "diff_max_blob_area": float(areas.max()) if len(areas) else 0.0,
        "diff_mean_intensity": float(diff.mean()),
    }
    return diff, mask, feats


# ════════════════════════════════════════════════════════════════════════
# FEATURES COMPLÈTES (12 colonnes = FEATURE_COLS)
# ════════════════════════════════════════════════════════════════════════
def extract_features(img_rgb: np.ndarray, template_rgb: np.ndarray) -> dict:
    """
    Renvoie les 12 features du modèle. Si feature_engineering.py est importable,
    on utilise SES fonctions (source de vérité) ; sinon, ré-implémentation locale.
    """
    if _HAS_FE:
        feats = {}
        feats.update(FE.diff_features(img_rgb, template_rgb))
        feats.update(FE.texture_edge_features(img_rgb))
        return {k: feats[k] for k in C.FEATURE_COLS}

    # --- Fallback local (mêmes formules que feature_engineering.py) ---
    if not (_HAS_CV2 and _HAS_SKIMAGE):
        raise RuntimeError("cv2 + scikit-image requis pour extraire les features en local.")
    _, _, dfeat = diff_map(img_rgb, template_rgb)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_var = float(cv2.Sobel(gray, cv2.CV_64F, 1, 0).var())
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean() / 255)
    gq = (gray.astype(np.float64) * (C.GLCM_LEVELS - 1) / 255).astype(np.uint8)
    glcm = graycomatrix(gq, distances=[1], angles=[0], levels=C.GLCM_LEVELS,
                        symmetric=True, normed=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    hist = hist[hist > 0]
    lbp_entropy = float(-(hist * np.log2(hist)).sum())
    feats = {
        **dfeat,
        "lap_var": lap_var, "sobel_var": sobel_var, "edge_density": edge_density,
        "glcm_contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "glcm_energy": float(graycoprops(glcm, "energy")[0, 0]),
        "glcm_correlation": float(graycoprops(glcm, "correlation")[0, 0]),
        "lbp_entropy": lbp_entropy,
    }
    return {k: feats[k] for k in C.FEATURE_COLS}


# ════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES (sans réentraînement)
# ════════════════════════════════════════════════════════════════════════
def load_feature_model():
    """Charge (modèle, scaler) joblib si présents, sinon (None, None)."""
    try:
        import joblib
        model = joblib.load(C.FEATURE_MODEL_PATH) if C.FEATURE_MODEL_PATH.exists() else None
        scaler = joblib.load(C.SCALER_PATH) if C.SCALER_PATH.exists() else None
        return model, scaler
    except Exception:
        return None, None


def load_autoencoder(category: str | None = None):
    """Charge l'autoencodeur Keras d'une catégorie (models/autoencoder_<categorie>.keras).
    Sans argument, retourne le premier autoencodeur disponible (rétro-compatibilité)."""
    try:
        from tensorflow import keras
        if category is not None:
            p = C.AUTOENCODER_DIR / f"autoencoder_{category}.keras"
            return keras.models.load_model(p, compile=False) if p.exists() else None
        if C.AUTOENCODER_PATH.exists():
            return keras.models.load_model(C.AUTOENCODER_PATH, compile=False)
        avail = available_autoencoders()
        return load_autoencoder(avail[0]) if avail else None
    except Exception:
        return None


def available_autoencoders() -> list[str]:
    """Catégories pour lesquelles un autoencodeur .keras est présent dans models/."""
    if not C.AUTOENCODER_DIR.exists():
        return []
    found = {p.stem.replace("autoencoder_", "") for p in C.AUTOENCODER_DIR.glob("autoencoder_*.keras")}
    return [c for c in C.AUTOENCODER_CATEGORIES if c in found] or sorted(found)


# ════════════════════════════════════════════════════════════════════════
# CNN SUPERVISÉ (15 catégories, entraîné par l'équipe — voir models/cnn_binary_*.keras)
# ════════════════════════════════════════════════════════════════════════
def available_cnn_variants() -> list[str]:
    return [name for name, p in C.CNN_MODELS.items() if p.exists()]


def load_cnn(variant: str):
    """Charge une variante du CNN binaire 15 catégories (aucun réentraînement)."""
    try:
        p = C.CNN_MODELS.get(variant)
        if p is None or not p.exists():
            return None
        from tensorflow import keras
        return keras.models.load_model(p, compile=False)
    except Exception:
        return None


def _resize_to_model(img_rgb: np.ndarray, model) -> np.ndarray:
    """Redimensionne l'image à l'entrée attendue par le modèle (peut différer de TARGET_SIZE)."""
    h, w = model.input_shape[1], model.input_shape[2]
    if (img_rgb.shape[0], img_rgb.shape[1]) == (h, w):
        return img_rgb
    if _HAS_CV2:
        return cv2.resize(img_rgb, (w, h))
    from PIL import Image as _Image
    return np.array(_Image.fromarray(img_rgb).resize((w, h)))


def score_cnn(img_rgb: np.ndarray, model) -> dict:
    """Score du CNN supervisé = probabilité de défaut (sortie sigmoïde), verdict au seuil 0,5."""
    x = _resize_to_model(img_rgb, model).astype("float32") / 255.0
    proba = float(model.predict(x[None, ...], verbose=0)[0][0])
    return {"score": proba, "verdict": proba > 0.5}


# ════════════════════════════════════════════════════════════════════════
# TRANSFER LEARNING RESNET50 (catégorie "bottle" — Ludovic)
# ════════════════════════════════════════════════════════════════════════
def _reassemble_parts(target_path) -> bool:
    """Reconstitue un fichier découpé en morceaux <fichier>.part-* (contournement de la
    limite GitHub de 100 Mo/fichier pour les gros modèles Keras). Ne fait rien si le
    fichier complet existe déjà ; renvoie False si aucun morceau n'est trouvé."""
    if target_path.exists():
        return True
    parts = sorted(target_path.parent.glob(target_path.name + ".part-*"))
    if not parts:
        return False
    try:
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with open(tmp_path, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
        tmp_path.rename(target_path)
        return True
    except Exception:
        return False


def available_transfer_learning() -> list[str]:
    """Catégories pour lesquelles un modèle de transfer learning est disponible
    (fichier complet déjà présent, ou reconstitué à partir de ses morceaux)."""
    if _reassemble_parts(C.TRANSFER_LEARNING_PATH):
        return C.TRANSFER_LEARNING_CATEGORIES
    return []


def load_transfer_learning(category: str):
    """Charge le modèle ResNet50 (transfer learning) d'une catégorie, sans réentraînement."""
    if category not in C.TRANSFER_LEARNING_CATEGORIES:
        return None
    try:
        if not _reassemble_parts(C.TRANSFER_LEARNING_PATH):
            return None
        from tensorflow import keras
        return keras.models.load_model(C.TRANSFER_LEARNING_PATH, compile=False)
    except Exception:
        return None


def score_transfer_learning(img_rgb: np.ndarray, model) -> dict:
    """Score du transfer learning ResNet50 = probabilité de défaut (sortie sigmoïde).
    Utilise le prétraitement spécifique ResNet50 (preprocess_input : conversion BGR +
    centrage sur les statistiques ImageNet), différent de la simple normalisation /255
    utilisée par le CNN from scratch — c'est celui utilisé à l'entraînement (voir
    transfer_learning_mvtec_8.py sur le dépôt GitHub d'équipe)."""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img_resized = _resize_to_model(img_rgb, model)
    x = preprocess_input(img_resized.astype("float32"))
    proba = float(model.predict(x[None, ...], verbose=0)[0][0])
    return {"score": proba, "verdict": proba > 0.5}


# ════════════════════════════════════════════════════════════════════════
# SCORING
# ════════════════════════════════════════════════════════════════════════
def score_baseline(img_rgb, template_rgb) -> dict:
    """
    Score d'anomalie BASELINE (fonctionne sans modèle entraîné) :
    intensité moyenne de la différence au template. Plus c'est haut, plus c'est anormal.
    """
    _, mask, feats = diff_map(img_rgb, template_rgb)
    return {"score": feats["diff_mean_intensity"], "features": feats, "mask": mask}


def score_autoencoder(img_rgb, model) -> dict:
    """Erreur de reconstruction de l'autoencodeur = score d'anomalie.
    Redimensionne à l'entrée attendue par le modèle (nos autoencodeurs sont en 64×64,
    différent de TARGET_SIZE=128×128 utilisé par le reste de l'appli)."""
    img_resized = _resize_to_model(img_rgb, model)
    x = img_resized.astype("float32") / 255.0
    recon = model.predict(x[None, ...], verbose=0)[0]
    err_map = np.mean((x - recon) ** 2, axis=-1)   # carte d'erreur par pixel
    return {"score": float(err_map.mean()), "recon": (recon * 255).astype(np.uint8),
            "err_map": err_map}
