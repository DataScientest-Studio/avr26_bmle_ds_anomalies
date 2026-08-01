# -*- coding: utf-8 -*-
"""
feature_engineering.py — Extraction de features pour la détection d'anomalies (MVTec AD).

Reproduit la logique déjà documentée dans 05_Streamlit/inference.py (fallback local) :
diff au template (Otsu + composantes connexes) + texture/edge (Laplacien, Sobel, Canny,
GLCM, LBP). Sert de source de vérité unique pour l'entraînement ET pour la démo Streamlit.
"""
from __future__ import annotations
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

GLCM_LEVELS = 24

FEATURE_COLS = [
    "diff_area_ratio", "diff_n_components", "diff_max_blob_area", "diff_mean_intensity",
    "lap_var", "sobel_var", "edge_density",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "lbp_entropy",
]


def diff_features(img_rgb: np.ndarray, template_rgb: np.ndarray) -> dict:
    """Carte de différence au template : absdiff -> blur -> Otsu -> morpho -> composantes connexes."""
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
    return {
        "diff_area_ratio":     float(mask.sum() / 255) / mask.size,
        "diff_n_components":   max(n_comp - 1, 0),
        "diff_max_blob_area":  float(areas.max()) if len(areas) else 0.0,
        "diff_mean_intensity": float(diff.mean()),
    }


def texture_edge_features(img_rgb: np.ndarray) -> dict:
    """Features de texture/contours : Laplacien, Sobel, Canny, GLCM, LBP."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_var = float(cv2.Sobel(gray, cv2.CV_64F, 1, 0).var())
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean() / 255)

    gq = (gray.astype(np.float64) * (GLCM_LEVELS - 1) / 255).astype(np.uint8)
    glcm = graycomatrix(gq, distances=[1], angles=[0], levels=GLCM_LEVELS,
                         symmetric=True, normed=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    hist = hist[hist > 0]
    lbp_entropy = float(-(hist * np.log2(hist)).sum())

    return {
        "lap_var": lap_var, "sobel_var": sobel_var, "edge_density": edge_density,
        "glcm_contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "glcm_energy": float(graycoprops(glcm, "energy")[0, 0]),
        "glcm_correlation": float(graycoprops(glcm, "correlation")[0, 0]),
        "lbp_entropy": lbp_entropy,
    }


def extract_all_features(img_rgb: np.ndarray, template_rgb: np.ndarray) -> dict:
    feats = {}
    feats.update(diff_features(img_rgb, template_rgb))
    feats.update(texture_edge_features(img_rgb))
    return {k: feats[k] for k in FEATURE_COLS}
