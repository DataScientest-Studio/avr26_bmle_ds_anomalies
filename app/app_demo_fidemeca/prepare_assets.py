# -*- coding: utf-8 -*-
"""
prepare_assets.py — Génère les artefacts nécessaires à la démo Streamlit (À LANCER UNE FOIS).

Ce script NE réentraîne AUCUN modèle. Il calcule seulement, pour chaque catégorie,
le TEMPLATE = moyenne pixel-à-pixel des images train/good (opération déterministe),
et le sauvegarde dans models/templates/<categorie>.npy.

Usage :
    python prepare_assets.py                       # utilise DATASET_ROOT de config.py
    python prepare_assets.py /chemin/vers/dataset  # chemin explicite

Le dataset attendu suit l'arborescence MVTec : <root>/<categorie>/train/good/*.png
"""
import sys
from pathlib import Path
import numpy as np

import config as C

try:
    import cv2
except Exception:
    raise SystemExit("OpenCV requis : pip install opencv-python-headless")


def build_template(cat_dir: Path, size=C.TARGET_SIZE):
    """Moyenne des images train/good d'une catégorie → template RGB uint8."""
    paths = sorted((cat_dir / "train" / "good").glob("*.png"))
    if not paths:
        return None
    imgs = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, size)
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not imgs:
        return None
    return np.stack(imgs).astype(np.float64).mean(axis=0).astype(np.uint8)


def main(root: Path):
    C.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        raise SystemExit(f"Dataset introuvable : {root}\n"
                         f"Adapte DATASET_ROOT dans config.py ou passe le chemin en argument.")
    n = 0
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tpl = build_template(cat_dir)
        if tpl is None:
            print(f"  ⚠️  ignoré (pas d'images train/good) : {cat_dir.name}")
            continue
        out = C.TEMPLATES_DIR / f"{cat_dir.name}.npy"
        np.save(out, tpl)
        print(f"  ✅ template {cat_dir.name} → {out.name}")
        n += 1
    print(f"\n{n} template(s) généré(s) dans {C.TEMPLATES_DIR}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else C.DATASET_ROOT
    main(root)
