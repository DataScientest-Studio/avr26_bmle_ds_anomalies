# -*- coding: utf-8 -*-
"""
plot_confusion_heatmaps.py - Heatmaps des matrices de confusion pour les
modeles CNN 15 categories deja entraines (64x64 Flatten et GAP), a partir
des metriques deja calculees (pas de reentrainement).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_DIR / "results"
FIGS_DIR = REPO_DIR / "figures"

MODELS = [
    ("cnn_15cat_metrics.json", "CNN 15 categories 64x64 (Flatten)", "cnn_15cat_confusion_heatmap.png"),
    ("cnn_15cat_gap_metrics.json", "CNN 15 categories 64x64 (GAP)", "cnn_15cat_gap_confusion_heatmap.png"),
]

for metrics_file, title, out_name in MODELS:
    with open(RESULTS_DIR / metrics_file) as f:
        data = json.load(f)
    cm = np.array(data["confusion_matrix"])
    auc = data["roc_auc"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Conforme", "Defectueux"],
                yticklabels=["Conforme", "Defectueux"])
    plt.xlabel("Predit")
    plt.ylabel("Reel")
    plt.title(f"{title}\nROC-AUC={auc:.3f}")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / out_name, dpi=120)
    plt.close()
    print(f"{out_name} sauvegarde.")