# 🔍 Streamlit — Détection d'anomalies (Fidémeca / MVTec AD)

Application de démonstration du projet DS. Cinq onglets : Contexte, Exploration,
Pré-processing & Feature Engineering, Modélisation & Démo, Conclusion.

> **Important (consigne mentor)** : l'appli **ne réentraîne jamais** de modèle.
> Elle **charge** des artefacts déjà produits (`models/`). Un mode *baseline*
> (différence au template) la rend fonctionnelle même sans modèle final.

## 🚀 Lancer l'appli

```bash
cd 05_Streamlit
python -m venv .venv && source .venv/bin/activate      # (Windows : .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

L'appli s'ouvre sur http://localhost:8501. **L'onglet Exploration marche immédiatement**
(les chiffres sont dans `data/mvtec_summary.csv`).

## 🧩 Activer la démo de détection (2 étapes)

1. **Générer les templates** (moyenne des images `train/good` par catégorie — déterministe, pas d'entraînement) :
   ```bash
   python prepare_assets.py /chemin/vers/dataBase_resized_128
   ```
   → crée `models/templates/<categorie>.npy`. Le mode **Baseline** est alors actif.

2. **Déposer le modèle final** (quand l'équipe l'a entraîné, ailleurs) dans `models/` :
   - Modèle sur features : `models/anomaly_model.joblib` (+ `models/scaler.joblib`)
   - ou Autoencodeur Keras : `models/autoencoder.keras`

   L'appli les détecte automatiquement (voir la barre latérale).

## 🗂️ Structure

```
05_Streamlit/
├── app.py               # appli principale (5 onglets)
├── config.py            # ⚙️ chemins & constantes — SEUL fichier à adapter
├── inference.py         # extraction de features + scoring (réutilise feature_engineering.py)
├── prepare_assets.py    # génère les templates .npy (à lancer une fois)
├── requirements.txt
├── data/
│   └── mvtec_summary.csv   # résumé du dataset (bundlé)
├── models/
│   └── templates/          # <categorie>.npy (générés)
└── .streamlit/config.toml  # thème
```

## 🔗 Lien avec le code de l'équipe

`inference.py` tente d'importer **`feature_engineering.py`** (le pipeline de Ludovic) pour
utiliser exactement les mêmes fonctions de features (`diff_features`, `texture_edge_features`).
Placez `feature_engineering.py` dans ce dossier ou dans le `PYTHONPATH`. À défaut, `inference.py`
utilise des ré-implémentations locales identiques.

## ✅ Checklist soutenance (mentor)
- [ ] Appli esthétique, plusieurs onglets — **fait**
- [ ] Fonctionnelle sans bug — tester le parcours complet
- [ ] **Sans réentraînement** — modèles chargés depuis `models/` — **fait**
- [ ] Relier les résultats à la problématique métier — onglets Contexte & Conclusion
- [ ] Répéter la démo 1–2 fois avant le jour J

## 📝 À adapter par l'équipe
- `config.py` : `DATASET_ROOT`, chemins des modèles.
- Onglet *Modélisation* : la convention de sortie du modèle (`predict` = 1/-1 = anomalie)
  est à ajuster selon le modèle réellement retenu (IsolationForest, OneClassSVM, autoencodeur…).
