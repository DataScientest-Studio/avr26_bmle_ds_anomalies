# 📋 Protocole — Application Streamlit (Détection d'anomalies Fidémeca)

Document à destination de l'équipe. Résume ce qui a été fait, où se trouve tout,
et comment lancer / mettre à jour / faire évoluer l'application de démonstration.

---

## 1. Ce que c'est

Une application **Streamlit** qui présente et démontre le projet de détection d'anomalies
(dataset MVTec AD), en **5 onglets** :

1. **🏭 Contexte** — problème métier Fidémeca, approche non-supervisée.
2. **🔍 Exploration** — les 5 graphiques du dataset (composition, % défauts, résolutions, modes couleur, types de défauts).
3. **🛠️ Pré-processing & FE** — resize 128×128, template moyen, 3 familles de features, top features discriminantes, + démo visuelle « carte de différence ».
4. **🤖 Modélisation & Démo** — on dépose une image → verdict conforme / défectueux + zones suspectes. 3 modes : *baseline* (différence au template), *modèle sur features* (joblib), *autoencodeur* (Keras).
5. **📌 Conclusion** — résultats, limites, perspectives.

**Principe clé (consigne mentor) :** l'appli **ne réentraîne jamais** de modèle.
Elle **charge** des artefacts déjà produits. Un mode *baseline* la rend fonctionnelle
sans modèle final (basé sur la différence au template moyen des images `train/good`).

---

## 2. Où se trouve tout

| Ressource | Lien |
|-----------|------|
| **Application en ligne** (publique) | https://fidemeca-app-gogr2htbphkjvmz2sx2wu5.streamlit.app |
| **Dépôt GitHub** (code de l'appli) | https://github.com/paul42170/fidemeca-streamlit |
| Hébergement | Streamlit Community Cloud (redéploiement auto à chaque `git push`) |

> ⚠️ L'appli de démo est sur un **repo perso** (paul42170) car le déploiement Streamlit
> exige les droits admin, non disponibles sur le repo d'organisation DataScientest.
> Le code « projet » (feature_engineering, notebooks) reste, lui, sur le repo d'équipe.

---

## 3. Structure des fichiers

```
fidemeca-streamlit/            (= contenu du dossier 05_Streamlit)
├── app.py                 # application principale (les 5 onglets)
├── config.py              # ⚙️ chemins & constantes — SEUL fichier à adapter
├── inference.py           # extraction de features + scoring (réutilise feature_engineering.py)
├── prepare_assets.py      # génère les templates .npy (à lancer une fois, en local)
├── requirements.txt       # dépendances
├── PROTOCOLE_EQUIPE.md     # ce document
├── README.md
├── data/
│   └── mvtec_summary.csv   # résumé du dataset (bundlé → l'onglet Exploration marche sans le dataset)
├── models/
│   └── templates/          # <categorie>.npy — 15 templates (inclus dans le repo, ~750 Ko)
└── .streamlit/config.toml  # thème
```

---

## 4. Lancer l'appli en local (Windows)

Prérequis : **Python 3.13** (ou 3.12). Si `python` ouvre le Microsoft Store, utiliser `py`.

Dans un terminal (cmd), depuis le dossier de l'appli :

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

→ l'appli s'ouvre sur http://localhost:8501.
Pour relancer plus tard : `.venv\Scripts\activate.bat` puis `py -m streamlit run app.py`.

**(Optionnel) activer la démo sur images réelles** — générer les templates une fois
(déjà inclus dans le repo, à refaire seulement si on change de dataset) :

```cmd
py prepare_assets.py "CHEMIN\VERS\LE\DATASET\archive"
```
(le dataset suit l'arborescence MVTec : `<archive>/<categorie>/train/good/*.png`)

---

## 5. Récupérer / mettre à jour le code (GitHub)

**Première fois** (cloner) :
```cmd
git clone https://github.com/paul42170/fidemeca-streamlit.git
cd fidemeca-streamlit
```

**Publier une modif** (déclenche le redéploiement automatique en ligne) :
```cmd
git add .
git commit -m "description de la modif"
git push
```

> Config Git une seule fois si besoin :
> `git config --global user.email "ton.email@exemple.com"`
> `git config --global user.name "Prénom Nom"`

---

## 6. Modèles branchés (mise à jour post-rapport final)

L'appli détecte automatiquement les modèles présents dans `models/` (voir la barre latérale).
Modèles actuellement inclus dans le dépôt (`tensorflow-cpu` activé dans `requirements.txt`) :

- **Baseline** (différence au template, 15 catégories) : toujours actif.
- **Modèle sur features (RF)** : `models/anomaly_model.joblib` + `models/scaler.joblib`, entraîné
  sur les 3 catégories pilotes (bottle/carpet/screw) — avertissement affiché pour les autres.
- **AutoEncodeur Keras** : un fichier par catégorie pilote, `models/autoencoder_<categorie>.keras`
  (bottle/carpet/screw).
- **CNN supervisé 15 catégories** : 3 variantes récupérées du dépôt GitHub d'équipe
  (`DataScientest-Studio/avr26_bmle_ds_anomalies`, branche `main`, dossier `models/`) —
  `cnn_binary_15cat_best.keras` (64×64 Flatten, production), `..._128_best.keras` (128×128),
  `..._gap_best.keras` (ablation GAP). Sélection de la variante dans l'onglet Modélisation.
- **EfficientNetB0 / ResNet50 / PaDiM** : résultats présentés dans le nouvel onglet
  **📊 Comparatif de l'équipe** (tableau + graphique), mais pas de démo live — poids non
  versionnés sur GitHub (trop volumineux, entraînés catégorie par catégorie).

Après tout ajout/màj de modèle : `git push` → l'appli en ligne se met à jour automatiquement.

⚠️ À vérifier dans `app.py` (onglet Modélisation) : la **convention de sortie** du modèle
(`predict` = 1 / -1 pour anomalie selon IsolationForest / OneClassSVM / autre) doit correspondre
au modèle réellement retenu.

---

## 7. Points à adapter par l'équipe

- `config.py` : chemins (`DATASET_ROOT`, chemins des modèles) selon l'environnement (VM / local).
- `inference.py` importe `feature_engineering.py` s'il est présent (mêmes features que le pipeline de Ludovic) ; sinon il utilise une ré-implémentation locale identique.
- Le thème et les textes des onglets Contexte / Conclusion peuvent être enrichis.

---

## 8. Checklist avant la soutenance

- [ ] Ouvrir l'appli **5 min avant** (le plan gratuit met l'appli en veille ; un clic la réveille).
- [ ] Vérifier que la barre latérale affiche « Templates : 15 dispo ».
- [ ] Répéter le parcours : Contexte → Exploration → Preprocessing → **Démo sur une image défectueuse ET une image saine** → Conclusion.
- [ ] Préparer 2-3 images de test (une bonne + une défectueuse par catégorie montrée).
- [ ] Chaque membre parle ; une seule personne partage l'écran.
- [ ] Rappeler que l'appli **ne réentraîne pas** (modèle chargé) — c'est une exigence du mentor.

---

