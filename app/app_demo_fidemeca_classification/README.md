# Fidémeca — App d'identification des anomalies (transfer learning)

App Streamlit **dédiée** à l'identification du *type* de défaut (classification multi-classe),
séparée de l'app de détection de l'équipe pour ne pas partager la RAM (~1 Go sur Streamlit Cloud).
Backbone **EfficientNetB0** (léger, déployable). Même charte visuelle que l'app d'équipe.

L'app ne réentraîne rien : elle charge des artefacts préparés à l'avance.

## Contenu

```
app.py                          # interface (onglets Démo + Performances)
config.py                       # couleurs, catégories, infos équipe
identification.py               # chargement modèle + Grad-CAM
prepare_identification_assets.py# génère les artefacts (à lancer en local)
requirements.txt / runtime.txt / .streamlit/config.toml
data/identification/            # index.json + <cat>_confusion.png   (générés)
models/identification/<cat>/    # model.keras                        (générés, commités)
assets/identification/<cat>/    # images d'exemple                   (générées)
```

## 1. Générer les modèles EfficientNetB0 (une fois)

Dans `exploration_ludovic/`, avec `entrainer_modeles_demo.py`, mets :

```python
CATS = ["bottle", "grid", "leather"]          # tes 3 catégories de démo
BACKBONES = {"efficientnetb0": "demo"}         # EfficientNetB0 uniquement
```

puis `python entrainer_modeles_demo.py`. Ça crée `models_demo/<cat>_efficientnetb0.keras`.

## 2. Préparer les artefacts de l'app

Ajuste `MODELES` en haut de `prepare_identification_assets.py` (les 3 catégories + chemins),
puis, **depuis ce dossier** :

```
python prepare_identification_assets.py
```

→ remplit `models/identification/`, `data/identification/`, `assets/identification/`.

## 3. Tester en local

```
pip install -r requirements.txt
streamlit run app.py
```

## 4. Déployer sur Streamlit Cloud

1. Pousse ce dossier dans un **nouveau repo GitHub** (les `model.keras` doivent être commités).
2. Sur https://share.streamlit.io → New app → ton repo → fichier `app.py`.
3. C'est en ligne. Les 3 catégories sont actives ; les 12 autres affichent
   « catégorie non prise en charge à ce stade ».

## Notes

- EfficientNetB0 ~30 Mo/modèle → tient dans GitHub, pas de découpage.
- Si tu ajoutes une catégorie : entraîne son modèle, ajoute-la à `MODELES`, relance l'étape 2.
- Métrique affichée = **F1 macro du modèle chargé** (EfficientNetB0), cohérente avec ce qui prédit.
