# -*- coding: utf-8 -*-
# Constantes de l'app (memes conventions que l'app de Paul, pour la coherence visuelle).

COLOR_PRIMARY = "#1F3864"      # navy — identique au theme de l'app d'equipe
COLOR_ACCENT  = "#E07B00"

# les 15 categories MVTec (toujours proposees dans le menu)
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

EQUIPE   = "Paul Fournel · Alex Mac-Kame · Ludovic Marquant · Fabrice Masola"
ENTREPRISE = "Fidemeca — mecanique de precision"
DATASET  = "MVTec AD (15 catégories, 5 354 images)"

IMG_SIZE = (224, 224)

# F1 macro EfficientNetB0 par categorie (benchmark 15 cat) — pour l'onglet Performances.
# Sert de reference ; les categories reellement chargees affichent en plus leur F1 recalcule.
F1_EFFICIENTNET = {
    "leather": 0.886, "bottle": 0.850, "wood": 0.813, "hazelnut": 0.786, "tile": 0.782,
    "toothbrush": 0.726, "metal_nut": 0.661, "carpet": 0.650, "screw": 0.629,
    "transistor": 0.545, "zipper": 0.471, "pill": 0.385, "cable": 0.365,
    "capsule": 0.362, "grid": 0.070,
}
