#!/usr/bin/env bash
# =====================================================================
# Télécharge le dataset MVTec AD (images) sur la VM depuis Google Drive,
# puis le décompresse dans data/images/.
#
# À lancer au début d'une session quand on a besoin des VRAIES images
# (ex. phase deep learning). Inutile pour les graphs d'exploration,
# qui tournent à partir de data/mvtec_manifest.csv.
#
# Usage :   bash download_data.sh
# =====================================================================
set -e

FILE_ID="1QgbWCsCjIpuaiSY-1WEHznX652PdjkeR"   # archive.zip sur le Drive partagé
REPO_DIR="$HOME/avr26_bmle_ds_anomalies"
DATA_DIR="$REPO_DIR/data/images"
ZIP="$REPO_DIR/data/archive.zip"
NB_ATTENDU=5354

cd "$REPO_DIR"
mkdir -p "$DATA_DIR"

# 1) Si le dataset est déjà là, on ne refait rien
COUNT=$(find "$DATA_DIR" -type f 2>/dev/null | wc -l)
if [ "$COUNT" -ge "$NB_ATTENDU" ]; then
    echo "Dataset déjà présent ($COUNT fichiers). Rien à faire."
    exit 0
fi

# 2) Outils nécessaires
if [ -d "$REPO_DIR/.venv" ]; then
    source "$REPO_DIR/.venv/bin/activate"
fi
command -v unzip >/dev/null 2>&1 || { echo "Installation d'unzip..."; sudo apt-get install -y -qq unzip; }
pip install --quiet gdown

# 3) Téléchargement depuis Google Drive
echo "==> Téléchargement du dataset depuis Google Drive (~5 Go)..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$ZIP"

# 4) Décompression
echo "==> Décompression..."
TMP=$(mktemp -d)
unzip -q "$ZIP" -d "$TMP"

# Le zip contient un dossier 'archive' avec les 15 catégories ;
# on détecte automatiquement le dossier parent de 'bottle'.
SRC=$(dirname "$(find "$TMP" -type d -name bottle | head -1)")
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
    echo "ERREUR : impossible de localiser les catégories dans l'archive." >&2
    exit 1
fi
echo "Source détectée : $SRC"

cp -r "$SRC"/* "$DATA_DIR"/

# 5) Nettoyage + vérification
rm -rf "$TMP" "$ZIP"
FINAL=$(find "$DATA_DIR" -type f | wc -l)
echo ""
echo "============================================================"
echo " Dataset installé : $FINAL fichiers dans $DATA_DIR"
[ "$FINAL" -ge "$NB_ATTENDU" ] && echo " OK (>= $NB_ATTENDU images attendues)" || echo " ⚠️ Moins d'images que prévu, à vérifier."
echo "============================================================"
