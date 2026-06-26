#!/usr/bin/env bash
# =====================================================================
# Réinstallation rapide de l'environnement projet sur la VM Liora.
# À relancer après chaque redémarrage de la VM (qui efface le dossier home).
#
# Usage :
#   bash setup_vm.sh
# =====================================================================
set -e

REPO_URL="https://github.com/DataScientest-Studio/avr26_bmle_ds_anomalies.git"
REPO_DIR="$HOME/avr26_bmle_ds_anomalies"
BRANCH="graphs-python"

echo "==> 1/4  Paquets système (venv + pip)"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12-venv python3-pip

echo "==> 2/4  Configuration Git"
git config --global user.name  "Paul Fournel"
git config --global user.email "paulfournel.71@gmail.com"

echo "==> 3/4  Clone du dépôt (si absent)"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"

echo "==> 4/4  Environnement Python"
mkdir -p data notebooks src figures
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet pandas numpy matplotlib seaborn pillow jupyter
pip freeze > requirements.txt

echo ""
echo "============================================================"
echo " Environnement prêt dans : $REPO_DIR"
echo " Branche                 : $BRANCH"
echo ""
echo " Activer l'environnement :  source $REPO_DIR/.venv/bin/activate"
echo " Lancer le notebook      :  notebooks/01_exploration_mvtec.ipynb -> Run All"
echo "============================================================"
