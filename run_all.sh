#!/usr/bin/env bash
# Full pipeline: download, preprocess, QC, train, explain, integrate, therapy lookup, report pages.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/01_download.py
python scripts/02_preprocess.py
python scripts/00_qc.py
python scripts/03_train.py
python scripts/04_explain.py
python scripts/05_integrate.py
python scripts/06_therapy.py
python scripts/07_networks.py
python scripts/make_report.py
for nb in notebooks/0*.ipynb; do
  python -m nbconvert --to notebook --execute --inplace "$nb"
done
python -m nbconvert --to html --embed-images --output-dir docs notebooks/0*.ipynb
