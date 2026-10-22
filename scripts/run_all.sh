#!/usr/bin/env bash
# Single-pair baselines vs. mixed training for Gamo, Gofa and Dawuro -> English.
# Usage: bash scripts/run_all.sh <data_dir> [config]
set -euo pipefail
D=${1:?"data directory required"}
CFG=${2:-configs/transformer.yaml}
LANGS=(gamo gofa dawuro)

for L in "${LANGS[@]}"; do
  echo ">> Baseline: $L -> en"
  python -m ethiomt.train --data-dir "$D" --langs "$L" --config "$CFG" --out "runs/${L}_single"
  python -m ethiomt.evaluate --run "runs/${L}_single" --data-dir "$D" --langs "$L"
done

echo ">> Mixed training: ${LANGS[*]} -> en"
python -m ethiomt.train --data-dir "$D" --langs "${LANGS[@]}" --tags --config "$CFG" --out runs/mixed
python -m ethiomt.evaluate --run runs/mixed --data-dir "$D" --langs "${LANGS[@]}"
