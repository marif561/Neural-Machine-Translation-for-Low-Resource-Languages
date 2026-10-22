#!/usr/bin/env bash
# Reproduce the paper's four experiments (Table 4: X -> English).
#
# Usage: bash scripts/run_experiments.sh <main_data_dir> <zeroshot_data_dir> [config]
#   main_data_dir      wolaita/{train,dev}  gamo|gofa|dawuro/{train,dev,test}   (Exp 1, 2, 4)
#   zeroshot_data_dir  wolaita/ (same as above) + gamo|gofa|dawuro/{dev,test}   (Exp 3)
# Add REVERSE=1 to the environment for English -> X (Table 5).
set -euo pipefail
MAIN=${1:?"main data directory required"}
ZS=${2:?"zero-shot data directory required"}
CFG=${3:-configs/transformer.yaml}
LOW=(gamo gofa dawuro)
REV=${REVERSE:+--reverse}
SFX=${REVERSE:+_en2x}

others() { for l in "${LOW[@]}"; do [[ $l != "$1" ]] && printf '%s ' "$l"; done; }

echo "== Exp 1: transfer learning (Wolaita parent -> fine-tune) =="
python -m ethiomt.train --data-dir "$MAIN" --train-langs wolaita $REV --config "$CFG" --out "runs/exp1_parent$SFX"
for L in "${LOW[@]}"; do
  python -m ethiomt.train --data-dir "$MAIN" --train-langs "$L" --init-from "runs/exp1_parent$SFX" $REV \
    --config "$CFG" --out "runs/exp1_$L$SFX"
  python -m ethiomt.evaluate --run "runs/exp1_$L$SFX" --data-dir "$MAIN"
done

echo "== Exp 2: Wolaita + one low-resource language =="
for L in "${LOW[@]}"; do
  python -m ethiomt.train --data-dir "$MAIN" --support-langs wolaita --train-langs "$L" $REV \
    --config "$CFG" --out "runs/exp2_$L$SFX"
  python -m ethiomt.evaluate --run "runs/exp2_$L$SFX" --data-dir "$MAIN"
done

echo "== Exp 3: Wolaita only, tested on unseen languages =="
python -m ethiomt.train --data-dir "$ZS" --support-langs wolaita --eval-langs "${LOW[@]}" $REV \
  --config "$CFG" --out "runs/exp3$SFX"
python -m ethiomt.evaluate --run "runs/exp3$SFX" --data-dir "$ZS"

echo "== Exp 4: Wolaita + all three low-resource languages =="
for L in "${LOW[@]}"; do
  python -m ethiomt.train --data-dir "$MAIN" --support-langs wolaita $(others "$L") --train-langs "$L" $REV \
    --config "$CFG" --out "runs/exp4_$L$SFX"
  python -m ethiomt.evaluate --run "runs/exp4_$L$SFX" --data-dir "$MAIN"
done
