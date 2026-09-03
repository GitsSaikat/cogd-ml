#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=0
for s in scripts/0[1-7]_*.py; do
  echo "== $s"
  python "$s"
done
echo "Outputs in outputs/"
