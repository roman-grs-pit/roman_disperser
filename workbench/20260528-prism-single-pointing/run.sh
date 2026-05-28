#!/usr/bin/env bash
# Prism single-pointing run driver (interactive, on a GPU node).
#
# Usage (from a GPU node):
#   bash run.sh           # all 18 SCAs       -> <output_dir>
#   bash run.sh 5         # smoke: SCA 5 only -> <output_dir>-smoke
#   bash run.sh 5,6,7     # smoke: those SCAs -> <output_dir>-smoke
#
# Grab an a10g first (pin the GRES so you don't land on the pricier l40s):
#   salloc --partition=gpu-med --gres=gpu:a10g:1 --cpus-per-task=4 \
#          --mem=24G --time=4:00:00
#   # ... then on the node:  bash run.sh 5   (smoke) ;  bash run.sh   (full)
#   # scancel the allocation when done -- a10g is ~$1.2/h on-demand.
#
# Steps:
#   1. Stage the catalog from /mnt (S3-Files) to EBS for fast SED reads.
#   2. Disperse the single PRISM pointing through build_grism_image.py.
#
# Smoke runs go to a SEPARATE output dir because the per-pointing
# skip-if-exists is directory-level (build_grism_image.py:1462): a smoke
# run that created <output_dir>/<pointing> would otherwise make the full
# run skip the pointing entirely.
set -euo pipefail

STUDY="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$STUDY/../.." && pwd)"
CONFIG="$STUDY/config.yaml"
ECSV="$STUDY/prism-single.sim.ecsv"
CACHE_DIR=/data/npadman/jax-cache-prism

# Catalog staging: /mnt (authoritative) -> EBS (fast reads). Must match
# catalog_dir in config.yaml.
SRC_CATALOG=/mnt/roman-science/grs/prism-testing-20260527/catalogs
EBS_CATALOG="$STUDY/data/catalogs"

SCAS="${1:-all}"

[ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }
[ -f "$ECSV" ]   || { echo "missing $ECSV"; exit 1; }
mkdir -p "$CACHE_DIR" "$EBS_CATALOG"

echo "=== Staging catalog: $SRC_CATALOG -> $EBS_CATALOG ==="
rsync -a --info=progress2 "$SRC_CATALOG/" "$EBS_CATALOG/"
[ -f "$EBS_CATALOG/metadata.parquet" ] || { echo "stage failed: no metadata.parquet"; exit 1; }
echo "Catalog staged."

if [ "$SCAS" = "all" ]; then
    RUN_CONFIG="$CONFIG"
    echo "=== Full run: all 18 SCAs ==="
else
    # Derive a smoke config: scas -> [list], output_dir -> <output>-smoke.
    RUN_CONFIG="$STUDY/data/config-smoke.yaml"
    pixi run --manifest-path "$REPO/pixi.toml" python - "$CONFIG" "$RUN_CONFIG" "$SCAS" <<'PY'
import sys, yaml
src, dst, scas = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(src))
cfg["scas"] = [int(s) for s in scas.split(",")]
cfg["output_dir"] = cfg["output_dir"].rstrip("/") + "-smoke"
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"  smoke config -> {dst}")
print(f"  scas={cfg['scas']}  output_dir={cfg['output_dir']}")
PY
    echo "=== Smoke run: SCAs $SCAS ==="
fi

echo "=== Dispersing (pixi -e cuda) ==="
pixi run --manifest-path "$REPO/pixi.toml" -e cuda \
    python "$REPO/scripts/build_grism_image.py" \
        --config "$RUN_CONFIG" \
        --pointings "$ECSV" \
        --cache-dir "$CACHE_DIR" \
        --gpu 0

echo "=== Done. See the output_dir printed above for the pointing directory ==="
