#!/usr/bin/env bash
# End-to-end Roman slitless simulation on a Slurm cluster (see docs/e2e_pipeline.md).
#
#   catalog (metadata.parquet + seds.zarr; make_catalog.py builds one from a
#   source table + spectra)
#     -> stage 1, GPU: build_dispersed_image.py -> per-SCA MODEL (e/s) + ISIM FITS
#     -> stage 2, CPU: wrap_with_romanisim.py (romanisim, MODEL injected as a source:
#        sky, QE/flat, Poisson, dark, IPC, CRs, non-linearity, read noise, ramp fit)
#        -> per-SCA L1 or L2 ASDF
#
# Usage:
#   scripts/slurm_e2e.sh CONFIG.yaml POINTINGS.ecsv [L2_OUT_DIR]
#
# Environment (defaults are Bouchet's):
#   SLURM_ACCOUNT      account                         (pi_np274)
#   GPU_PARTITION, CPU_PARTITION                       (gpu, day)
#   GPU_GRES           GPU request; L40S is fastest    (l40s:1)
#   DISP_ACTIVATE      shell to enter the JAX env      (module load miniconda; conda activate roman)
#   WRAP_ACTIVATE      shell to enter the romanisim env (... conda activate romanisim)
#   ROMANISIM_SRC      romanisim checkout to put first on PYTHONPATH (needs IPC fix #384)
#   WRAP_LEVEL         1 or 2                           (2)
#   WRAP_WORKERS       CPU jobs for stage 2 (array)     (4)
#   WRAP_THREADS       concurrent SCAs per worker; ~8 GB RAM each (4)
set -euo pipefail
CFG=$(readlink -f "$1"); PTS=$(readlink -f "$2")
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG" 2>/dev/null \
      || grep -E '^output_dir:' "$CFG" | awk '{print $2}')
L2=${3:-${OUT%/}_l${WRAP_LEVEL:-2}}
ACC=${SLURM_ACCOUNT:-pi_np274}
DISP=${DISP_ACTIVATE:-"module load miniconda; conda activate roman"}
WRAP=${WRAP_ACTIVATE:-"module load miniconda; conda activate romanisim"}
RSRC=${ROMANISIM_SRC:-$HOME/Roman/romanisim}
NW=${WRAP_WORKERS:-4}; NT=${WRAP_THREADS:-4}; LEVEL=${WRAP_LEVEL:-2}
mkdir -p "$OUT" "$L2"

J1=$(sbatch --parsable --account=$ACC --partition=${GPU_PARTITION:-gpu} --gpus=${GPU_GRES:-l40s:1} --nodes=1 \
  --cpus-per-task=8 --mem=64G --time=12:00:00 --job-name=e2e-disperse \
  --output="$OUT/e2e-disperse-%j.log" <<EOS
#!/usr/bin/env bash
set -eo pipefail
$DISP
export PYTHONPATH=$(dirname "$HERE")/src\${PYTHONPATH:+:\$PYTHONPATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name --format=csv,noheader
python "$HERE/build_dispersed_image.py" --config "$CFG" --pointings "$PTS"
EOS
)
J2=$(sbatch --parsable --account=$ACC --partition=${CPU_PARTITION:-day} --dependency=afterok:$J1 \
  --array=0-$((NW - 1)) --cpus-per-task=$NT --mem=$((NT * 9))G --time=12:00:00 \
  --job-name=e2e-wrap --output="$L2/e2e-wrap-%A_%a.log" <<EOS
#!/usr/bin/env bash
set -eo pipefail
$WRAP
export PYTHONPATH=$RSRC\${PYTHONPATH:+:\$PYTHONPATH}
python -c "import romanisim; print('romanisim', romanisim.__version__, romanisim.__file__)"
python "$HERE/wrap_with_romanisim.py" --input-dir "$OUT" --output-dir "$L2" --level $LEVEL \
  --num-threads $NT --worker-index \$SLURM_ARRAY_TASK_ID --num-workers $NW
EOS
)
echo "stage 1 (disperse, GPU): job $J1 -> $OUT"
echo "stage 2 (romanisim L$LEVEL, CPU array x$NW): job $J2 -> $L2"
