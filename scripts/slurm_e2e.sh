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
#   ROMAN_DISPERSER_DATA  data dir from hydrate (exported to both stages; needed unless
#                      the repo's data/ is hydrated)
#   SLURM_ACCOUNT      account                         (pi_np274)
#   GPU_PARTITION, CPU_PARTITION                       (gpu, day)
#   GPU_GRES           GPU request; L40S is fastest    (l40s:1)
#   DISP_ACTIVATE      shell to enter the JAX env      (module load miniconda; conda activate roman)
#   WRAP_ACTIVATE      shell to enter the romanisim env (... conda activate romanisim)
#   ROMANISIM_SRC      romanisim checkout to put first on PYTHONPATH (needs IPC fix #384)
#   WRAP_LEVEL         1 or 2                           (2)
#   WRAP_WORKERS       CPU jobs for stage 2 (array)     (6)
#   WRAP_CPUS          CPUs per stage-2 job             (16)
#   WRAP_THREADS       concurrent SCAs per job (4); the job asks for 12 GB RAM per
#                      concurrent SCA (peak ~9 GB). romanisim (svh/newrd fork) threads each
#                      SCA over WRAP_CPUS/WRAP_THREADS cores (ROMANISIM_NTHREADS).
#                      Measured on a 16-CPU node, 18 SCAs (notes RESULTS_old_vs_new.md):
#                      1x16 670-730 s, 2x8 370-405 s, 4x4 250-325 s, 8x2 250-290 s,
#                      16x1 310-340 s. One SCA cannot keep 16 threads busy, so run >= 4 at
#                      once; 4x4 ties 8x2 at a third of the memory. Each worker gets
#                      ceil(n_SCA / WRAP_WORKERS) SCAs, so concurrency is capped by that too.
#   CRDS_MODE          local when CRDS_PATH is set (complete cache; saves 10-50 s/SCA)
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
NW=${WRAP_WORKERS:-6}; NT=${WRAP_THREADS:-4}; NC=${WRAP_CPUS:-16}; LEVEL=${WRAP_LEVEL:-2}
RT=$(( NC / NT > 0 ? NC / NT : 1 ))
CMODE=${CRDS_MODE:-${CRDS_PATH:+local}}
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
  --array=0-$((NW - 1)) --cpus-per-task=$NC --mem=$((NT * 12))G --time=12:00:00 \
  --job-name=e2e-wrap --output="$L2/e2e-wrap-%A_%a.log" <<EOS
#!/usr/bin/env bash
set -eo pipefail
$WRAP
export PYTHONPATH=$RSRC\${PYTHONPATH:+:\$PYTHONPATH}
export ROMANISIM_NTHREADS=$RT
${CMODE:+export CRDS_MODE=$CMODE}
python -c "import romanisim; print('romanisim', romanisim.__version__, romanisim.__file__)"
python "$HERE/wrap_with_romanisim.py" --input-dir "$OUT" --output-dir "$L2" --level $LEVEL \
  --num-threads $NT --worker-index \$SLURM_ARRAY_TASK_ID --num-workers $NW
EOS
)
echo "stage 1 (disperse, GPU): job $J1 -> $OUT"
echo "stage 2 (romanisim L$LEVEL, CPU array x$NW): job $J2 -> $L2"
