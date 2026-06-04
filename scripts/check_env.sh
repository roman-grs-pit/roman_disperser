#!/usr/bin/env bash
# check_env.sh — Verify that all expected environment variables are set to
# the correct values before running roman_disperser.
#
# Fill in the VALUES below, then run:
#   bash scripts/check_env.sh

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

ok()  { printf "${GREEN}  ✓  %-24s = %s${NC}\n" "$1" "$2"; }
fail() { printf "${RED}  ✗  %-24s = %s (expected %s)${NC}\n" "$1" "${!1:-<unset>}" "$2"; }

# ── Helpers ─────────────────────────────────────────────────────────────
# Check whether a conda environment is active.
conda_active() {
  [[ -n "${CONDA_PREFIX:-}" ]]
}

# Check that a kernel.json for the given env exists and references kernel-helper.
check_conda_kernel() {
  local env_name="${CONDA_DEFAULT_ENV:-}"
  if [[ -z "$env_name" ]]; then
    return 1   # no conda env active — skip silently
  fi
  total=$((total + 1))

  local kernel_json
  kernel_json="${HOME}/.local/share/jupyter/kernels/$(basename "$env_name")/kernel.json"

  if [[ ! -f "$kernel_json" ]]; then
    fail "kernel.json" "<missing>" "$kernel_json"
    return 1
  fi
  # The kernel.json should start with the kernel-helper argv block.
  # Read the first few lines and look for the opening brace + argv pattern.
  if [[ "$(head -c 120 "$kernel_json" | grep "kernel-helper.sh")" == "/global/common/software/m4943/kernel-helper.sh" ]] ; then
    ok "kernel.json" "${env_name} → kernel-helper.sh"
    pass=$((pass + 1))
  else
    fail "kernel.json" "does not source kernel-helper.sh" "$kernel_json"
  fi
}

total=0; pass=0

check() {
  local var="$1" expected="$2"
  total=$((total + 1))
  local actual="${!var:-}"
  if [[ "$actual" == "$expected" ]]; then
    ok "$var" "$actual"
    pass=$((pass + 1))
  else
    fail "$var" "$expected"
  fi
}

echo "Checking roman_disperser environment variables…
"

# ── Core / development ──────────────────────────────────────────────────
# check "PIXI_PROJECT_ROOT" "some_value"

# ── Catalog builder (build_sed_calc_catalog.py) ─────────────────────────
check "github_dir"          "/global/common/software/m4943/grizli0"

# ── CRDS (romanisim environment) ────────────────────────────────────────
check "CRDS_SERVER_URL"     "https://roman-crds.stsci.edu"
check "CRDS_PATH"           "/global/cfs/cdirs/m4943/grismsim/crds_cache"
# check "CRDS_CONTEXT"        "some_value"

# ── STPSF (romanisim environment) ───────────────────────────────────────
check "STPSF_PATH"          "/dvs_ro/cfs/cdirs/m4943/grismsim/stpsf-data"

# ── JAX / GPU ───────────────────────────────────────────────────────────
# check "JAX_PLATFORMS"       "some_value"
# check "JAX_COMPILATION_CACHE_DIR" "some_value"
# check "CUDA_VISIBLE_DEVICES"  "some_value"

# ── stsynphot ───────────────────────────────────────────────────────────
# check "PYSYN_CDBS"          "some_value"

# ── Conda / Jupyter kernel ──────────────────────────────────────────────
if conda_active; then
  check_conda_kernel
else
  printf "  - conda not active — skipping kernel.json check\n"
fi

echo ""
printf "Results: %d/%d checks passed\n" "$pass" "$total"

if [[ "$pass" -eq "$total" ]]; then
  echo "All checks passed! ✅"
  exit 0
else
  echo "Some checks failed. ⚠️"
  exit 1
fi
