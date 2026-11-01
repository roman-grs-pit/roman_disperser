# Performance regression benchmarks

Catches backend performance regressions in the dispersal hot path — the kind
that produce correct images slowly, which no correctness test sees. The
motivating incident: jax 0.11.0's GPU scatter-add regression
([jax-ml/jax#39959](https://github.com/jax-ml/jax/issues/39959)) made the
deposit step ~16x slower (157/177/125 ms/galaxy by grism order on an A10G vs
the usual ~9 ms flat) in a user's pip environment, turning a ~2 h pointing
into ~47 h. That release is now excluded (`jax>=0.7,!=0.11.0`), and this
suite exists so the next one is caught before a user finds it.

## What is measured

`bench_deposit.py` times the **production** galaxy disperser
(`galaxy_disperser.make_galaxy_disperser`, the same code path as
`scripts/build_dispersed_image.py`) on a seeded synthetic workload at
production geometry (30 px native Sérsic stamps, 184² 4×-oversampled PSFs,
full grism band at 2 Å = 5501 wavelength samples, batch=100 fori_loop), per
grism order, reporting **ms per galaxy** (wall time / n_gal, after a
compile+warmup call). Alongside it, a **noscatter** reference — the identical
computation with the deposit scatter replaced by a scalar reduction — gives
the compute floor.

Next to that gated pair it times both source types at the **production
defaults** of the driver (`default`: interp deposit, count rebinning 40 / 4 /
2 Å in orders 0 / 1 / 2, star PSF hoist) and at the historical setting for
reference (`floor`: floor deposit at 2 Å, no hoist): stars through
`pipeline.run_star_batch` (the production 184² 4× PSF, 47² native stamps),
galaxies through `run_galaxy_batch`, reported as ms per source per order and
printed as a table at the end. `--kinds gate,star,galaxy` and
`--settings default,floor` select what runs. The script is identical on CPU
and GPU, so a laptop result and a GPU-node result of the same commit compare
directly with `compare.py`.

`check_perf.py` applies two gates (details and rationale in its docstring):

1. **Ratio gate** (hardware-insensitive, always on): baseline/noscatter ≤ 1.5
   (3 until v0.16.0, which let conda-forge jax 0.10.2's 1.9–2.8 through).
   Healthy jax measures ~0.73–0.77 on GPU since the native-deposit port
   (the fused deposit kernel beats noscatter's separate reduction kernel;
   pre-native16 it was ~1.1–1.2). The 0.11.0 regression measured 15–21 on
   the old deposit and would still blow far past 1.5 on the new one, since a
   scatter regression inflates only the baseline.
2. **Absolute gate** (when the baselines file has the device): baseline
   ms/gal ≤ 1.5× the recorded reference, and the same for every star /
   galaxy `default` / `floor` row the baseline lists (`ms_per_src`).
   Catches uniform slowdowns that the ratio cannot see. GPU results are
   keyed by the `nvidia-smi` name in `baselines/gpu.json`, CPU results by the
   CPU model string in `baselines/cpu.json`; the file is picked by the
   result's backend unless `--baselines` is given.

## Golden jax versions

We support `jax>=0.11.2,<0.12` but cannot perf-test every release, so
`run_golden.sh` sweeps the small fixed set in `golden-jax-versions.txt`
(currently: the 0.11.2 floor, and `latest`, which resolves to the newest
release the pyproject cap allows) — one throwaway venv per version, each
with its own JIT compilation cache. These are pip (PyPI-wheel) installs; the
conda-forge build in the pixi envs is benched by the one-off command below,
so a pre-tag check runs both. The policy for evolving the set is documented
in that file.

## Running

Needs a GPU node and hydrated reference data (`pixi run hydrate`, or
`$ROMAN_DISPERSER_DATA` pointing at a shared cache).

```bash
# one-off, current environment (~2-3 min after compile):
pixi run -e cuda python benchmarks/bench_deposit.py --out results/mine.json
python benchmarks/check_perf.py results/mine.json

# full golden sweep (~15-30 min, dominated by pip installs):
bash benchmarks/run_golden.sh
```

### CPU vs GPU comparison (stars and galaxies, production defaults)

The PSF caches must exist for the SCA used (`--sca`, default 1; the laptop
baseline used SCA 5, the only cache there). Run the same command on both
machines from the same commit, then compare:

```bash
# laptop / CPU (the baselines/cpu.json numbers; ~15 min on an i7-9750H):
python benchmarks/bench_deposit.py --sca 5 --n-gal 20 --batch 20 \
    --n-star 100 --star-batch 100 --repeats 2 --out results/cpu.json

# GPU node (same flags, so the rows match; add more sources if you like):
sbatch -p <gpu-partition> --gres=gpu:1 -t 0:30:00 --wrap \
    'pixi run -e cuda python benchmarks/bench_deposit.py --sca 5 \
     --n-gal 20 --batch 20 --n-star 100 --star-batch 100 --repeats 2 \
     --out results/gpu.json'

python benchmarks/check_perf.py results/gpu.json     # gates vs gpu.json
python benchmarks/compare.py results/cpu.json results/gpu.json
```

`compare.py` prints ms per source per (kind, setting, order) for each file
and the ratio to the first. To start a GPU baseline for the star / galaxy
rows, add an `ms_per_src` block (same layout as in `baselines/cpu.json`)
under the GPU's entry in `baselines/gpu.json` from healthy runs; the
existing `ms_per_gal` / `noscatter_ms_per_gal` fields keep their meaning.

On a SLURM cluster, wrap the sweep:

```bash
sbatch -p <gpu-partition> --gres=gpu:1 -t 1:00:00 \
    --wrap 'bash benchmarks/run_golden.sh'
```

**When to run:** before tagging a release, and after any change to the
deposit/dispersal path or a jax floor bump. It is deliberately not in CI —
it needs a GPU and costs real money on cloud nodes.

## Updating baselines

`baselines/gpu.json` maps `nvidia-smi` GPU names to reference ms/gal (see the
`provenance` field for how the current numbers were measured). Add a new GPU
by running the bench on healthy versions (at least two golden versions should
agree) and recording the steady-state numbers; never update a baseline to
absorb a slowdown without understanding it first.

`baselines/cpu.json` holds the laptop CPU numbers (one machine, see its
`provenance`). The absolute gate applies to CPU results of that CPU model;
the ratio gate was calibrated on GPU (CPU measures a different healthy
ratio, see the provenance), so pass `--max-ratio 2.5` when gating
CPU.
