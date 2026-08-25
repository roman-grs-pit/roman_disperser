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

`check_perf.py` applies two gates (details and rationale in its docstring):

1. **Ratio gate** (hardware-insensitive, always on): baseline/noscatter ≤ 3.
   Healthy jax measures ~0.73–0.75 on GPU since the native-deposit port
   (the fused deposit kernel beats noscatter's separate reduction kernel;
   pre-native16 it was ~1.1–1.2). The 0.11.0 regression measured 15–21 on
   the old deposit and would still blow far past 3 on the new one, since a
   scatter regression inflates only the baseline.
2. **Absolute gate** (when `baselines/gpu.json` has the GPU): baseline
   ms/gal ≤ 1.5× the recorded reference. Catches uniform slowdowns that the
   ratio cannot see.

## Golden jax versions

We support `jax>=0.7,!=0.11.0` but cannot perf-test every release, so
`run_golden.sh` sweeps the small fixed set in `golden-jax-versions.txt`
(currently: the pixi.lock version 0.7.2, the post-exclusion pip-era 0.11.1,
and PyPI latest) — one throwaway venv per version, each with its own JIT
compilation cache. The policy for evolving the set is documented in that
file.

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

CPU runs work and are reported, but both gates were calibrated on GPU;
treat CPU results as informational.
