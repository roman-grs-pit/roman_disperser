# Prism-merge validation gates (feature/prism-merge-fable)

Validation for the prism merge: two questions, one SCA (SCA 5) each, per the
merge specification. All renders are 1-SCA, 1-pointing batch runs on
`gpu-med` (a10g pinned), ~5.5 min GPU each once JIT-warm.

The one-row ECSVs here keep their **original filename stems**
(`acceptance-testing-spectro.sim.ecsv`, `prism-single.sim.ecsv`) — the stem
salts the RNG key (`make_pointing_key`), so renaming would void any ISIM
comparison.

## Q1 — grism unchanged

Re-render SCA 5 of acceptance pointing `001.001.001.001.004.002`
(RA 8.722756, Dec 0.599108, PA 60, seed 42, exptime 190.22 s).

- `grism-gate/config-main-a.yaml`, `config-main-b.yaml` — the **baseline**:
  unmodified `main` (7998c22), rendered twice. The a/b pair measures the GPU
  run-to-run noise floor (issue #22: scatter-add nondeterminism at f32
  epsilon); a-vs-archive measures how far current `main` already is from the
  archived product.
- `grism-gate/config-branch.yaml` — the branch render (`element: grism`).

**Primary gate: branch ≡ main** (MODEL and ISIM), at the a/b noise floor.
The archive comparison is secondary: the archive was rendered at
`v0.7.0-12-g5258db2`, which **predates** v0.11 (sky→FPA float32 precision),
v0.12 (flat-sky → gnomonic projection) and v0.13 (per-SCA RNG keys), so:

- archive MODEL is expected to differ from any current render at the
  position-fix level (sub-pixel source shifts) — branch-vs-archive must show
  the *same* delta as main-vs-archive;
- archive ISIM is unreachable **by construction**: its header key
  `RNDSEED=(4084139779, 1312707973)` reproduces exactly under the old
  `jax.random.split(key, 18)[4]` scheme and not under v0.13's
  `fold_in(key, 5)` — verified on CPU without a render (session 13a96caa).

## Q2 — prism smoke

Re-render SCA 5 of the archived prism single pointing
(RA 10, Dec 0, PA 0, seed 42, exptime 452.42 s), with the /mnt 7500 Å
catalog (catalog-v2 fails the band-coverage check, by design).

- `prism-gate/config-branch-a.yaml`, `config-branch-b.yaml` — the branch,
  twice (self-consistency floor, and prism ISIM determinism at fixed keys).

Archive: rendered from the **prism branch** at `3ec36ab` (v0.8-era), i.e.
flat-sky projection and old RNG scheme. So vs the archive we gate on total
flux and on the MODEL delta having the same character/magnitude as Q1's
main-vs-archive delta (same cause: v0.11+v0.12 position fixes); the archived
ISIM is unreachable for the same reason as Q1's.

## Running

```bash
LOGDIR=/data/npadman/tmp/slurm-logs/prism-merge-fable   # created by the job
V=$PWD/workbench/20260805-prism-merge-validation
sbatch --export=ALL,CHECKOUT=/data/npadman/1-Projects/roman/roman_disperser/main,CONFIG=$V/grism-gate/config-main-a.yaml,ECSV=$V/grism-gate/acceptance-testing-spectro.sim.ecsv $V/submit_gate.sh
# ... same pattern for the other four configs; run a before b so the second
# render of a pair reuses the JIT cache.
```

Outputs land in `../../..//prism_merge_fable_outputs/` (outside the repo).
Compare with:

```bash
pixi run python $V/compare_gate.py <A.fits> <B.fits> --label "branch vs main"
```

`compare_gate.py` prints rel_sum_diff, max diffs, allclose failures,
flux-weighted centroid shift, and ISIM count-flip statistics; thresholds and
verdicts live in the writeup, not the tool.
