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

## Results (2026-08-05, SLURM 7086-7091, all a10g)

GPU test suite (7089): **468 passed, 0 failed** (full suite, `-m 'not slow'`,
both elements).

MODEL comparisons (rel_sum_diff / pixels failing `allclose(rtol=1e-5,
atol=1e-8*max)` / flux-weighted centroid shift):

| Pair | rel_sum_diff | failing px | centroid shift [px] |
|---|---|---|---|
| main-a vs main-b (noise floor) | 1.1e-9 | 0 / 16.7M | 0.0000 |
| **branch vs main (Q1 primary)** | **4.3e-11** | **0 / 16.7M** | **0.0000** |
| prism-a vs prism-b (Q2 floor) | 1.8e-9 | 0 / 16.7M | 0.0000 |
| main-a vs archive (grism) | 4.6e-5 | 16.4M | (+0.356, +0.161) |
| branch vs archive (grism) | 4.6e-5 | 16.4M | (+0.356, +0.161) |
| prism-a vs archive (prism) | 8.1e-5 | 13.8M | (−0.799, −0.132) |

ISIM: branch and main derive **identical** per-SCA keys
(fold_in reproduced bit-exactly); differing-count pixels out of 16.7M —
main-a/b 132, branch-vs-main 135, prism-a/b 158 — all at the issue-#22 GPU
scatter-add nondeterminism floor. Archive ISIMs differ everywhere, as
predicted (old split-based keys; verified on CPU without a render).

**Verdicts.** Q1: PASS — the branch's grism output is indistinguishable
from unmodified `main` at the measured same-code noise floor, and its
archive delta matches `main`'s to within 3 of 16.4M pixels (inherited
v0.11/v0.12 position fixes, +0.36 px). Q2: PASS as smoke — prism renders
deterministically at the same floor; vs the v0.8-era archive it conserves
flux to 8e-5 with a sub-pixel position shift of the same character and
cause as Q1's archive delta.
