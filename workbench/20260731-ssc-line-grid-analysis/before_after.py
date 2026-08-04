#!/usr/bin/env python
"""Before/after analysis of the 2026-07-31 v0.13.0 line-grid rerun.

Compares the 20260731 rerun of the SSC line-grid waves (1b: stars/lines-only,
2: +continuum, 3: Sersic galaxies) against the 20260724 originals. The two
campaigns use bit-identical catalogs; the only change is the disperser code
(v0.10.x -> v0.13.0: TF32/float32 sky->FPA precision fixes, flat-sky ->
gnomonic TAN projection, per-SCA RNG keys). Three questions:

1. **Blast radius** — did the reported science metrics (per-SCA median d_disp,
   the line-centering residual along dispersion, in native pixels) move?
2. **Placement shift** — how far did undispersed source positions (xsca, ysca,
   1-indexed FITS px) move, and where on the focal plane?
3. **Smoothness gate** — is the independent-validation audit's defect (the
   (RA, Dec) -> (xsca, ysca) map non-smooth at 0.6-2.9 px rms) fixed? Metric:
   rms of the residual after a per-SCA degree-3 bivariate polynomial fit of
   tangent-plane offsets -> pixel position. A physical (distortion-only) map
   fits to ~1e-3 px; the audit-era defect leaves ~1 px.

Outputs (figures, summary.json, per-SCA tables) go to OUT (the S3 store's
analysis/ dir), NOT the repo — deliverables live with the data by Nikhil's
2026-07-31 decision. Run from the line_test worktree:

    pixi run python workbench/20260731-ssc-line-grid-analysis/before_after.py

Provenance: reads codever/gitsha back from the truth tables' embedded parquet
metadata (roman_disperser_provenance) and records both sides in summary.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GRS = Path("/mnt/roman-science/grs")
OLD, NEW = "line-tests-20260724", "line-tests-20260731"
SUFFIX = {"1b": "", "2": "-cont", "3": "-gal"}
VARIANTS = ["stpsf", "gauss"]
PAS = ["pa000", "pa010"]
OUT = GRS / NEW / "analysis"
OUT.mkdir(exist_ok=True)

# Old/new pair colors (CVD-safe: gray vs blue), sequential = viridis.
C_OLD, C_NEW = "#8a8a8a", "#2563b0"


def pdir(campaign, suffix, variant, pa):
    return (GRS / f"{campaign}{suffix}" / f"lines_{variant}"
            / f"pointing_{pa}_001.001.001.001.001.001")


def med_ddisp(campaign, suffix, variant, pa):
    """Per-SCA median d_disp [native px] from the closure residuals.

    `d_disp` is written by check_line_centering as the projection of
    (measured - predicted) onto the local dispersion tangent, with the detector
    axis chosen per line from |u_disp| — it is NOT `x_meas - x_pred_jax`. Do not
    reconstruct it here if the column is ever missing: that would silently
    compare a different statistic on one side of the before/after. Fix the
    closure run instead.
    """
    r = pd.read_parquet(pdir(campaign, suffix, variant, pa) / "residuals.parquet")
    return r.groupby("sca")["d_disp"].median()


def truth(campaign, suffix, variant, pa, cols):
    return pd.read_parquet(pdir(campaign, suffix, variant, pa) / "truth_lines.parquet",
                           columns=cols)


def embedded_provenance(campaign, suffix, variant, pa):
    md = pq.read_schema(pdir(campaign, suffix, variant, pa) / "truth_lines.parquet").metadata
    raw = (md or {}).get(b"roman_disperser_provenance")
    return json.loads(raw) if raw else None


def smoothness_rms(campaign, suffix, variant, pa):
    """Per-SCA rms [px] of (RA,Dec)->(xsca,ysca) after a degree-3 poly fit.

    Per SCA: center sky coordinates on the per-SCA mean,
    u = (ra - mean(ra)) cos(mean(dec)), v = dec - mean(dec) [deg] — a flat
    local centering, not a formal projection (at 0.2 deg SCA scale the
    difference is absorbed by the polynomial). Then fit x and y
    independently by ordinary least squares to all monomials u^i v^j with
    total degree i + j <= 3 (10 coefficients per axis):

        xhat = sum_{i+j<=3} a_ij u^i v^j ;  yhat likewise.

    Statistic: true rms of the 2-D residual, sqrt(<rx^2 + ry^2>). Degree 3
    comfortably absorbs real optical distortion at SCA scale while leaving
    the audit-era non-smoothness (~1 px) in the residual.
    """
    t = truth(campaign, suffix, variant, pa,
              ["catalog_index", "sca", "ra", "dec", "xsca", "ysca"])
    t = t.drop_duplicates(["catalog_index", "sca"])
    rows = []
    for sca, g in t.groupby("sca"):
        if len(g) < 40:
            continue
        u = (g.ra - g.ra.mean()) * np.cos(np.deg2rad(g.dec.mean()))
        v = g.dec - g.dec.mean()
        A = np.column_stack([u**i * v**j for i in range(4) for j in range(4 - i)])
        rx = g.xsca - A @ np.linalg.lstsq(A, g.xsca, rcond=None)[0]
        ry = g.ysca - A @ np.linalg.lstsq(A, g.ysca, rcond=None)[0]
        rows.append((sca, float(np.sqrt(np.mean(rx**2 + ry**2))), len(g)))
    return pd.DataFrame(rows, columns=["sca", "rms_px", "n"]).set_index("sca")


# ---------------------------------------------------------------------------
# 1+2. Placement shifts and science stability (wave 1b, all variants/PAs)
# ---------------------------------------------------------------------------
summary = {"old": OLD, "new": NEW,
           "provenance_new": embedded_provenance(NEW, "", "stpsf", "pa000")}

shift_tabs, sci_tabs = {}, {}
for v in VARIANTS:
    for pa in PAS:
        to = truth(OLD, "", v, pa, ["catalog_index", "sca", "order", "line_id",
                                    "ra", "dec", "xsca", "ysca"])
        tn = truth(NEW, "", v, pa, ["catalog_index", "sca", "order", "line_id",
                                    "xsca", "ysca"])
        m = to.merge(tn, on=["catalog_index", "sca", "order", "line_id"],
                     suffixes=("_o", "_n"))
        m = m.drop_duplicates(["catalog_index", "sca"])  # shifts are per (source, SCA)
        m["dx"] = m.xsca_n - m.xsca_o
        m["dy"] = m.ysca_n - m.ysca_o
        m["dr"] = np.hypot(m.dx, m.dy)
        shift_tabs[(v, pa)] = m
        mo, mn = med_ddisp(OLD, "", v, pa), med_ddisp(NEW, "", v, pa)
        sci_tabs[(v, pa)] = pd.DataFrame({"old": mo, "new": mn})

m0 = shift_tabs[("stpsf", "pa000")]
summary["placement_shift_px"] = {
    "median": float(m0.dr.median()), "p95": float(m0.dr.quantile(.95)),
    "max": float(m0.dr.max()),
    "per_sca_median_min": float(m0.groupby("sca").dr.median().min()),
    "per_sca_median_max": float(m0.groupby("sca").dr.median().max()),
}
summary["science_blast_radius_px"] = {
    f"{v}_{pa}": float((t.new - t.old).abs().max())
    for (v, pa), t in sci_tabs.items()
}

# ---------------------------------------------------------------------------
# 3. Smoothness gate (wave 1b stpsf pa000 old vs new)
# ---------------------------------------------------------------------------
sm_old = smoothness_rms(OLD, "", "stpsf", "pa000")
sm_new = smoothness_rms(NEW, "", "stpsf", "pa000")
summary["smoothness_rms_px"] = {
    "old_min": float(sm_old.rms_px.min()), "old_max": float(sm_old.rms_px.max()),
    "new_min": float(sm_new.rms_px.min()), "new_max": float(sm_new.rms_px.max()),
}

# ---------------------------------------------------------------------------
# Wave 2 null + wave 3 transfer, replicated on the new stores
# ---------------------------------------------------------------------------
w2, w3 = {}, {}
for v in VARIANTS:
    for pa in PAS:
        w2[f"{v}_{pa}"] = float((med_ddisp(NEW, "-cont", v, pa)
                                 - med_ddisp(NEW, "", v, pa)).abs().max())
        a, b = med_ddisp(NEW, "", v, pa), med_ddisp(NEW, "-gal", v, pa)
        j = pd.concat([a, b], axis=1, keys=["s", "g"]).dropna()
        w3[f"{v}_{pa}"] = {"r": float(np.corrcoef(j.s, j.g)[0, 1]),
                           "max_abs_delta": float((j.g - j.s).abs().max())}
summary["wave2_null_max_delta_px"] = w2
summary["wave3_transfer"] = w3

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})

# Fig 1 — focal-plane map of placement shifts. Positions shown in sky coords
# (RA scaled by cos Dec, so distances are honest); arrows are the pixel-space
# shift direction, color = |shift| in px (sequential viridis).
fig, ax = plt.subplots(figsize=(7.5, 7))
s = m0.sample(min(4000, len(m0)), random_state=0)
q = ax.quiver((s.ra - s.ra.mean()) * np.cos(np.deg2rad(s.dec.mean())), s.dec,
              s.dx, s.dy, s.dr, cmap="viridis", scale=120, width=0.0018)
fig.colorbar(q, ax=ax, label="|position shift|  [native px]")
ax.set_xlabel(r"$(\mathrm{RA}-\overline{\mathrm{RA}})\cos\delta$  [deg]")
ax.set_ylabel("Dec  [deg]")
ax.set_title(f"Undispersed source-position shift, {OLD} → {NEW}\n"
             "wave 1b stpsf pa000 · arrows: pixel-space (Δx, Δy)")
ax.set_aspect("equal")
# Astronomical orientation: East (increasing RA) to the LEFT, North up.
# Arrow components are pixel-space (Δx, Δy), so their on-screen direction is
# schematic either way; the color (|shift|) carries the magnitude.
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(OUT / "fig1_shift_map.png")

# Fig 2 — CDF of |shift| overall + per-SCA medians.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
xs = np.sort(m0.dr.to_numpy())
axes[0].plot(xs, np.linspace(0, 1, len(xs)), color=C_NEW, lw=2)
axes[0].axvline(m0.dr.median(), color=C_OLD, ls="--", lw=1)
axes[0].annotate(f"median {m0.dr.median():.2f} px",
                 (m0.dr.median(), 0.5), textcoords="offset points",
                 xytext=(6, 0))
axes[0].set_xlabel("|position shift|  [native px]")
axes[0].set_ylabel("CDF")
per = m0.groupby("sca").dr.median()
axes[1].bar(per.index, per.values, color=C_NEW)
axes[1].set_xlabel("SCA")
axes[1].set_ylabel("median |shift|  [native px]")
axes[1].set_xticks(range(1, 19))
fig.suptitle("Placement shift, wave 1b stpsf pa000", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig2_shift_cdf.png", bbox_inches="tight")

# Fig 3 — science stability: per-SCA median d_disp old vs new.
fig, ax = plt.subplots(figsize=(4.6, 4.4))
for (v, pa), t in sci_tabs.items():
    ax.plot(t.old, t.new, "o", ms=5, alpha=0.8,
            label=f"{v} {pa} (max |Δ| {(t.new - t.old).abs().max():.4f})")
lim = max(abs(np.concatenate([t.to_numpy().ravel()
                              for t in sci_tabs.values()]))) * 1.15
ax.plot([-lim, lim], [-lim, lim], color=C_OLD, lw=1, zorder=0)
ax.set_xlabel("per-SCA median $d_\\mathrm{disp}$, 20260724  [px]")
ax.set_ylabel("per-SCA median $d_\\mathrm{disp}$, 20260731  [px]")
ax.set_title("Science metric before vs after (wave 1b)")
ax.legend(fontsize=7)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "fig3_science_stability.png")

# Fig 4 — smoothness gate: per-SCA poly3 residual rms, log scale.
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.semilogy(sm_old.index, sm_old.rms_px, "o-", color=C_OLD, label="20260724 (pre-fix)")
ax.semilogy(sm_new.index, sm_new.rms_px, "o-", color=C_NEW, label="20260731 (v0.13.0)")
ax.axhspan(0.6, 2.9, color=C_OLD, alpha=0.12,
           label="audit-reported range (0.6–2.9 px)")
ax.set_xlabel("SCA")
ax.set_ylabel("poly3 sky→pixel residual rms  [px]")
ax.set_xticks(range(1, 19))
ax.set_title("Independent-validation smoothness defect: before vs after")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig4_smoothness.png")

# ---------------------------------------------------------------------------
# Persist tables + summary
# ---------------------------------------------------------------------------
pd.concat({k: t for k, t in sci_tabs.items()}, axis=1).to_csv(OUT / "per_sca_ddisp_old_new.csv")
pd.concat([sm_old.rms_px.rename("old"), sm_new.rms_px.rename("new")], axis=1
          ).to_csv(OUT / "per_sca_smoothness_rms.csv")
m0.groupby("sca").dr.describe().to_csv(OUT / "per_sca_shift_stats.csv")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"figures + tables in {OUT}")
