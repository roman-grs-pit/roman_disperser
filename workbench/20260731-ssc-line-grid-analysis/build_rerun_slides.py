"""Build the v0.13.0 rerun before/after addendum deck.

Addendum to the wave-1b draft deck (line_grid_wave1b_draft.pptx): four content
slides, one per question of the 2026-07-31 before/after briefing. All numbers
are quoted from that briefing / summary.json — regenerate before_after.py
outputs first if the stores change.

Output goes to the S3 analysis dir, NOT the repo (Nikhil's 2026-07-31
decision: decks live with the data). python-pptx runs from an ephemeral env:

    pixi exec --spec python=3.12 --spec python-pptx --spec pillow \
        python build_rerun_slides.py
"""

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ANALYSIS = Path("/mnt/roman-science/grs/line-tests-20260731/analysis")
OUT = ANALYSIS / "line_grid_v0130_rerun_addendum.pptx"

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.55)
INK = RGBColor(0x20, 0x20, 0x20)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)

prs = Presentation()
prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
BLANK = prs.slide_layouts[6]


def add_text(slide, left, top, width, height, lines, size=14, color=INK):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = color
    return box


def title(slide, text):
    add_text(slide, MARGIN, Inches(0.3), SLIDE_W - 2 * MARGIN, Inches(0.7),
             [text], size=26)


def add_fig(slide, path, left, top, max_w, max_h):
    w, h = Image.open(path).size
    scale = min(max_w / w, max_h / h)
    slide.shapes.add_picture(str(path), left, top,
                             width=int(w * scale), height=int(h * scale))


# 1 — title/context
s = prs.slides.add_slide(BLANK)
add_text(s, MARGIN, Inches(2.2), SLIDE_W - 2 * MARGIN, Inches(1.2),
         ["SSC line grid — v0.13.0 rerun, before/after"], size=34)
add_text(s, MARGIN, Inches(3.6), SLIDE_W - 2 * MARGIN, Inches(2.6), [
    "Waves 1b / 2 / 3 regenerated 2026-07-31 with identical catalogs after the "
    "sky→FPA precision (v0.11), gnomonic projection (v0.12), and "
    "reproducibility (v0.13) fixes.",
    "Stores: line-tests-20260731{,-cont,-gal}  ·  36 SLURM jobs (7011–7046), no failures.",
    "Truth tables now self-identify: codever 0.13.0, git 3d94b0f (embedded parquet metadata).",
    "Addendum to line_grid_wave1b_draft.pptx — full details in briefing_v0130_rerun.md.",
], size=16, color=MUTED)

# 2 — blast radius
s = prs.slides.add_slide(BLANK)
title(s, "Reported science metrics barely move")
add_fig(s, ANALYSIS / "fig3_science_stability.png",
        MARGIN, Inches(1.3), Inches(5.6), Inches(5.6))
add_text(s, Inches(6.6), Inches(1.6), Inches(6.2), Inches(5), [
    "Per-SCA median d_disp, 20260724 vs 20260731: identity line.",
    "Max |Δ| over 18 SCAs ≤ 0.0059 px (stpsf pa000); "
    "0.0046 / 0.0047 / 0.0029 px for the other combos.",
    "Reading: the package was internally self-consistent — placement errors "
    "cancelled between simulation and prediction, so the shipped wave "
    "conclusions were robust.",
    "Residual ~0.005 px deltas ≈ edge-source reshuffling (row counts "
    "25,680 → 25,695) + estimator noise.",
], size=15)

# 3 — placement shifts
s = prs.slides.add_slide(BLANK)
title(s, "Absolute placement moved up to ~7 px at the field edge")
add_fig(s, ANALYSIS / "fig1_shift_map.png",
        MARGIN, Inches(1.25), Inches(5.9), Inches(5.9))
add_fig(s, ANALYSIS / "fig2_shift_cdf.png",
        Inches(6.7), Inches(1.5), Inches(6.3), Inches(3.0))
add_text(s, Inches(6.7), Inches(4.8), Inches(6.2), Inches(2.3), [
    "|dr|: median 1.82 px, p95 4.87 px, max 6.82 px "
    "(per-SCA medians 0.54–3.21 px, worst SCA 18).",
    "Grows with field radius — the audit's radial signature.",
    "Matches the sequencing doc's predicted 1.84 px median / 7.1 px max.",
    "Anyone dispersing from the shipped absolute positions should use the "
    "20260731 stores.",
], size=15)

# 4 — smoothness gate
s = prs.slides.add_slide(BLANK)
title(s, "Independent-validation smoothness defect: eliminated")
add_fig(s, ANALYSIS / "fig4_smoothness.png",
        MARGIN, Inches(1.5), Inches(7.6), Inches(4.4))
add_text(s, Inches(8.4), Inches(1.7), Inches(4.4), Inches(5), [
    "Per-SCA degree-3 sky→pixel fit residual rms:",
    "pre-fix 0.39–1.09 px  →  v0.13.0: 1.1e-4–3.6e-4 px.",
    "~3.5 orders of magnitude — smooth at float precision.",
    "The audit's one real defect (0.6–2.9 px non-smooth map) was exactly "
    "the v0.11/v0.12 placement bugs. Thread closed.",
], size=15)

# 5 — replication + provenance
s = prs.slides.add_slide(BLANK)
title(s, "Waves 2 and 3 replicate; provenance")
add_text(s, MARGIN, Inches(1.4), SLIDE_W - 2 * MARGIN, Inches(5.6), [
    "Wave 2 (continuum null): max per-SCA |Δ median d_disp| vs wave 1b = "
    "0.0045 / 0.0043 / 0.0033 / 0.0022 px — old bound 0.0046 px. "
    "Continuum still does not move line centroids.",
    "Wave 3 (morphology transfer): star→galaxy per-SCA r = 0.998 / 0.996 "
    "(STPSF), max |gal−star| ≤ 0.008 px. Gaussian-control r ≈ 0.09 is "
    "noise-on-noise, as in the old campaign — the headline r = 0.998 is the "
    "STPSF number.",
    "Not compared: ISIM realisations (per-SCA keys change every draw; GPU "
    "scatter-add nondeterminism #22). All gates are MODEL + tolerances.",
    "",
    "Code: roman_disperser v0.13.0; line_test branch 3d94b0f (merges main "
    "12b98fc). Catalogs bit-copied from 20260724 stores; "
    "CRDS_CONTEXT=roman_0058.pmap; seed 42; MA 1036.",
    "Analysis: workbench/20260731-ssc-line-grid-analysis/before_after.py; "
    "outputs + this deck in line-tests-20260731/analysis/.",
], size=16)

prs.save(OUT)
print(f"wrote {OUT}")
