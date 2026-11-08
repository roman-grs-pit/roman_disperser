#!/usr/bin/env python
"""Build a unified roman_disperser catalog from a source table and spectra.

This is the user-facing front door of the end-to-end pipeline (see
``docs/e2e_pipeline.md``): a plain table of sources plus their spectra in,
the ``metadata.parquet`` + ``seds.zarr`` catalog of ``data/catalogs/README.md``
out, ready for ``build_dispersed_image.py --catalog-dir`` / ``catalog_dir:``.

Input 1: source table (ECSV, CSV or FITS; one row per source)

    column             unit    required  notes
    id                 -       yes       any unique label
    ra, dec            deg     yes       ICRS
    type               -       no        "PSF" (point source, default) or "SER" (Sersic)
    mag                AB      no        F158 magnitude; rescales the spectrum to it.
                                         Omit to use the spectrum's absolute FLAM as given.
    n                  -       SER       Sersic index
    half_light_radius  arcsec  SER
    pa                 deg     no        E of N (default 0)
    ba                 -       no        axis ratio b/a (default 1)
    spectrum           -       no        key into the spectra file (default: id)
    z                  -       no        stored as z_obs (bookkeeping only)

Input 2: spectra (ECSV/CSV/FITS table, or .npz)

    A table with a ``wavelength`` column in Angstrom and one FLAM column
    (erg/s/cm^2/A) per spectrum key; or an .npz with ``wavelength`` plus one
    array per key. Spectra are linearly interpolated onto the catalog grid
    (2 A, 7500-21000 A; zero outside their own coverage) -- so give them at
    <= 2 A resolution where lines matter.

Mock mode (``--mock``) writes a small test field instead: stars, emission-line
galaxies and a continuum galaxy, placed on one SCA of a pointing, with a
``truth.ecsv`` listing each source's SCA position, magnitude and lines.

Usage:
    python make_catalog.py --sources src.ecsv --spectra spec.fits --out cat/
    python make_catalog.py --mock --pointing 9.5 -44.0 0 --sca 5 --out cat/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WL_MIN, WL_MAX, WL_STEP = 7500.0, 21000.0, 2.0
C_AA = 2.99792458e18          # Angstrom / s
FNU_AB0 = 3631e-23            # erg/s/cm^2/Hz for AB mag 0
# build_dispersed_image.load_galaxy_seds zeroes galaxy FLAM bins above this
# as catalog corruption; anything this bright is refused here instead.
GALAXY_FLAM_GUARD = 1e-12


def wl_grid():
    n = int(round((WL_MAX - WL_MIN) / WL_STEP)) + 1
    return np.linspace(WL_MIN, WL_MAX, n)


def f158_weights(wl):
    """Weights so that f_nu_eff = (flam @ wnum) / den for the F158 band."""
    from astropy import units as u
    from roman_disperser.refdata import get_f158_band
    thr = get_f158_band()(wl * u.AA).value
    dl = np.gradient(wl)
    wnum = thr * wl * dl            # integral f_lam T lam dlam
    den = np.sum(thr * C_AA / wl * dl)  # integral T c/lam dlam
    return wnum, den


def ab_mag_f158(flam, wnum, den):
    fnu = (np.asarray(flam, np.float64) @ wnum) / den
    return -2.5 * np.log10(fnu / FNU_AB0)


def read_table(path):
    from astropy.table import Table
    path = Path(path)
    if path.suffix == ".csv":
        return Table.read(path, format="ascii.csv")
    return Table.read(path)


def read_spectra(path, wl):
    """Return {key: flam on wl} from a table or .npz."""
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path)
        src_wl = np.asarray(d["wavelength"], np.float64)
        cols = {k: np.asarray(d[k], np.float64) for k in d.files if k != "wavelength"}
    else:
        t = read_table(path)
        src_wl = np.asarray(t["wavelength"], np.float64)
        cols = {k: np.asarray(t[k], np.float64) for k in t.colnames if k != "wavelength"}
    if np.any(np.diff(src_wl) <= 0):
        raise ValueError(f"{path}: wavelength must be strictly increasing")
    step = np.median(np.diff(src_wl))
    if step > WL_STEP * 1.01:
        print(f"note: {path} sampled at {step:.2f} A (> {WL_STEP} A); "
              "narrow lines will be broadened by the interpolation")
    return {k: np.interp(wl, src_wl, v, left=0.0, right=0.0) for k, v in cols.items()}


def write_catalog(out, rows, seds, wl):
    """rows: list of dicts; seds: [N, n_wl] absolute FLAM per row."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zarr
    from zarr.codecs import BloscCodec

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    wnum, den = f158_weights(wl)
    mags = np.array([ab_mag_f158(s, wnum, den) for s in seds])
    f158 = (10.0 ** (-0.4 * mags)).astype(np.float32)
    is_star = np.array([r["type"] == "PSF" for r in rows])

    # Stars: template normalized to 0 mag F158, flux_scale = F158 (maggies).
    # Galaxies: pre-scaled absolute FLAM, flux_scale = 1 (catalog convention).
    star_idx = np.cumsum(is_star) - 1
    gal_idx = np.cumsum(~is_star) - 1
    star_seds = np.array([seds[i] / f158[i] for i in range(n) if is_star[i]], np.float32)
    gal_seds = np.array([seds[i] for i in range(n) if not is_star[i]], np.float32)
    if gal_seds.size and np.abs(gal_seds).max() > GALAXY_FLAM_GUARD:
        raise ValueError("a galaxy SED exceeds |FLAM| 1e-12, which the disperser "
                         "treats as corruption; is it in erg/s/cm^2/A?")

    cols = {
        "ra": np.array([r["ra"] for r in rows], np.float64),
        "dec": np.array([r["dec"] for r in rows], np.float64),
        "type": np.array([r["type"] for r in rows], dtype=object),
        "n": np.array([r.get("n", 0.0) if r["type"] == "SER" else 0.0 for r in rows], np.float32),
        "half_light_radius": np.array([r.get("half_light_radius", 0.0) if r["type"] == "SER" else 0.0
                                       for r in rows], np.float32),
        "pa": np.array([r.get("pa", 0.0) for r in rows], np.float32),
        "ba": np.array([r.get("ba", 1.0) for r in rows], np.float32),
        "F158": f158,
        "z_obs": np.array([r.get("z", 0.0) for r in rows], np.float32),
        "z_cosmo": np.array([r.get("z", 0.0) for r in rows], np.float32),
        "sed_index": np.where(is_star, star_idx, gal_idx).astype(np.int32),
        "flux_scale": np.where(is_star, f158, 1.0).astype(np.float32),
        "sim": np.where(is_star, 0, 0).astype(np.int16),
        "src_index": np.arange(n, dtype=np.int32),
        "src_id": np.array([str(r["id"]) for r in rows], dtype=object),
    }
    schema = pa.schema([
        pa.field("ra", pa.float64(), metadata={"unit": "deg"}),
        pa.field("dec", pa.float64(), metadata={"unit": "deg"}),
        pa.field("type", pa.string()),
        pa.field("n", pa.float32()),
        pa.field("half_light_radius", pa.float32(), metadata={"unit": "arcsec"}),
        pa.field("pa", pa.float32(), metadata={"unit": "deg"}),
        pa.field("ba", pa.float32()),
        pa.field("F158", pa.float32(), metadata={"unit": "maggies"}),
        pa.field("z_obs", pa.float32()),
        pa.field("z_cosmo", pa.float32()),
        pa.field("sed_index", pa.int32()),
        pa.field("flux_scale", pa.float32()),
        pa.field("sim", pa.int16()),
        pa.field("src_index", pa.int32()),
        pa.field("src_id", pa.string()),
    ])
    pq.write_table(pa.table(cols, schema=schema), out / "metadata.parquet")

    comp = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
    store = zarr.open(str(out / "seds.zarr"), mode="w")
    store.create_array("wavelengths", data=wl.astype(np.float64), compressors=comp,
                       attributes={"units": "Angstrom"})
    if star_seds.size == 0:
        star_seds = np.zeros((1, wl.size), np.float32)
    store.create_array("star_seds", data=star_seds, chunks=(1, wl.size), compressors=comp,
                       attributes={"units": "FLAM normalized to 0 mag F158"})
    g = store.create_group("galaxy_seds")
    if gal_seds.size:
        store.create_array("galaxy_seds/sim_000", data=gal_seds, chunks=(1, wl.size),
                           compressors=comp, attributes={"units": "FLAM (erg/s/cm^2/A)"})
        g.attrs.update({"n_partitions": 1})
    else:
        g.attrs.update({"n_partitions": 0})
    store.attrs.update({"format_version": "1.0", "description": "make_catalog.py"})
    return mags


def from_inputs(args):
    wl = wl_grid()
    t = read_table(args.sources)
    spectra = read_spectra(args.spectra, wl)
    wnum, den = f158_weights(wl)
    rows, seds = [], []
    for r in t:
        d = {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in t.colnames}
        d["type"] = str(d.get("type", "PSF")).upper()
        if d["type"] not in ("PSF", "SER"):
            raise ValueError(f"source {d['id']}: type must be PSF or SER")
        if d["type"] == "SER" and ("n" not in d or "half_light_radius" not in d):
            raise ValueError(f"source {d['id']}: SER needs n and half_light_radius")
        key = str(d.get("spectrum", d["id"]))
        if key not in spectra:
            raise KeyError(f"source {d['id']}: spectrum {key!r} not in {args.spectra}")
        s = spectra[key].copy()
        if "mag" in d and d["mag"] is not None and np.isfinite(d["mag"]):
            s *= 10 ** (-0.4 * (d["mag"] - ab_mag_f158(s, wnum, den)))
        rows.append(d); seds.append(s)
    return rows, np.array(seds), wl, None


def mock(args):
    """A few stars, emission-line galaxies and a continuum galaxy on one SCA."""
    from roman_disperser import optical_model_jax as omj
    rng = np.random.default_rng(args.seed)
    wl = wl_grid()
    ra0, dec0, pa0 = args.pointing
    # candidate sky positions in the field; keep those whose direct image
    # lands well inside the target SCA
    r = 0.45 * np.sqrt(rng.random(400000)); th = 2 * np.pi * rng.random(400000)
    ra = ra0 + r * np.cos(th) / np.cos(np.radians(dec0)); dec = dec0 + r * np.sin(th)
    xf, yf = omj.get_fpa_pos(ra, dec, ra0, dec0, pa0)
    from roman_disperser import paths
    from roman_disperser.optical_model import RomanOpticalModel
    model = RomanOpticalModel(config_file=str(paths.optical_model_path(None)))
    det = omj.make_sca_payload(model, sca=args.sca, order="1")["det"]
    xs, ys = omj._fpa_to_sca_f64(det, np.asarray(xf, np.float64), np.asarray(yf, np.float64))
    xs, ys = np.asarray(xs), np.asarray(ys)
    ok = np.flatnonzero((xs > 400) & (xs < 3700) & (ys > 400) & (ys < 3700))
    if ok.size < 50:
        raise RuntimeError(f"only {ok.size} candidates land on SCA {args.sca}")
    pick = rng.choice(ok, size=args.n_star + args.n_elg + args.n_cont, replace=False)

    flat_ab = C_AA * FNU_AB0 / wl**2   # FLAM of a 0-mag flat-f_nu source
    def gauss(c, fwhm):
        s = fwhm / 2.3548
        return np.exp(-0.5 * ((wl - c) / s) ** 2) / (np.sqrt(2 * np.pi) * s)  # per A, unit area

    rows, seds, truth = [], [], []
    k = 0
    for i in range(args.n_star):
        mag = rng.uniform(17, 21)
        t_eff = rng.choice([3500, 5800, 9000])
        # blackbody shape, AB-normalized later
        bb = 1 / (wl**5 * (np.exp(1.4388e8 / (wl * t_eff)) - 1))
        rows.append(dict(id=f"star{i}", ra=ra[pick[k]], dec=dec[pick[k]], type="PSF", mag=mag))
        seds.append(bb); truth.append(dict(id=f"star{i}", kind="star", mag=mag, teff=t_eff, lines="")); k += 1
    for i in range(args.n_elg):
        mag = rng.uniform(19.5, 22); z = rng.uniform(1.0, 1.8)
        cont = flat_ab * (wl / 15000.0) ** -0.5
        # H-alpha + [NII] at the redshift (only those landing in band count)
        lines = [(6564.6, 1.0), (6585.3, 0.3), (6549.9, 0.1), (5008.2, 0.6), (4862.7, 0.3)]
        ew_rest = rng.uniform(80, 300)           # H-alpha rest EW, A
        spec = cont.copy()
        cont_ha = np.interp(6564.6 * (1 + z), wl, cont)
        ha_flux = ew_rest * (1 + z) * cont_ha    # observed-frame flux units of cont*A
        lt = []
        for lam0, rel in lines:
            lo = lam0 * (1 + z)
            if WL_MIN < lo < WL_MAX:
                spec += rel * ha_flux * gauss(lo, 12.0)  # 12 A FWHM (~240 km/s), resolved on the 2 A grid
                # observed-frame EW (A): independent of the later rescaling to `mag`
                lt.append(f"{lo:.1f}:{rel * ha_flux / np.interp(lo, wl, cont):.1f}")
        rows.append(dict(id=f"elg{i}", ra=ra[pick[k]], dec=dec[pick[k]], type="SER", mag=mag,
                         n=1.0, half_light_radius=float(rng.uniform(0.15, 0.4)),
                         pa=float(rng.uniform(0, 180)), ba=float(rng.uniform(0.4, 1.0)), z=z))
        seds.append(spec)
        truth.append(dict(id=f"elg{i}", kind="elg", mag=mag, z=z, lines=";".join(lt))); k += 1
    for i in range(args.n_cont):
        mag = rng.uniform(19, 21)
        spec = flat_ab * (wl / 15000.0) ** 1.0
        spec[wl < 4000 * (1 + 0.9)] *= 0.3       # a 4000-A-break-like step at z=0.9
        rows.append(dict(id=f"cont{i}", ra=ra[pick[k]], dec=dec[pick[k]], type="SER", mag=mag,
                         n=4.0, half_light_radius=0.5, pa=30.0, ba=0.7, z=0.9))
        seds.append(spec); truth.append(dict(id=f"cont{i}", kind="continuum", mag=mag, z=0.9, lines="")); k += 1

    wnum, den = f158_weights(wl)
    seds = np.array([s * 10 ** (-0.4 * (r["mag"] - ab_mag_f158(s, wnum, den)))
                     for r, s in zip(rows, seds)])
    # truth.ecsv lines are "observed_wavelength_A:observed_EW_A", scale-free
    for t, r, s, p in zip(truth, rows, seds, pick):
        t.update(ra=r["ra"], dec=r["dec"], x_direct=float(xs[p]), y_direct=float(ys[p]))
    return rows, seds, wl, truth


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--sources"); p.add_argument("--spectra")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--pointing", nargs=3, type=float, metavar=("RA", "DEC", "PA"))
    p.add_argument("--sca", type=int, default=5)
    p.add_argument("--n-star", type=int, default=8)
    p.add_argument("--n-elg", type=int, default=12)
    p.add_argument("--n-cont", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.mock:
        if args.pointing is None:
            p.error("--mock needs --pointing RA DEC PA")
        rows, seds, wl, truth = mock(args)
    else:
        if not (args.sources and args.spectra):
            p.error("give --sources and --spectra, or --mock")
        rows, seds, wl, truth = from_inputs(args)
    mags = write_catalog(args.out, rows, seds, wl)
    for r, m in zip(rows, mags):
        if "mag" in r and abs(m - r["mag"]) > 1e-3:
            raise AssertionError(f"{r['id']}: F158 {m:.4f} != requested {r['mag']:.4f}")
    if truth is not None:
        from astropy.table import Table
        Table(rows=[{**t, "mag_f158": float(m)} for t, m in zip(truth, mags)]).write(
            Path(args.out) / "truth.ecsv", overwrite=True)
    print(f"wrote {len(rows)} sources ({sum(r['type']=='PSF' for r in rows)} PSF, "
          f"{sum(r['type']=='SER' for r in rows)} SER) to {args.out}; "
          f"F158 mags {mags.min():.2f}-{mags.max():.2f}")


if __name__ == "__main__":
    main()
