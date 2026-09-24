"""Water-ice throughput model for the dispersers.

Thin films of water ice grow on the WFI detectors between decontamination
("decon") cycles and change the throughput as a function of wavelength by up
to about +-20 % (thin-film interference -- the ratio can exceed 1, this is not
a pure loss). This module turns the "toy" ice model from
``RomanSpaceTelescope/ice-toy-model`` (``ice_rep_sim.py``, 2026-09-21) into a
per-wavelength multiplicative factor that the dispersers apply to each
source's flux vector.

Model
-----
For a detector position (x, y) and observation time t (MJD):

    d(x, y, t) = r(bin(y), bin(x)) * dt,   dt = (t - t0) mod P

    f(lambda; x, y, t) = T(lambda; d) / T(lambda; 0)

* ``r`` is a growth-rate map [nm/day] on a coarse grid of square bins
  (``pix_per_bin`` = 32 detector pixels in the toy model -> 128 x 128 bins
  per SCA), constant in time. The bin lookup is *nearest bin* (integer
  division), exactly as the toy model does it.
* ``t0`` is the decon epoch and ``P`` the decon period: the ice thickness is
  reset to zero over the whole FPA every ``P`` days (a steady growth rate
  between resets). Defaults follow the toy model: MJD 61295.0
  (2026-09-12T00:00 UTC) and 20 days. MJDs before ``t0`` are rejected.
* ``T(lambda; d) / T(lambda; 0)`` is a table of transmission ratios relative
  to the ice-free state, tabulated at a uniform set of thicknesses (0 ... 300
  nm in 30 nm steps in the current SCIPA-TVAC-derived table) on a fine
  wavelength grid. It is interpolated **linearly in thickness** between the
  two bracketing columns (again matching the toy model) and pre-interpolated
  in wavelength onto the run's wavelength grid once, at load time. Thickness
  beyond the table's last column is clamped to it; ``load_ice_payload`` warns
  at load time when the model can reach that regime (``max(r) * P`` exceeds
  the table), because a warning cannot be raised inside jitted code.

The existing sensitivity curves are, by this definition, the ice-free
(``d = 0``) baseline.

Where the factor is applied
---------------------------
``deposit_stack_native`` deposits one native-resolution stamp per fine
wavelength at a single dispersed centre, so the factor is evaluated once per
fine wavelength **at the stamp's dispersed centre** and multiplied into the
flux vector before the deposit (``star_disperser.disperse_star_psf``,
``galaxy_disperser.disperse_galaxy``). This is within the model's own
32-pixel resolution except where a stamp straddles a bin boundary -- an
accepted approximation (decision 2026-09-24; a per-pixel factor inside the
deposit is the fallback if it ever matters).

Units and conventions
---------------------
* Wavelengths: the table file is in **nm**; the payload and the dispersers
  work in **microns** (the run grid). Thickness: **nm**. Time: **days** (MJD).
* Positions are 1-indexed FITS SCA pixels, as everywhere in this package;
  they are converted to 0-indexed before binning (the toy model's pixel
  coordinates are 0-based). Bin indices are clipped to the map, so dispersed
  positions that fall off the detector (whose deposits are dropped anyway)
  simply take the edge bin's thickness.
* The toy model's within-detector (row, col) orientation is assumed to
  coincide with our (y, x) with no flips. The map is "representative, not
  realistic", so this is a convention to pin down, not a correctness risk.

Static vs dynamic
-----------------
The ratio table on the run grid (``trans``, [n_thick, N_wl]) and the rate map
are pointing-independent and are captured in the dispersers' closures like
the optical and PSF payloads. The *thickness map* ``r * dt`` depends on the
observation time, so it is passed as a **dynamic argument**
(``ice_thickness_nm``, [n_bins, n_bins]) -- baking it into the closure would
recompile every pointing.

Data location and embargo
-------------------------
The rate mosaic and the transmission table are **not** distributed with the
package or the public reference-data releases: the table header restricts it
to the Roman calibration and spectrophotometric working groups. They are
read from a directory (default ``<data>/ice/``, see ``paths.ice_dir``) that
the user populates by hand, indexed by ``ice_map.yaml``::

    rate_mosaic: ice_rate_mosaic.npz     # toy-model growth-rate mosaic
    epoch_mjd: 61295.0                   # optional, default below
    decon_period_days: 20.0              # optional, default below
    tables:                              # per-SCA ratio table (ECSV)
      SCA1: spectral_responses_SCA10.ecsv
      ...
      SCA18: spectral_responses_SCA10.ecsv

Only an SCA10 table exists today; every SCA points at it.
"""

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml
from astropy.table import Table

ICE_MAP_FILE = "ice_map.yaml"
DEFAULT_EPOCH_MJD = 61295.0        # 2026-09-12T00:00:00 UTC (toy model)
DEFAULT_DECON_PERIOD_DAYS = 20.0   # toy model
SCA_NPIX = 4096                    # toy-model detector extent per axis
_THICKNESS_COLUMN_PREFIX = "T_ratio_d"


# ---------------------------------------------------------------------------
# Host-side loading
# ---------------------------------------------------------------------------

def read_ice_map(ice_dir):
    """Parse ``ice_map.yaml`` from ``ice_dir`` and fill in the defaults.

    Returns the dict with keys ``rate_mosaic``, ``tables`` (``{"SCAn": file}``),
    ``epoch_mjd`` and ``decon_period_days``.
    """
    ice_dir = Path(ice_dir)
    map_path = ice_dir / ICE_MAP_FILE
    if not map_path.exists():
        raise FileNotFoundError(
            f"{map_path} not found. The ice model data are embargoed and not "
            f"hydrated; copy them into {ice_dir} by hand (see docs/ice.md).")
    with open(map_path) as f:
        ice_map = yaml.safe_load(f)
    for key in ("rate_mosaic", "tables"):
        if key not in ice_map:
            raise ValueError(f"{map_path}: missing required key '{key}'")
    ice_map.setdefault("epoch_mjd", DEFAULT_EPOCH_MJD)
    ice_map.setdefault("decon_period_days", DEFAULT_DECON_PERIOD_DAYS)
    return ice_map


def load_rate_map(npz_path, sca):
    """Cut one SCA's growth-rate map [nm/day] out of the FPA mosaic.

    The ``.npz`` holds ``rate_mosaic`` [rows, cols] (NaN between detectors),
    ``det_row_offsets`` / ``det_col_offsets`` (dicts, 1-indexed SCA -> bin
    offset of the detector's corner in the mosaic) and ``pix_per_bin``.

    Returns
    -------
    rate : np.ndarray [n_bins, n_bins] float32
    pix_per_bin : int
    """
    with np.load(npz_path, allow_pickle=True) as z:
        mosaic = np.asarray(z["rate_mosaic"], dtype=np.float32)
        row_off = z["det_row_offsets"].item()
        col_off = z["det_col_offsets"].item()
        pix_per_bin = int(z["pix_per_bin"])
    if sca not in row_off or sca not in col_off:
        raise ValueError(f"SCA {sca} not in rate mosaic offsets "
                         f"(have {sorted(row_off)})")
    n_bins = SCA_NPIX // pix_per_bin
    r0, c0 = row_off[sca], col_off[sca]
    rate = mosaic[r0:r0 + n_bins, c0:c0 + n_bins]
    if rate.shape != (n_bins, n_bins):
        raise ValueError(f"SCA {sca} block {rate.shape} does not fit in the "
                         f"mosaic {mosaic.shape}")
    if not np.all(np.isfinite(rate)):
        raise ValueError(f"SCA {sca} growth-rate map contains non-finite "
                         f"values (mosaic gaps?)")
    return np.ascontiguousarray(rate), pix_per_bin


def load_transmission_table(ecsv_path, wavelengths_um):
    """Read a ratio table and resample it onto the run wavelength grid.

    The ECSV has a ``wavelength`` column [nm] and one ``T_ratio_d<N>nm``
    column per tabulated thickness ``N`` [nm]; the thicknesses must be
    uniformly spaced and start at 0. Wavelengths outside the table hold the
    edge values (``np.interp``); the current table (480-2300 nm) covers both
    WFI dispersers' bands, so this never triggers in production.

    Returns
    -------
    trans : np.ndarray [n_thick, N_wl] float32
    thicknesses_nm : np.ndarray [n_thick] float64
    """
    tab = Table.read(ecsv_path)
    cols = [c for c in tab.colnames if c.startswith(_THICKNESS_COLUMN_PREFIX)]
    if not cols:
        raise ValueError(f"{ecsv_path}: no '{_THICKNESS_COLUMN_PREFIX}*' columns")
    thick = np.array([float(c[len(_THICKNESS_COLUMN_PREFIX):-2]) for c in cols])
    order = np.argsort(thick)
    thick, cols = thick[order], [cols[i] for i in order]
    if len(thick) < 2:
        raise ValueError(f"{ecsv_path}: need >= 2 thickness columns")
    steps = np.diff(thick)
    if thick[0] != 0.0 or not np.allclose(steps, steps[0]):
        raise ValueError(f"{ecsv_path}: thickness columns must start at 0 and "
                         f"be uniformly spaced, got {thick}")
    wl_nm = np.asarray(tab["wavelength"], dtype=np.float64)
    run_nm = np.asarray(wavelengths_um, dtype=np.float64) * 1e3
    trans = np.stack([np.interp(run_nm, wl_nm, np.asarray(tab[c], dtype=np.float64))
                      for c in cols]).astype(np.float32)
    return trans, thick


def load_ice_payload(ice_dir, sca, wavelengths_um):
    """Build the (pointing-independent) ice payload for one SCA.

    Parameters
    ----------
    ice_dir : str or Path
        Directory holding ``ice_map.yaml`` and the files it names.
    sca : int
        1-indexed SCA number.
    wavelengths_um : array [N_wl]
        The run's wavelength grid (microns). The dispersers must later be
        called with exactly this grid: the payload's ``trans`` is aligned
        with it by index.

    Returns
    -------
    payload : dict
        ``rate_nm_per_day`` (jnp [n_bins, n_bins]), ``pix_per_bin`` (int),
        ``trans`` (jnp [n_thick, N_wl]), ``thickness_step_nm``,
        ``thickness_max_nm``, ``epoch_mjd``, ``decon_period_days`` (floats),
        ``n_wl`` (int), plus provenance strings ``sca``, ``table_file``,
        ``rate_file``.
    """
    ice_dir = Path(ice_dir)
    ice_map = read_ice_map(ice_dir)
    sca_key = f"SCA{int(sca)}"
    if sca_key not in ice_map["tables"]:
        raise ValueError(f"{sca_key} not in {ICE_MAP_FILE} 'tables' "
                         f"(have {sorted(ice_map['tables'])})")
    table_file = ice_map["tables"][sca_key]
    rate_file = ice_map["rate_mosaic"]

    rate, pix_per_bin = load_rate_map(ice_dir / rate_file, int(sca))
    trans, thick = load_transmission_table(ice_dir / table_file, wavelengths_um)
    step = float(thick[1] - thick[0])
    thickness_max = float(thick[-1])
    period = float(ice_map["decon_period_days"])

    reachable = float(rate.max()) * period
    if reachable > thickness_max:
        warnings.warn(
            f"ice: SCA {sca} can reach {reachable:.1f} nm within the "
            f"{period:g}-day decon period but the table ends at "
            f"{thickness_max:g} nm; thicker ice is clamped to the last column.",
            stacklevel=2)

    return {
        "rate_nm_per_day": jnp.asarray(rate),
        "pix_per_bin": int(pix_per_bin),
        "trans": jnp.asarray(trans),
        "thickness_step_nm": step,
        "thickness_max_nm": thickness_max,
        "epoch_mjd": float(ice_map["epoch_mjd"]),
        "decon_period_days": period,
        "n_wl": int(trans.shape[1]),
        "sca": int(sca),
        "table_file": str(table_file),
        "rate_file": str(rate_file),
    }


# ---------------------------------------------------------------------------
# Time -> thickness (host side, per pointing)
# ---------------------------------------------------------------------------

def time_since_decon(mjd, epoch_mjd=DEFAULT_EPOCH_MJD,
                     period_days=DEFAULT_DECON_PERIOD_DAYS):
    """Days since the most recent decon before ``mjd``: ``(mjd - t0) mod P``.

    Kept separate from the thickness map so the decon schedule can be
    replaced (e.g. by a list of actual decon dates) without touching the
    rest. Raises for ``mjd < epoch_mjd`` (undefined in the toy model).
    """
    mjd = float(mjd)
    if mjd < epoch_mjd:
        raise ValueError(f"MJD {mjd} is before the decon epoch {epoch_mjd}")
    if period_days <= 0:
        raise ValueError(f"decon period must be positive, got {period_days}")
    return (mjd - epoch_mjd) % period_days


def thickness_map(payload, mjd):
    """Ice thickness [nm] on the payload's bin grid at time ``mjd``.

    ``rate * time_since_decon``; **not** clamped -- the clamp to the table
    lives in :func:`transmission_factor`, so this map reports the model's
    actual thickness (for headers / diagnostics).

    Returns a jnp array [n_bins, n_bins] float32 -- the dynamic argument the
    ice-enabled dispersers take.
    """
    dt = time_since_decon(mjd, payload["epoch_mjd"], payload["decon_period_days"])
    return (payload["rate_nm_per_day"] * jnp.float32(dt)).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Jittable factor
# ---------------------------------------------------------------------------

def transmission_factor(payload, ice_thickness_nm, xsca, ysca):
    """Relative transmission at each (position, wavelength) pair.

    Vectorised over the fine-wavelength axis: element ``k`` is evaluated at
    detector position ``(xsca[k], ysca[k])`` and at the run wavelength with
    index ``k`` (``payload["trans"][:, k]``), i.e. the caller passes the
    dispersed centre positions of the run's wavelength grid, in grid order.

    Parameters
    ----------
    payload : dict
        From :func:`load_ice_payload` (closure constant).
    ice_thickness_nm : jnp.ndarray [n_bins, n_bins]
        From :func:`thickness_map` (dynamic, per pointing).
    xsca, ysca : jnp.ndarray [N_wl]
        Positions, 1-indexed FITS SCA pixels.

    Returns
    -------
    factor : jnp.ndarray [N_wl] float32
    """
    trans = payload["trans"]
    n_thick, n_wl = trans.shape
    if xsca.shape[-1] != n_wl:
        raise ValueError(
            f"ice payload was built for {n_wl} wavelengths but got "
            f"{xsca.shape[-1]} positions; the dispersers must be called with "
            f"the same wavelength grid the payload was loaded with")
    n_bins = ice_thickness_nm.shape[0]
    ppb = payload["pix_per_bin"]
    step = payload["thickness_step_nm"]

    # Nearest bin (toy model): 0-indexed pixel // pix_per_bin, clipped.
    ix = jnp.clip(jnp.floor((xsca - 1.0) / ppb), 0, n_bins - 1).astype(jnp.int32)
    iy = jnp.clip(jnp.floor((ysca - 1.0) / ppb), 0, n_bins - 1).astype(jnp.int32)
    d = ice_thickness_nm[iy, ix]

    # Linear interpolation between bracketing thickness columns, clamped to
    # the table (the last column beyond its end, column 0 below zero).
    f = jnp.clip(d / step, 0.0, float(n_thick - 1))
    i0 = jnp.clip(jnp.floor(f), 0, n_thick - 2).astype(jnp.int32)
    t = f - i0
    k = jnp.arange(n_wl)
    lo = trans[i0, k]
    hi = trans[i0 + 1, k]
    return lo + t * (hi - lo)
