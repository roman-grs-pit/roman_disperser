"""Wrap disperser grism FITS through romanisim to produce L2 ASDF products.

Walks ``--input-dir`` for ``grism_*_detSCA??.fits`` files (output from
``build_grism_image.py``), launches ``romanisim-make-image`` per file as a
subprocess, and writes L2 ASDF files to a sibling tree under ``--output-dir``
with the same relative structure.

Run inside the romanisim pixi environment so ``romanisim-make-image`` is on
PATH:

    pixi run -e romanisim python scripts/wrap_with_romanisim.py \\
        --input-dir <run>/output \\
        --output-dir <run>/output_l2 \\
        --num-threads 8 \\
        --worker-index 0 --num-workers 4

Header-driven arguments (read per FITS):
    WFICENRA, WFICENDEC      -> --radec
    WFICENPA - 60            -> --roll      (focal-plane vs spacecraft offset)
    DETNUM                   -> --sca
    MA_TABLE                 -> --ma_table_number
    RNDSEED0 ^ RNDSEED1      -> --rng_seed  (XOR fold to 32-bit)

Static arguments (CLI-tunable; see --help):
    --bandpass GRISM
    --usecrds --stpsf
    --nobj 0
    --extra-counts <fits> ISIM
    --date    (default 2026-01-01T12:00:00.000)
    --level   (default 2)

Multi-worker partitioning: each worker globs the full file list, sorts,
hashes, and processes ``[idx % num_workers == worker_index]``. The manifest
hash is logged at startup so cross-worker drift can be detected from logs.

Idempotent: outputs that already exist are skipped, so re-submitting after
a partial completion just fills in the gaps.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from astropy.io import fits

LOG = logging.getLogger("wrap-romanisim")

DEFAULT_DATE = "2026-01-01T12:00:00.000"
PA_OFFSET = -60.0  # WFICENPA - 60 -> --roll (mirrors archetype wrap script).


@dataclasses.dataclass
class WrapResult:
    fits: Path
    out: Path
    status: str  # "ok" | "skipped" | "failed"
    elapsed: float
    exit_code: int | None
    log_path: Path | None
    note: str | None = None


def derive_output_path(fits_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = fits_path.relative_to(input_root)
    return output_root / rel.with_name(rel.stem + "_l2.asdf")


def read_header_args(fits_path: Path) -> dict:
    with fits.open(fits_path) as hdul:
        h = hdul[0].header
        return {
            "ra": float(h["WFICENRA"]),
            "dec": float(h["WFICENDEC"]),
            "pa": float(h["WFICENPA"]) + PA_OFFSET,
            "sca": int(h["DETNUM"]),
            "ma_table": int(h["MA_TABLE"]),
            "rng_seed": int(h["RNDSEED0"]) ^ int(h["RNDSEED1"]),
        }


def build_command(hdr: dict, fits_path: Path, out: Path,
                  date: str, bandpass: str, level: int) -> list[str]:
    return [
        "romanisim-make-image",
        str(out),
        "--extra-counts", str(fits_path), "ISIM",
        "--radec", f"{hdr['ra']}", f"{hdr['dec']}",
        "--roll", f"{hdr['pa']}",
        "--sca", str(hdr["sca"]),
        "--rng_seed", str(hdr["rng_seed"]),
        "--date", date,
        "--ma_table_number", str(hdr["ma_table"]),
        "--bandpass", bandpass,
        "--level", str(level),
        "--usecrds",
        "--stpsf",
        "--nobj", "0",
    ]


def wrap_one(fits_path: Path, input_root: Path, output_root: Path,
             log_dir: Path, date: str, bandpass: str, level: int,
             dry_run: bool) -> WrapResult:
    out = derive_output_path(fits_path, input_root, output_root)
    if out.exists():
        return WrapResult(fits_path, out, "skipped", 0.0, None, None,
                          note="output exists")

    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{out.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        hdr = read_header_args(fits_path)
    except (KeyError, OSError) as exc:
        return WrapResult(fits_path, out, "failed", 0.0, None, log_path,
                          note=f"header read error: {exc}")

    cmd = build_command(hdr, fits_path, out, date, bandpass, level)

    if dry_run:
        return WrapResult(fits_path, out, "skipped", 0.0, None, log_path,
                          note=f"dry-run: {' '.join(cmd)}")

    t0 = time.monotonic()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - t0

    if proc.returncode != 0:
        # Don't leave a partial output around; otherwise a re-run would skip it.
        if out.exists():
            out.unlink()
        return WrapResult(fits_path, out, "failed", elapsed, proc.returncode,
                          log_path, note="non-zero exit")

    return WrapResult(fits_path, out, "ok", elapsed, 0, log_path)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Root directory containing disperser FITS output")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Root for L2 ASDF output (sibling tree mirroring input)")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Per-file romanisim log directory "
                        "(default: <output-dir>/_logs)")
    p.add_argument("--num-threads", type=int, default=8,
                   help="Concurrent romanisim subprocesses per worker (default: 8)")
    p.add_argument("--worker-index", type=int, default=0,
                   help="This worker's index (0-based)")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Total workers for round-robin partitioning")
    p.add_argument("--date", default=DEFAULT_DATE,
                   help=f"UTC date passed to romanisim --date (default: {DEFAULT_DATE})")
    p.add_argument("--bandpass", default="GRISM",
                   help="--bandpass to romanisim (default: GRISM)")
    p.add_argument("--level", type=int, default=2,
                   help="L1 (1) or L2 (2) output (default: 2)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print intended romanisim commands; do not execute")
    p.add_argument("--manifest-file", type=Path, default=None,
                   help="Optional file listing FITS paths (one per line). "
                        "Skips the glob; partitioning still applies.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.num_workers <= 0 or not (0 <= args.worker_index < args.num_workers):
        p.error("--worker-index must be in [0, --num-workers)")

    if not args.dry_run and shutil.which("romanisim-make-image") is None:
        LOG.error("romanisim-make-image not on PATH; "
                  "run inside the romanisim pixi env "
                  "(pixi run -e romanisim python ...)")
        sys.exit(2)

    log_dir = args.log_dir or (args.output_dir / "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest_file is not None:
        with args.manifest_file.open() as f:
            all_files = [Path(line.strip()) for line in f if line.strip()]
        all_files.sort()
        source = f"manifest={args.manifest_file}"
    else:
        all_files = sorted(args.input_dir.glob("**/grism_*_detSCA*.fits"))
        source = f"glob={args.input_dir}"

    if not all_files:
        LOG.error("No input FITS found (%s)", source)
        sys.exit(2)

    manifest_hash = hashlib.sha256(
        "\n".join(str(f) for f in all_files).encode()
    ).hexdigest()[:12]

    my_files = [f for i, f in enumerate(all_files)
                if i % args.num_workers == args.worker_index]

    LOG.info("source=%s total=%d worker=%d/%d my_files=%d manifest_sha=%s",
             source, len(all_files), args.worker_index, args.num_workers,
             len(my_files), manifest_hash)

    if not my_files:
        LOG.warning("No files for this worker; exiting.")
        return

    t_start = time.monotonic()
    results: list[WrapResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as pool:
        futures = {
            pool.submit(
                wrap_one, f, args.input_dir, args.output_dir, log_dir,
                args.date, args.bandpass, args.level, args.dry_run,
            ): f
            for f in my_files
        }
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            LOG.info("[%-7s] %s  elapsed=%.1fs  %s",
                     res.status, res.fits.name, res.elapsed,
                     res.note or "")

    elapsed_total = time.monotonic() - t_start

    n_ok = sum(1 for r in results if r.status == "ok")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_failed = sum(1 for r in results if r.status == "failed")
    LOG.info("DONE worker=%d/%d ok=%d skipped=%d failed=%d wall=%.1fs",
             args.worker_index, args.num_workers,
             n_ok, n_skipped, n_failed, elapsed_total)

    if n_failed:
        LOG.warning("Failures (per-file logs in %s):", log_dir)
        for r in results:
            if r.status == "failed":
                LOG.warning("  %s -- %s", r.fits, r.note)
        sys.exit(1)


if __name__ == "__main__":
    main()
