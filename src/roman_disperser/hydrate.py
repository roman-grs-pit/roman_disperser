"""Hydrate vendored reference data for ``roman_disperser``.

All reference data lives in the public ``roman-grs-pit/roman_disperser_data``
GitHub releases, versioned independently of the code. This module downloads the
blessed versions into the data directory (see :mod:`roman_disperser.paths`).
It is exposed as the ``roman-disperser-hydrate`` console command, so a plain
``pip install`` user can fetch data without a repo checkout or pixi.

Version selection follows the manifest/lock model (see
``docs/data_vendoring_plan.md``):

* ``--lock FILE``     install exactly the versions pinned in ``FILE``.
* ``--manifest REF``  use the manifest at that git ref of the data repo
  (tag/sha/branch) for reproducible pins; default is the current manifest.
* neither             use the current remote manifest, falling back to the
  versions baked in here if the remote is unavailable (e.g. before it is
  published).

Every run writes the resolved versions to ``<dest>/data-versions.lock`` (merged
with any existing lock), so producing a reproducible pin is automatic.
"""

import argparse
import json
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from roman_disperser.paths import data_dir

DATA_REPO = "roman-grs-pit/roman_disperser_data"
_RELEASE_API = f"https://api.github.com/repos/{DATA_REPO}/releases/tags/"
_RAW = f"https://raw.githubusercontent.com/{DATA_REPO}/{{ref}}/manifest.json"
LOCK_NAME = "data-versions.lock"


@dataclass(frozen=True)
class Asset:
    """A vendored data asset and how to install it."""

    key: str           # manifest key
    subdir: str        # destination subdirectory under the data dir ("" = root)
    extract: bool = False     # release assets are tarballs to extract
    done_marker: str = ""     # presence check for extract assets
    sca_filter: bool = False  # honor --sca (PSF caches only)


# Registry of vendored assets. Versions (release tags) come from the manifest,
# not from here. The optical model lands directly in the data dir (subdir "");
# sensitivities/synphot are tarballs extracted into their subdirs.
ASSETS = {
    "optical_model": Asset("optical_model", ""),
    "sensitivities": Asset("sensitivities", "sensitivities", extract=True,
                           done_marker="sensitivity_map.yaml"),
    "synphot": Asset("synphot", "synphot", extract=True,
                     done_marker="roman_wfi_f158.fits"),
    "psf": Asset("psf", "psf_cache", sca_filter=True),
    "catalog": Asset("catalog", "catalogs", extract=True, done_marker="metadata.parquet"),
    # Prism assets. No releases exist yet (deliberately unpublished until the
    # prism-merge attempts are compared), so the remote manifest carries no
    # version for these keys and hydrate skips them with a message; once the
    # releases are cut and the manifest gains the keys, they hydrate like any
    # other asset. Prism PSF caches share psf_cache/ with the grism ones —
    # filenames carry the STPSF filter (PRISM vs GRISM0/1), so no collision.
    "optical_model_prism": Asset("optical_model_prism", ""),
    "sensitivities_prism": Asset("sensitivities_prism", "sensitivities_prism",
                                 extract=True,
                                 done_marker="sensitivity_map.yaml"),
    "psf_prism": Asset("psf_prism", "psf_cache", sca_filter=True),
}

# Bootstrap fallback used only when the remote manifest is unavailable. The
# remote manifest in roman_disperser_data is the source of truth.
DEFAULT_MANIFEST = {
    "optical_model": "optical-model-v0.8",
    "sensitivities": "sensitivities-v1",
    "synphot": "synphot-v1",
    "psf": "psf-v1",
    "catalog": "catalog-v2",
}


# ---------------------------------------------------------------------------
# Version resolution (manifest / lock)
# ---------------------------------------------------------------------------

def _get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def resolve_manifest(lock=None, manifest_ref=None):
    """Return ``(versions, source_label)`` mapping asset key -> release tag."""
    if lock is not None:
        versions = json.loads(Path(lock).read_text())
        return versions, f"lock:{lock}"
    ref = manifest_ref or "main"
    try:
        versions = _get_json(_RAW.format(ref=ref))
        return versions, ("manifest:current" if manifest_ref is None
                          else f"manifest:{ref}")
    except Exception as exc:  # noqa: BLE001 - any network/parse failure -> fallback
        print(f"  (remote manifest unavailable: {exc}; using built-in defaults)")
        return dict(DEFAULT_MANIFEST), "default"


# ---------------------------------------------------------------------------
# Release file discovery + download
# ---------------------------------------------------------------------------

def list_release_files(tag):
    """Return ``[(name, download_url), ...]`` for all assets in release ``tag``."""
    data = _get_json(_RELEASE_API + tag)
    return [(a["name"], a["browser_download_url"]) for a in data["assets"]]


def _filter_sca(files, sca):
    """Keep only PSF files whose name contains one of the requested SCAs."""
    if not sca:
        return files
    wanted = tuple(f"WFI{n:02d}" for n in sca)
    return [(n, u) for (n, u) in files if any(w in n for w in wanted)]


def _download(url, dest, force=False):
    """Download ``url`` to ``dest`` atomically. Return True if downloaded."""
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return True


def hydrate_asset(asset, tag, base, sca=None, force=False, dry_run=False):
    """Install one asset at ``tag`` into ``base/asset.subdir``."""
    out = base / asset.subdir
    files = list_release_files(tag)

    if asset.extract:
        present = asset.done_marker and (out / asset.done_marker).exists()
        if present and not force:
            print(f"    present, skipping ({asset.done_marker} exists)")
            return
        if dry_run:
            print(f"    would download + extract {len(files)} archive(s) -> {out}")
            return
        out.mkdir(parents=True, exist_ok=True)
        for name, url in files:
            print(f"    extracting {name} ...")
            with tempfile.NamedTemporaryFile(suffix=name, delete=False) as tf:
                tmp = Path(tf.name)
            try:
                urllib.request.urlretrieve(url, tmp)
                with tarfile.open(tmp) as tar:
                    tar.extractall(out, filter="data")
            finally:
                tmp.unlink(missing_ok=True)
        return

    if asset.sca_filter:
        files = _filter_sca(files, sca)
    if dry_run:
        print(f"    would download {len(files)} file(s) -> {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    got = 0
    for i, (name, url) in enumerate(files, 1):
        if _download(url, out / name, force):
            got += 1
            print(f"    [{i}/{len(files)}] {name}")
    print(f"    {got} downloaded, {len(files) - got} already present -> {out}")


# ---------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------

def write_lock(base, installed):
    """Merge ``installed`` into ``<base>/data-versions.lock`` and write it."""
    lock_path = base / LOCK_NAME
    current = {}
    if lock_path.exists():
        try:
            current = json.loads(lock_path.read_text())
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(installed)
    base.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return lock_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="roman-disperser-hydrate",
        description="Download vendored roman_disperser reference data.",
    )
    parser.add_argument("--dest", help="data directory (default: resolved via "
                        "roman_disperser.paths.data_dir)")
    parser.add_argument("--only", help="comma-separated assets to hydrate "
                        f"(default: all of {', '.join(ASSETS)})")
    parser.add_argument("--sca", type=int, nargs="+",
                        help="restrict PSF caches to these SCA numbers (1-18)")
    parser.add_argument("--manifest", metavar="REF",
                        help="manifest git ref of the data repo for pinned, "
                        "reproducible versions (default: current)")
    parser.add_argument("--lock", help="install exact versions from this lock file")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and report what would be fetched; download nothing")
    args = parser.parse_args(argv)

    base = data_dir(args.dest)

    if args.only:
        which = [a.strip() for a in args.only.split(",") if a.strip()]
        unknown = [a for a in which if a not in ASSETS]
        if unknown:
            parser.error(f"unknown asset(s) {unknown}; choose from {list(ASSETS)}")
    else:
        which = list(ASSETS)

    versions, source = resolve_manifest(args.lock, args.manifest)
    print(f"Hydrating {which} into {base}  (versions from {source})")

    installed = {}
    for key in which:
        tag = versions.get(key)
        if tag is None:
            print(f"  {key}: no version in {source}, skipping")
            continue
        print(f"  {key} @ {tag}")
        hydrate_asset(ASSETS[key], tag, base, sca=args.sca,
                      force=args.force, dry_run=args.dry_run)
        installed[key] = tag

    if args.dry_run:
        print("Dry run: no files written, no lock updated.")
        return 0

    lock_path = write_lock(base, installed)
    print(f"Wrote {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
