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
* ``--update``        resolve from the current remote manifest even when the
  destination's own lock pins older versions — the one way a plain re-run
  can move already-installed data.
* neither             an already-hydrated dir reuses its own
  ``data-versions.lock``: the run repairs or completes the installation at
  the pinned versions (assets the lock does not know yet — e.g. newly
  published ones — come from the manifest) but never upgrades, so a casual
  re-hydrate cannot silently change science data. A fresh dir uses the
  current remote manifest, falling back to the versions baked in here if
  the remote is unavailable.

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
    # Prism assets (published since v0.14.0: optical-model-prism-v0.8,
    # sensitivities-prism-v1, psf-prism-v1). Prism PSF caches share
    # psf_cache/ with the grism ones — filenames carry the STPSF filter
    # (PRISM vs GRISM0/1), so no collision.
    "optical_model_prism": Asset("optical_model_prism", ""),
    "sensitivities_prism": Asset("sensitivities_prism", "sensitivities_prism",
                                 extract=True,
                                 done_marker="sensitivity_map.yaml"),
    "psf_prism": Asset("psf_prism", "psf_cache", sca_filter=True),
    # Golden regression frames for tests/test_golden_frame.py. The tarball
    # extracts a version-named directory (golden-frames-vN/), so successive
    # versions coexist and the test's pinned GOLDEN_VERSION selects one; the
    # done marker is version-specific so publishing a new version triggers a
    # fresh download on the next hydrate.
    "golden_frames": Asset("golden_frames", "golden_frames", extract=True,
                           done_marker="golden-frames-v1/cpu/provenance.json"),
}

# Bootstrap fallback used only when the remote manifest is unavailable. The
# remote manifest in roman_disperser_data is the source of truth.
DEFAULT_MANIFEST = {
    "optical_model": "optical-model-v0.8",
    "sensitivities": "sensitivities-v1",
    "synphot": "synphot-v1",
    "psf": "psf-v1",
    "catalog": "catalog-v2",
    "optical_model_prism": "optical-model-prism-v0.8",
    "sensitivities_prism": "sensitivities-prism-v1",
    "psf_prism": "psf-prism-v1",
    "golden_frames": "golden-frames-v1",
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


def hydrate_asset(asset, tag, base, sca=None, force=False, dry_run=False,
                  locked_tag=None):
    """Install one asset at ``tag`` into ``base/asset.subdir``.

    For extract assets the files inside the tarball carry no version, so
    "already installed" is judged by the done-marker *plus* ``locked_tag`` —
    the version the data dir's lock says is installed. A marker with a
    different (or missing) lock entry means the contents are some other
    version and the asset is re-extracted, so the lock written afterwards
    always describes the actual contents. Re-extraction overwrites in place;
    files dropped between releases may linger (harmless for these assets,
    which are read by name).

    Non-extract assets get the same treatment: their filenames need not
    carry the release version either (the PSF caches don't), so a file that
    is "already present" may be from another delivery. Present files whose
    lock entry disagrees with ``tag`` are re-downloaded, keeping the
    written lock truthful about what is on disk. Because that check runs on
    the ``--sca``-filtered list, an ``--sca``-filtered install at a tag the
    lock disagrees with is *refused* when files outside the filter are
    present — otherwise the lock would record ``tag`` for files still at
    the old delivery. The one remaining ``--sca`` caveat is completeness:
    a filtered hydrate records the full release tag while only the
    requested SCAs are on disk.
    """
    out = base / asset.subdir

    if asset.extract:
        # Up-to-date check BEFORE any network call, so a pinned re-hydrate of
        # an installed asset touches nothing remote.
        present = asset.done_marker and (out / asset.done_marker).exists()
        if present and locked_tag == tag and not force:
            print(f"    up to date ({tag} per lock; {asset.done_marker} exists)")
            return
        if present and locked_tag != tag:
            have = locked_tag or "an unrecorded version"
            print(f"    installed contents are {have}, not {tag}; reinstalling")
        files = list_release_files(tag)
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

    files = list_release_files(tag)
    if asset.sca_filter and sca and locked_tag != tag:
        # A filtered install at a different tag would leave files outside
        # the filter at the old delivery while the lock records the new
        # one for all of them. Refuse rather than write a lying lock.
        wanted = tuple(f"WFI{n:02d}" for n in sca)
        outside = [n for n, _ in files
                   if (out / n).exists()
                   and not any(w in n for w in wanted)]
        if outside:
            raise ValueError(
                f"Refusing an --sca-filtered install of {tag}: "
                f"{len(outside)} present file(s) outside the requested SCAs "
                f"are {locked_tag or 'an unrecorded version'}, and the lock "
                f"would record {tag} for all of them. Re-run without --sca "
                f"to move the whole asset."
            )
    if asset.sca_filter:
        files = _filter_sca(files, sca)
    refetch = []
    if locked_tag != tag and not force:
        # Present files may be another delivery's under the same name
        # (version is not in the filename) — re-fetch them so the lock
        # written afterwards describes the actual contents.
        refetch = [n for n, _ in files if (out / n).exists()]
    if dry_run:
        extra = (f" ({len(refetch)} present but "
                 f"{locked_tag or 'unrecorded'}; would re-download)"
                 if refetch else "")
        print(f"    would download {len(files)} file(s) -> {out}{extra}")
        return
    if refetch:
        have = locked_tag or "an unrecorded version"
        print(f"    {len(refetch)} present file(s) are {have}, not {tag}; "
              f"re-downloading")
        force = True
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

def read_lock(base):
    """Versions recorded in ``<base>/data-versions.lock`` ({} if absent/bad)."""
    try:
        return json.loads((Path(base) / LOCK_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_lock(base, installed):
    """Merge ``installed`` into ``<base>/data-versions.lock`` and write it."""
    lock_path = base / LOCK_NAME
    current = read_lock(base)
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
    parser.add_argument("--update", action="store_true",
                        help="resolve versions from the current manifest even if "
                        "the data dir's own lock pins older ones; without this, "
                        "a re-hydrate reuses the pinned versions and never "
                        "upgrades installed assets")
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

    locked = read_lock(base)
    if args.lock or args.manifest or args.update or not locked:
        versions, source = resolve_manifest(args.lock, args.manifest)
    else:
        # Already-hydrated dir, no explicit version request: stay pinned to
        # its own lock. Repair or complete the installation at those
        # versions; consult the manifest only for assets the lock does not
        # know yet (e.g. newly published ones). Upgrades are opt-in
        # (--update), so a casual re-hydrate can never move science data.
        versions = dict(locked)
        source = f"{LOCK_NAME} (pinned; --update to move to current versions)"
        if any(k not in versions for k in which):
            manifest_versions, _ = resolve_manifest(None, None)
            for k in which:
                if k not in versions and k in manifest_versions:
                    versions[k] = manifest_versions[k]
    print(f"Hydrating {which} into {base}  (versions from {source})")

    installed = {}
    for key in which:
        tag = versions.get(key)
        if tag is None:
            print(f"  {key}: no version in {source}, skipping")
            continue
        print(f"  {key} @ {tag}")
        hydrate_asset(ASSETS[key], tag, base, sca=args.sca,
                      force=args.force, dry_run=args.dry_run,
                      locked_tag=locked.get(key))
        installed[key] = tag

    if args.dry_run:
        print("Dry run: no files written, no lock updated.")
        return 0

    lock_path = write_lock(base, installed)
    print(f"Wrote {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
