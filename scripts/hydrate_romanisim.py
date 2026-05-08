"""Hydrate the reference data caches required by romanisim.

Syncs the CRDS pmap + references into ``$CRDS_PATH`` and reports STPSF
data status. The cache locations are taken from the environment (we
expect ``CRDS_PATH``, ``CRDS_SERVER_URL``, ``STPSF_PATH`` to be set in
your shell — typically via ``~/.bashrc`` — see INSTALL.md). Idempotent.

Run via:

    pixi run -e romanisim hydrate-romanisim
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _patch_crds_locator() -> None:
    # CRDS 13.1.16 has two issues that break bulk ``crds sync --fetch-references``
    # for Roman before any reference is downloaded:
    #
    # 1. ``crds.core.utils.header_to_instrument({})`` raises ``KeyError`` rather
    #    than returning ``None``, so the Roman locator's intended fallback to
    #    ``file_to_instrument(refname)`` is never reached.
    # 2. ``file_to_instrument(refname)`` opens the file with astropy.io.fits to
    #    read the ``TELESCOP`` keyword, but the file doesn't exist yet at this
    #    point in sync. Some Roman reference files (e.g. the SFD galactic dust
    #    map) also have no instrument keyword to read.
    #
    # Roman has a single instrument (WFI), so we patch both lookups to return
    # ``"WFI"`` as a safe default.
    from crds.core import utils

    orig_header = utils.header_to_instrument
    orig_file = utils.file_to_instrument

    def safe_header_to_instrument(header, default=None):
        try:
            return orig_header(header, default=default)
        except KeyError:
            return default

    def safe_file_to_instrument(filename):
        try:
            return orig_file(filename)
        except (FileNotFoundError, OSError, KeyError):
            return "WFI"

    utils.header_to_instrument = safe_header_to_instrument
    utils.file_to_instrument = safe_file_to_instrument
    # The Roman locator imports the ``utils`` module at load time; bind the
    # patched callables on its attribute too.
    try:
        from crds.roman import locate as roman_locate

        roman_locate.utils.header_to_instrument = safe_header_to_instrument
        roman_locate.utils.file_to_instrument = safe_file_to_instrument
    except ImportError:
        pass


def sync_crds() -> None:
    server = os.environ.get("CRDS_SERVER_URL")
    path = os.environ.get("CRDS_PATH")
    if not (server and path):
        print(
            "[fail] CRDS_PATH / CRDS_SERVER_URL not set in the shell.\n"
            "       Add to ~/.bashrc:\n"
            "         export CRDS_PATH=/data/npadman/refdata/crds\n"
            "         export CRDS_SERVER_URL=https://roman-crds.stsci.edu",
            file=sys.stderr,
        )
        sys.exit(2)
    Path(path).mkdir(parents=True, exist_ok=True)

    _patch_crds_locator()

    context = os.environ.get("CRDS_CONTEXT")
    if context:
        argv = ["crds.sync", "--contexts", context, "--fetch-references"]
        print(f"[run] crds sync --contexts {context} --fetch-references")
    else:
        argv = ["crds.sync", "--all", "--fetch-references"]
        print("[run] crds sync --all --fetch-references (operational context)")
    print(f"      CRDS_PATH={path}")
    print(f"      CRDS_SERVER_URL={server}")

    from crds.sync import SyncScript

    saved_argv = sys.argv
    sys.argv = argv
    try:
        rc = SyncScript()()
    finally:
        sys.argv = saved_argv
    if rc:
        print(f"[fail] CRDS sync exited with status {rc}", file=sys.stderr)
        sys.exit(rc)
    print("[ok]  CRDS sync complete")


def check_stpsf() -> None:
    stpsf_path_env = os.environ.get("STPSF_PATH")
    if not stpsf_path_env:
        print(
            "[fail] STPSF_PATH not set in the shell.\n"
            "       Add to ~/.bashrc:\n"
            "         export STPSF_PATH=/data/npadman/refdata/stpsf-data",
            file=sys.stderr,
        )
        sys.exit(2)
    stpsf_path = Path(stpsf_path_env)
    if (stpsf_path / "WFI").is_dir():
        print(f"[ok]  STPSF data present at {stpsf_path}")
        return
    print(
        f"[todo] STPSF data missing at {stpsf_path}\n"
        "       Download the latest stpsf-data archive from\n"
        "         https://stpsf.readthedocs.io/en/latest/installation.html#data-files\n"
        f"       and unpack so that {stpsf_path}/WFI/ exists."
    )


def main() -> int:
    sync_crds()
    check_stpsf()
    return 0


if __name__ == "__main__":
    sys.exit(main())
