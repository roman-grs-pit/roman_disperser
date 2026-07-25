#!/usr/bin/env python
"""Download pre-generated PSF caches from GitHub releases.

Backwards-compatible wrapper around the packaged hydrator. Prefer the console
command directly::

    roman-disperser-hydrate --only psf            # grism, all 18 SCAs x 2 orders
    roman-disperser-hydrate --only psf --sca 1 2  # just a couple of SCAs
    roman-disperser-hydrate --only psf_prism      # prism, all 18 SCAs

This script forwards any extra arguments (e.g. ``--force``, ``--sca``) to it::

    python scripts/download_psf_caches.py --force
    python scripts/download_psf_caches.py --mode prism

Grism and prism caches share the ``psf_cache/`` directory -- their filenames
carry the STPSF filter name, so they cannot collide.

NOTE (2026-07-25): the prism PSF release does not exist yet. The prism caches
are currently staged by hand into the shared reference-data directory on the
cluster (see its ``PROVENANCE.md``); ``--mode prism`` will 404 until
``psf-prism-v1`` is published.
"""

import sys

from roman_disperser.hydrate import main

_ASSET = {"grism": "psf", "prism": "psf_prism"}


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    mode = "grism"
    if "--mode" in argv:
        i = argv.index("--mode")
        mode = argv[i + 1]
        del argv[i:i + 2]
    if mode not in _ASSET:
        sys.exit(f"unknown mode {mode!r}; choose from {sorted(_ASSET)}")
    sys.exit(main(["--only", _ASSET[mode], *argv]))
