#!/usr/bin/env python
"""Download pre-generated PSF caches from GitHub releases.

Backwards-compatible wrapper around the packaged hydrator. Prefer the console
command directly::

    roman-disperser-hydrate --only psf            # grism, all 36 SCAs (~4.3 GB)
    roman-disperser-hydrate --only psf --sca 1 2  # just a couple of SCAs
    roman-disperser-hydrate --only psf_prism      # prism (psf-prism-v1, ~2 GB)

This script forwards any extra arguments (e.g. ``--force``, ``--sca``) to it;
``--element prism`` selects the prism caches::

    python scripts/download_psf_caches.py --force
    python scripts/download_psf_caches.py --element prism
"""

import sys

from roman_disperser.hydrate import main

if __name__ == "__main__":
    args = sys.argv[1:]
    asset = "psf"
    if "--element" in args:
        i = args.index("--element")
        element = args[i + 1]
        del args[i:i + 2]
        if element not in ("grism", "prism"):
            sys.exit(f"unknown element {element!r}; expected grism or prism")
        asset = "psf" if element == "grism" else "psf_prism"
    sys.exit(main(["--only", asset, *args]))
