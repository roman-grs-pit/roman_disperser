#!/usr/bin/env python
"""Download pre-generated PSF caches from GitHub releases.

Backwards-compatible wrapper around the packaged hydrator. Prefer the console
command directly::

    roman-disperser-hydrate --only psf            # all 36 SCAs (~4.3 GB)
    roman-disperser-hydrate --only psf --sca 1 2  # just a couple of SCAs

This script forwards any extra arguments (e.g. ``--force``, ``--sca``) to it::

    python scripts/download_psf_caches.py --force
"""

import sys

from roman_disperser.hydrate import main

if __name__ == "__main__":
    sys.exit(main(["--only", "psf", *sys.argv[1:]]))
