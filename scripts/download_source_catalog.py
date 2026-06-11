#!/usr/bin/env python
"""Download the pre-built source catalog from GitHub releases.

Backwards-compatible wrapper around the packaged hydrator. Prefer the console
command directly::

    roman-disperser-hydrate --only catalog

This script forwards any extra arguments (e.g. ``--force``) to it::

    python scripts/download_source_catalog.py --force
"""

import sys

from roman_disperser.hydrate import main

if __name__ == "__main__":
    sys.exit(main(["--only", "catalog", *sys.argv[1:]]))
