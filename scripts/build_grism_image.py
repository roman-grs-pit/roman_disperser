#!/usr/bin/env python
"""Deprecated name for ``build_dispersed_image.py`` — grism specialization.

The pipeline script was renamed when prism support landed: it now simulates
either dispersing element, with the grism still the default. This wrapper
keeps every existing invocation working unchanged (same CLI, same defaults,
same outputs) while emitting a FutureWarning; switch drivers to::

    pixi run -e cuda python scripts/build_dispersed_image.py ...

and this file will be removed in a later release.
"""

import sys
import warnings

warnings.warn(
    "scripts/build_grism_image.py is deprecated; it now forwards to "
    "scripts/build_dispersed_image.py (grism remains the default element). "
    "Update invocations to the new name.",
    FutureWarning,
    stacklevel=2,
)

# Same directory, so this resolves when run as a script or imported from
# scripts/. Re-export everything so `from build_grism_image import ...`
# keeps working for existing helpers/notebooks.
from build_dispersed_image import *          # noqa: F401,F403
from build_dispersed_image import main       # noqa: F401
from build_dispersed_image import build_dispersed_image as build_grism_image  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
