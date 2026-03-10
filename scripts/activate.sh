#!/bin/bash
# Set default paths for STPSF and synphot data if not already set.
# Users can override by exporting these variables before running pixi.
export STPSF_PATH="${STPSF_PATH:-$HOME/data/Roman/stpsf-data}"
export PYSYN_CDBS="${PYSYN_CDBS:-$HOME/data/synphot/grp/redcat/trds}"
