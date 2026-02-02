#!/bin/bash
# setup_pod.sh

# 1. Handle Pixi Environment
if [ -d ".pixi" ] && [ ! -L ".pixi" ]; then
    echo "📦 Backing up network .pixi and moving to local container disk..."
    mv .pixi .pixi-backup
fi

# Create local target on the fast disk
LOCAL_PIXI="/root/.pixi_envs/$(basename "$PWD")"
mkdir -p "$LOCAL_PIXI"

# Symlink it back so Pixi is 'tricked' into speed
ln -sfn "$LOCAL_PIXI" .pixi

echo "🛠️  Running pixi install (using local NVMe)..."
pixi install

export DATA_ROOT="/root/data"
mkdir -p "$DATA_ROOT/Roman"

echo "🌊 Hydrating data from Backblaze..."
# Restore synphot
restic restore latest --tag synphot-data --target "$DATA_ROOT"
# Restore stpsf-data to the Roman subfolder
restic restore latest --tag stpsf-data --target "$DATA_ROOT/Roman"


