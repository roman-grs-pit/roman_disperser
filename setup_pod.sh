#!/bin/bash
# setup_pod.sh

# Start global timer
START_TOTAL=$(date +%s)

# 1. Handle Pixi Environment
START_PIXI=$(date +%s)
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
END_PIXI=$(date +%s)

# 2. Hydrate data
START_DATA=$(date +%s)
export DATA_ROOT="/root/data"
mkdir -p "$DATA_ROOT/Roman"

# List of archives to look for in /workspace/scratch/
# Format: "archive_name:target_directory"
ARCHIVES=("synphot:$DATA_ROOT" "stpsf-data:$DATA_ROOT/Roman")

for entry in "${ARCHIVES[@]}"; do
    archive="${entry%%:*}"
    target="${entry#*:}"
    
    if [ -f "/workspace/scratch/$archive.tar" ]; then
        echo -n "  -> Extracting $archive to $target... "
        mkdir -p "$target"
        # Silent extraction
        tar -xf "/workspace/scratch/$archive.tar" -C "$target/"
        echo "Done."
    else
        echo "  ⚠️  Warning: /workspace/scratch/$archive.tar not found!"
    fi
done
END_DATA=$(date +%s)

# 99. Output timing stats
END_TOTAL=$(date +%s)

# Calculate durations
DURATION_PIXI=$((END_PIXI - START_PIXI))
DURATION_DATA=$((END_DATA - START_DATA))
DURATION_TOTAL=$((END_TOTAL - START_TOTAL))

echo -e "\n------------------------------------"
echo "⏱️  Timing Summary:"
echo "  - Pixi Install:  ${DURATION_PIXI}s"
echo "  - Data Hydration:  ${DURATION_DATA}s"
echo "  - Total Setup:   ${DURATION_TOTAL}s"
echo "------------------------------------"

