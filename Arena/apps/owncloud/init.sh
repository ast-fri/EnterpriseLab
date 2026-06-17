#!/bin/bash
set -e

echo "Preparing OwnCloud with pre-filled data..."

# Note: For OwnCloud in docker-compose, the data will be populated
# when the container starts. This script prepares the data structure.

# Copy pre-filled data if it exists
if [ -d "/mnt/data/prefill_files" ]; then
    echo "Pre-filled data found, will be copied to /mnt/data/files on container start"
    # The data is already in the image at /mnt/data/prefill_files
    # It will be copied to the actual location in the entrypoint script override
else
    echo "No pre-filled data found"
fi

echo "OwnCloud image preparation complete!"
