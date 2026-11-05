#!/bin/sh
set -eu

# Move to the repository root
CDPATH= cd "$(dirname "$0")/.."

# Extract all ZIP archives and remove them after successful extraction
find datas -type f -name '*.zip' -exec sh -c '
    set -eu
    for archive do
        unzip -n "$archive" -d "$(dirname "$archive")"
        rm "$archive"
    done
' sh {} +