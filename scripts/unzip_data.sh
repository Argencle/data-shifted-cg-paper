#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

while IFS= read -r -d '' archive; do
    unzip -n "$archive" -d "$(dirname -- "$archive")"
    rm -- "$archive"
done < <(find datas -type f -name '*.zip' -print0)
