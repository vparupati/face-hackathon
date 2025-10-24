#!/usr/bin/env bash
set -euo pipefail
bash scripts/fetch_orl.sh
bash scripts/fetch_yale.sh
bash scripts/fetch_fer2013.sh
echo "All datasets fetched and aligned."
