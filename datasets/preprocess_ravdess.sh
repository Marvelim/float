#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (one level up from this script's dir)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths (store data at project root to avoid confusion)
RAW_DIR="${RAW_DIR:-$PROJECT_ROOT/ravdess_raw}"
PROCESSED_DIR="${PROCESSED_DIR:-$PROJECT_ROOT/ravdess_processed}"

# Default preprocessing options
ACTORS="${ACTORS:-1}"
MODALITIES="${MODALITIES:-speech}"

# Avoid face-alignment stalls by default; allow override via NO_FACE=0
NO_FACE_FLAG="--no_face"
if [[ "${NO_FACE:-1}" == "0" ]]; then
  NO_FACE_FLAG=""
fi

# Short timeout for face-alignment init if enabled
FA_TIMEOUT="${FA_TIMEOUT:-5}"

# Warn if ffmpeg is missing (merging audio will be skipped in processing step)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[WARN] ffmpeg not found in PATH; video+audio merge will be skipped if required." >&2
fi

set -x
python "$SCRIPT_DIR/download_and_preprocess_ravdess.py" \
  --preprocess \
  --actors "$ACTORS" \
  --modalities "$MODALITIES" \
  --raw_dir "$RAW_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  $NO_FACE_FLAG \
  --fa_init_timeout "$FA_TIMEOUT" \
  "$@"
set +x
