#!/usr/bin/env bash
set -euo pipefail
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root_dir"
output_root=${1:-.work/sanitizers}
san_bin=${COMPUTE_SANITIZER:-/usr/local/cuda-13.2/bin/compute-sanitizer}
mkdir -p "$output_root"
export PYTHONPATH=src TRITON_INTERPRET=0 TRITON_CACHE_DIR="$root_dir/.work/gpu-cache"
for check in memcheck racecheck synccheck; do
  "$san_bin" --tool "$check" --error-exitcode 86 .venv/bin/python -m gpu_qualification.execute \
    --mode gpu --output "$output_root/$check" > "$output_root/$check.log" 2>&1
  tail -3 "$output_root/$check.log"
done
