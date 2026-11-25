#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash run.sh --legacy [physical_GPU_ID]'
    echo 'Options: IMAGE RESULT_DIR CACHE_DIR DRY_RUN=1'
    echo 'Runs the original nine-candidate, two-phase HVAT benchmark with historical parameters.'
    exit 0
fi
gpu="${1:-0}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'Select one numeric physical GPU ID.' >&2; exit 2; }
image="${IMAGE:-misyangekkk9/hvatgemmbase:rocm10.0.0_py3.14_pytorch2.12.0_aiter0.1.22_gfx1201-20261103}"
result="$(realpath -m "${RESULT_DIR:-$root/results/gemm8192-$(date -u +%Y%m%dT%H%M%SZ)-$$}")"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ ! -e "$result" && ! -e "$result.tar.gz" ]] || { echo 'Use a new RESULT_DIR.' >&2; exit 2; }
    (cd "$root" && sha256sum --quiet -c SHA256SUMS)
fi
finish() {
    local status=$?
    trap - EXIT
    if [[ -d "$result" ]]; then
        printf '%s\n' "$status" > "$result/exit-status.txt"
        tar -C "$(dirname "$result")" -czf "$result.tar.gz" -- "$(basename "$result")"
        printf 'Results: %s\nArchive: %s.tar.gz\nExit: %s\n' "$result" "$result" "$status"
    fi
    exit "$status"
}
[[ "${DRY_RUN:-0}" == 1 ]] || trap finish EXIT
source "$root/vendor/common/run_in_image.sh"
TUNE_IMAGE_NETWORK=none
TUNE_IMAGE_ARCHIVE=0
tune_image_prepare "$image" "$gpu" "$result" "$root" /repro python3
cache_key="$(printf '%s:image-wheel' "$TUNE_IMAGE_ID" | sha256sum | cut -c1-16)"
cache="$(realpath -m "${CACHE_DIR:-$root/.cache/$cache_key}")"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "$cache/tmp"
    cp "$root/PROVENANCE.json" "$result/package-provenance.json"
    cp "$root/SHA256SUMS" "$result/package-sha256sums.txt"
fi
TUNE_IMAGE_COMMAND+=(
    --mount "type=bind,src=$cache,dst=/cache"
    --env "EXPERIMENT_IMAGE=$image" --env "EXPERIMENT_IMAGE_ID=$TUNE_IMAGE_ID"
    --env AITER_MODE=image --env PYTHONPATH= --env AITER_SOURCE_ROOT= --env AITER_SOURCE_REVISION=
    --env EXPERIMENT_AITER_REVISION=fc1a037a6fb90235f6c947226e781b2a3fea685b
    --env EXPERIMENT_AITER_VERSION=0.1.22+amd.gfx1201.r9600.rocm10.gfc1a037a6f
    --env PYTORCH_TUNABLEOP_ENABLED=1 --env PYTORCH_TUNABLEOP_TUNING=0
    --env AITER_JIT_DIR=/cache/aiter --env AITER_ROOT_DIR=/cache/aiter-root
    --env TRITON_CACHE_DIR=/cache/triton --env XDG_CACHE_HOME=/cache/xdg --env TMPDIR=/cache/tmp
)
status=0
tune_image_run -B /repro/vendor/gemm8192/benchmark_8192_gemm_gfx1201.py \
    --gpu 0 --output-dir /reports/benchmark \
    --tune-warmup 1 --tune-samples 5 --torch-tune-duration-ms 100 --torch-tune-iterations 100 \
    --warmup 5 --samples 11 --repeats 3 --max-nrmse 0.02 || status=$?
if [[ "${DRY_RUN:-0}" != 1 && -f "$result/benchmark/results.json" ]]; then
    docker run --rm --pull=never --read-only --network=none --user "$(id -u):$(id -g)" \
        --mount "type=bind,src=$root,dst=/repro,readonly" \
        --mount "type=bind,src=$result,dst=/reports" --entrypoint python3 "$TUNE_IMAGE_ID" \
        -B /repro/verify_run.py /reports/benchmark/results.json \
        --reference /repro/reference/results.json --output /reports/reproduction-check.json || status=$?
fi
exit "$status"
