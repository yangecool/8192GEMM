#!/usr/bin/env bash
# Shared existing-image runtime for standalone GPU probes. Source from launchers.
# tune_image_prepare IMAGE GPU_IDS RESULT_DIR SOURCE_DIR CONTAINER_DIR PYTHON
# Then append workload-only Docker options to TUNE_IMAGE_COMMAND if needed,
# and invoke tune_image_run SCRIPT [ARGS...]. The image ID is inserted by the helper.
# TUNE_IMAGE_ARCHIVE=0 leaves archiving to a launcher that must first stop writers.

tune_image_prepare() {
    local image="$1" gpu_ids="$2" source_dir="$4" container_dir="$5" python="$6"
    local device
    TUNE_IMAGE_RESULT_DIR="$(realpath -m "$3")" || return
    TUNE_IMAGE_ID="$image"
    [[ "$gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'GPU_IDS must be 0 or a list such as 0,1' >&2; return 2; }
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        [[ ! -e "$TUNE_IMAGE_RESULT_DIR" && ! -e "${TUNE_IMAGE_RESULT_DIR}.tar.gz" ]] || {
            echo "RESULT_DIR and bundle must be new: $TUNE_IMAGE_RESULT_DIR" >&2; return 2;
        }
        mkdir -p "$TUNE_IMAGE_RESULT_DIR" || return
        TUNE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$image")" || return
        docker image inspect "$TUNE_IMAGE_ID" > "${TUNE_IMAGE_RESULT_DIR}/image-inspect.json" || return
    fi
    TUNE_IMAGE_COMMAND=(docker run --rm --pull=never --read-only "--network=${TUNE_IMAGE_NETWORK:-none}"
        --device=/dev/kfd --device=/dev/dri
        --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,exec,nosuid,size=2g
        --env "HIP_VISIBLE_DEVICES=$gpu_ids"
        --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1
        --env TRITON_CACHE_DIR=/tmp/triton --env XDG_CACHE_HOME=/tmp/cache
        --mount "type=bind,src=${source_dir},dst=${container_dir},readonly"
        --mount "type=bind,src=${TUNE_IMAGE_RESULT_DIR},dst=/reports"
        --workdir /tmp --entrypoint "$python")
    # Host/image group names may differ; the device node's numeric group is sufficient.
    for device in /dev/kfd /dev/dri/renderD*; do
        [[ ! -e "$device" ]] || TUNE_IMAGE_COMMAND+=(--group-add "$(stat -c %g "$device")")
    done
    printf 'Image: %s\nGPU_IDS: %s\nReports: %s\n' "$image" "$gpu_ids" "$TUNE_IMAGE_RESULT_DIR"
}

tune_image_run() {
    local status=0
    local -a pipeline_status
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${TUNE_IMAGE_COMMAND[@]}" "$TUNE_IMAGE_ID" "$@"
        printf '\n'
        return 0
    fi
    if "${TUNE_IMAGE_COMMAND[@]}" "$TUNE_IMAGE_ID" "$@" 2>&1 | tee "${TUNE_IMAGE_RESULT_DIR}/docker.log"; then
        status=0
    else
        pipeline_status=("${PIPESTATUS[@]}")
        status=${pipeline_status[0]}
        if (( status == 0 )); then status=${pipeline_status[1]}; fi
    fi
    if [[ "${TUNE_IMAGE_ARCHIVE:-1}" != 0 ]]; then
        tar -C "$(dirname "$TUNE_IMAGE_RESULT_DIR")" -czf "${TUNE_IMAGE_RESULT_DIR}.tar.gz" \
            -- "$(basename "$TUNE_IMAGE_RESULT_DIR")" || return
        printf 'Summary: %s/summary.json\nBundle: %s.tar.gz\nExit: %s\n' \
            "$TUNE_IMAGE_RESULT_DIR" "$TUNE_IMAGE_RESULT_DIR" "$status"
    else
        printf 'Summary: %s/summary.json\nExit: %s\n' "$TUNE_IMAGE_RESULT_DIR" "$status"
    fi
    return "$status"
}
