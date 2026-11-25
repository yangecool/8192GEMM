#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --legacy ]]; then
    shift
    exec bash "$root/run_legacy.sh" "$@"
fi
exec python3 "$root/rapt/launch.py" "$@"
