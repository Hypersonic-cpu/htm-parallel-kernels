#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <run_root> <run_arg> <profile> [log_name]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ROOT="$1"
RUN_ARG="$2"
PROFILE="$3"
LOG_NAME="${4:-${RUN_ROOT}.launch.log}"
LOG_PATH="${KERNEL_DIR}/${LOG_NAME}"

mkdir -p "$(dirname "${LOG_PATH}")"

nohup bash -lc "
  set -euo pipefail
  cd '${KERNEL_DIR}'
  echo \"[submit] cwd=\$(pwd)\"
  echo \"[submit] run_root=${RUN_ROOT} run_arg=${RUN_ARG} profile=${PROFILE}\"
  echo \"[submit] start=\$(date '+%F %T %z')\"
  exec '${SCRIPT_DIR}/launch_cais_no_timeout.sh' '${RUN_ROOT}' '${RUN_ARG}' '${PROFILE}'
" >"${LOG_PATH}" 2>&1 < /dev/null &

echo "$! ${LOG_PATH}"
