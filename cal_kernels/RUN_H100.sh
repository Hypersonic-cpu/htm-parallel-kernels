#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARCH="${ARCH:-native}"
CONF="${CONF:-GH100}"
GPU="${GPU:-0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-}"
MAKE_JOBS="${MAKE_JOBS:-1}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${SCRIPT_DIR}/logs/h100/$(date +%Y%m%d-%H%M%S)}"

KERNELS=(
  a_float_op_latency
  b_l1_hit_latency
  b2_l1_size_sweep
  b3_l1_warp_parallel
  c_smem_latency
  d_l2_hit_latency
  d2_l2_size_assoc
  d3_l2_bank_parallel
  e_hbm_hit_latency
  f_hbm_bank_parallel
  REAL_reduce
  REAL_spmm
  REAL_gather
  REAL_scatter
  REAL_softmax
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [kernel_dir ...]

Runs cal_kernels workloads serially on one H100-visible GPU.

Environment:
  ARCH=native|perf       Default: native
  CONF=GH100             Default: GH100
  GPU=0                  Used when CUDA_VISIBLE_DEVICES is unset
  MONITOR_GPU=id         nvidia-smi id to monitor; default: first visible GPU
  RUN_TIMEOUT=seconds    Optional per-workload timeout forwarded to make
  MAKE_JOBS=N            Default: 1
  RUN_LOG_DIR=path       Default: cal_kernels/logs/h100/<timestamp>

Examples:
  ./cal_kernels/RUN_H100.sh
  ARCH=perf ./cal_kernels/RUN_H100.sh REAL_reduce REAL_softmax
  GPU=1 RUN_TIMEOUT=300 ./cal_kernels/RUN_H100.sh
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

if [[ "${ARCH}" != "native" && "${ARCH}" != "perf" ]]; then
  echo "ERROR: RUN_H100.sh supports ARCH=native or ARCH=perf, got ARCH=${ARCH}" >&2
  exit 2
fi

if [[ "${CONF}" != "GH100" ]]; then
  echo "ERROR: RUN_H100.sh is for H100 runs; use CONF=GH100" >&2
  exit 2
fi

if [[ $# -gt 0 ]]; then
  KERNELS=("$@")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
MONITOR_GPU="${MONITOR_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"
mkdir -p "${RUN_LOG_DIR}"

STATUS_LOG="${RUN_LOG_DIR}/status.log"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*" | tee -a "${STATUS_LOG}"
}

gpu_snapshot() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --id="${MONITOR_GPU}" --query-gpu=timestamp,index,name,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
  else
    echo "nvidia-smi not found"
  fi
}

monitor_gpu() {
  local out="$1"
  while true; do
    {
      printf '\n[%(%Y-%m-%dT%H:%M:%S%z)T]\n' -1
      gpu_snapshot
    } >>"${out}" 2>&1
    sleep 5
  done
}

run_kernel() {
  local kernel="$1"
  local kernel_dir="${SCRIPT_DIR}/${kernel}"
  local safe_kernel="${kernel//\//_}"
  local run_log="${RUN_LOG_DIR}/${safe_kernel}.log"
  local gpu_log="${RUN_LOG_DIR}/${safe_kernel}.gpu.log"
  local timeout_args=()

  if [[ ! -f "${kernel_dir}/Makefile" ]]; then
    log "FAIL ${kernel}: missing ${kernel_dir}/Makefile"
    return 1
  fi

  if [[ -n "${RUN_TIMEOUT}" ]]; then
    timeout_args=("RUN_TIMEOUT=${RUN_TIMEOUT}")
  fi

  log "START ${kernel} ARCH=${ARCH} CONF=${CONF} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  gpu_snapshot | tee -a "${gpu_log}" >/dev/null

  monitor_gpu "${gpu_log}" &
  local monitor_pid=$!
  set +e
  make -C "${kernel_dir}" -j"${MAKE_JOBS}" ARCH="${ARCH}" CONF="${CONF}" "${timeout_args[@]}" run report 2>&1 | tee "${run_log}"
  local rc=${PIPESTATUS[0]}
  set -e
  kill "${monitor_pid}" >/dev/null 2>&1 || true
  wait "${monitor_pid}" >/dev/null 2>&1 || true

  if [[ ${rc} -eq 0 ]]; then
    log "PASS ${kernel}"
  else
    log "FAIL ${kernel} rc=${rc}"
  fi
  return "${rc}"
}

log "H100 serial run started: ARCH=${ARCH} CONF=${CONF} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MONITOR_GPU=${MONITOR_GPU}"
log "Logs: ${RUN_LOG_DIR}"

failures=0
for kernel in "${KERNELS[@]}"; do
  if ! run_kernel "${kernel}"; then
    failures=$((failures + 1))
  fi
done

if [[ ${failures} -eq 0 ]]; then
  log "DONE all ${#KERNELS[@]} kernels passed"
else
  log "DONE ${failures}/${#KERNELS[@]} kernels failed"
fi

exit "${failures}"
