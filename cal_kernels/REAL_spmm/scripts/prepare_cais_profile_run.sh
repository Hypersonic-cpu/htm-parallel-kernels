#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <run_root> <run_arg> <profile>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ROOT="$1"
RUN_ARG="$2"
PROFILE="$3"

case "${PROFILE}" in
  mu4rq16)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 4\n-gpgpu_n_ldst_response_buffer_size 16\n'
    ;;
  tuneA)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 8\n-gpgpu_n_ldst_response_buffer_size 32\n-gpgpu_max_insn_issue_per_warp 2\n-gpgpu_operand_collector_num_units_mem 4\n-gpgpu_operand_collector_num_in_ports_mem 2\n-gpgpu_operand_collector_num_out_ports_mem 2\n'
    ;;
  tuneB)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 16\n-gpgpu_n_ldst_response_buffer_size 64\n-gpgpu_max_insn_issue_per_warp 2\n-gpgpu_operand_collector_num_units_mem 8\n-gpgpu_operand_collector_num_in_ports_mem 4\n-gpgpu_operand_collector_num_out_ports_mem 4\n-gpgpu_reg_file_port_throughput 4\n-gpgpu_l1_banks 8\n'
    ;;
  tuneC)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 16\n-gpgpu_n_ldst_response_buffer_size 64\n-gpgpu_max_insn_issue_per_warp 2\n-gpgpu_operand_collector_num_units_mem 8\n-gpgpu_operand_collector_num_in_ports_mem 4\n-gpgpu_operand_collector_num_out_ports_mem 4\n-gpgpu_reg_file_port_throughput 4\n-gpgpu_l1_banks 16\n'
    ;;
  tuneD)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 16\n-gpgpu_n_ldst_response_buffer_size 64\n-gpgpu_max_insn_issue_per_warp 2\n-gpgpu_operand_collector_num_units_mem 8\n-gpgpu_operand_collector_num_in_ports_mem 4\n-gpgpu_operand_collector_num_out_ports_mem 4\n-gpgpu_reg_file_port_throughput 4\n-gpgpu_l1_banks 32\n'
    ;;
  tuneE)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 32\n-gpgpu_n_ldst_response_buffer_size 128\n-gpgpu_max_insn_issue_per_warp 2\n-gpgpu_operand_collector_num_units_mem 16\n-gpgpu_operand_collector_num_in_ports_mem 8\n-gpgpu_operand_collector_num_out_ports_mem 8\n-gpgpu_reg_file_port_throughput 8\n'
    ;;
  tuneF)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 64\n-gpgpu_n_ldst_response_buffer_size 256\n-gpgpu_max_insn_issue_per_warp 2\n-gpgpu_operand_collector_num_units_mem 32\n-gpgpu_operand_collector_num_in_ports_mem 16\n-gpgpu_operand_collector_num_out_ports_mem 16\n-gpgpu_reg_file_port_throughput 8\n'
    ;;
  tuneG)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 64\n-gpgpu_n_ldst_response_buffer_size 512\n-gpgpu_max_insn_issue_per_warp 2\n-gpgpu_operand_collector_num_units_mem 32\n-gpgpu_operand_collector_num_in_ports_mem 16\n-gpgpu_operand_collector_num_out_ports_mem 16\n-gpgpu_reg_file_port_throughput 8\n'
    ;;
  tuneH)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 128\n-gpgpu_n_ldst_response_buffer_size 512\n-gpgpu_max_insn_issue_per_warp 4\n-gpgpu_operand_collector_num_units_mem 64\n-gpgpu_operand_collector_num_in_ports_mem 32\n-gpgpu_operand_collector_num_out_ports_mem 32\n-gpgpu_reg_file_port_throughput 16\n'
    ;;
  tuneI)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 128\n-gpgpu_n_ldst_response_buffer_size 1024\n-gpgpu_max_insn_issue_per_warp 4\n-gpgpu_operand_collector_num_units_mem 64\n-gpgpu_operand_collector_num_in_ports_mem 32\n-gpgpu_operand_collector_num_out_ports_mem 32\n-gpgpu_reg_file_port_throughput 16\n'
    ;;
  tuneJ)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 256\n-gpgpu_n_ldst_response_buffer_size 1024\n-gpgpu_max_insn_issue_per_warp 4\n-gpgpu_operand_collector_num_units_mem 64\n-gpgpu_operand_collector_num_in_ports_mem 32\n-gpgpu_operand_collector_num_out_ports_mem 32\n-gpgpu_reg_file_port_throughput 16\n'
    ;;
  bypassA)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 8\n-gpgpu_n_ldst_response_buffer_size 32\n-gpgpu_max_insn_issue_per_warp 2\n-gpgpu_operand_collector_num_units_mem 4\n-gpgpu_operand_collector_num_in_ports_mem 2\n-gpgpu_operand_collector_num_out_ports_mem 2\n-gpgpu_reg_file_port_throughput 4\n-gpgpu_gmem_skip_L1D 1\n'
    ;;
  bypassB)
    SIM_CONFIG_APPEND=$'-gpgpu_mem_unit_ports 16\n-gpgpu_n_ldst_response_buffer_size 64\n-gpgpu_max_insn_issue_per_warp 2\n-gpgpu_operand_collector_num_units_mem 8\n-gpgpu_operand_collector_num_in_ports_mem 4\n-gpgpu_operand_collector_num_out_ports_mem 4\n-gpgpu_reg_file_port_throughput 4\n-gpgpu_gmem_skip_L1D 1\n'
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    exit 2
    ;;
esac

cd "${KERNEL_DIR}"
export SIM_CONFIG_APPEND
make ARCH=cais CONF=GH100 RUN_ROOT="${RUN_ROOT}" RUN_ARGS="${RUN_ARG}" stage_run
