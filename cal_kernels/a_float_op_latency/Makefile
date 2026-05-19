REPO_ROOT := $(shell git rev-parse --show-toplevel)
BIN_NAME := dep_chain
REPORT_KIND := dep_chain
SUPPORTED_ARCHES := single cais native perf
NATIVE_KERNEL_NAME := dep_chain_.*
# Align ncu instruction/cycle counters with GPGPU-Sim gpu_ipc/gpu_sim_insn/gpu_sim_cycle.
NATIVE_METRICS := sm__inst_executed.sum,sm__cycles_elapsed.avg,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum
NATIVE_REPORT_PATTERNS := dep_chain_fp(,dep_chain_int(
include $(REPO_ROOT)/cal_kernels/common.mk
