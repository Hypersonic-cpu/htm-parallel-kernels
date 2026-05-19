REPO_ROOT := $(shell git rev-parse --show-toplevel)
BIN_NAME := val_fp_op
REPORT_KIND := dep_chain
SUPPORTED_ARCHES := single cais native perf
NATIVE_KERNEL_NAME := dep_chain_fp_measure_.*
NATIVE_METRICS := sm__inst_executed.sum,sm__cycles_elapsed.avg,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
NATIVE_REPORT_PATTERNS := dep_chain_fp_measure(
include $(REPO_ROOT)/cal_kernels/common.mk
