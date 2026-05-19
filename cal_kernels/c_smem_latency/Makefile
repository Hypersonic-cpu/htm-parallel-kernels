REPO_ROOT := $(shell git rev-parse --show-toplevel)
BIN_NAME := smem_bank
REPORT_KIND := smem
SUPPORTED_ARCHES := single cais native perf
# Align ncu shared-memory conflict/wavefront counters with GPGPU-Sim shared-bank stats.
NATIVE_METRICS := l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
NATIVE_REPORT_PATTERNS := smem_bank_conflict(
include $(REPO_ROOT)/cal_kernels/common.mk
