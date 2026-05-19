REPO_ROOT := $(shell git rev-parse --show-toplevel)
BIN_NAME := ptr_chase
REPORT_KIND := l1
SUPPORTED_ARCHES := single cais native perf
# Align ncu sector hit/read counters with GPGPU-Sim L1/L2 access stats.
NATIVE_METRICS := l1tex__t_sector_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_op_read.sum
NATIVE_REPORT_PATTERNS := ptr_chase_l1
include $(REPO_ROOT)/cal_kernels/common.mk
