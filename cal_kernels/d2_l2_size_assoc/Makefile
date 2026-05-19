REPO_ROOT := $(shell git rev-parse --show-toplevel)
BIN_NAME := l2_capacity_sweep
REPORT_KIND := l2
SUPPORTED_ARCHES := single cais native perf
NATIVE_METRICS := lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum
NATIVE_REPORT_PATTERNS := probe_kernel(
include $(REPO_ROOT)/cal_kernels/common.mk
