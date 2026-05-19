REPO_ROOT := $(shell git rev-parse --show-toplevel)
BIN_NAME := val_l2_parallel
REPORT_KIND := l2_parallel
SUPPORTED_ARCHES := single cais native perf
NATIVE_METRICS := lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,gpu__time_duration.sum
NATIVE_REPORT_PATTERNS := probe_kernel_measure
include $(REPO_ROOT)/cal_kernels/common.mk
