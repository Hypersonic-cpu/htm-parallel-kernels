REPO_ROOT := $(shell git rev-parse --show-toplevel)
BIN_NAME := ptr_chase_l2
REPORT_KIND := l2
SUPPORTED_ARCHES := single cais native perf
# Align ncu L2 sector hit/read/write counters with GPGPU-Sim L2 stats.
NATIVE_METRICS := lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum
NATIVE_REPORT_PATTERNS := ptr_chase_l2
include $(REPO_ROOT)/cal_kernels/common.mk
