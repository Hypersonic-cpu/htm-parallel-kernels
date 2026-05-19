REPO_ROOT := $(shell git rev-parse --show-toplevel)
BIN_NAME := hbm_miss_latency
REPORT_KIND := hbm
SUPPORTED_ARCHES := single cais native perf
NATIVE_METRICS := dram__bytes_read.sum,dram__bytes_write.sum,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,gpu__time_duration.sum,lts__t_sector_hit_rate.pct
NATIVE_REPORT_PATTERNS := probe_kernel(
ifeq ($(ARCH),perf)
NCU_RUN_ARGS := 
# --cache-control all
endif
include $(REPO_ROOT)/cal_kernels/common.mk
