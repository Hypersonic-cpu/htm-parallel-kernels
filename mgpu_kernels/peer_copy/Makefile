REPO_ROOT := $(shell git rev-parse --show-toplevel)
BIN_NAME := peer_copy
REPORT_KIND := generic
RUN_TIMEOUT := 5m
SUPPORTED_ARCHES := cais native
SIM_NUM_DEVICES ?= $(or $(firstword $(RUN_ARGS)),1)
include $(REPO_ROOT)/cal_kernels/common.mk
