SKIP_ARCH_CHECK := $(if $(filter clean,$(MAKECMDGOALS)),$(if $(filter-out clean,$(MAKECMDGOALS)),,1),)
ifeq ($(SKIP_ARCH_CHECK),)
ARCH ?=
ifeq ($(ARCH),)
$(error ARCH is required. Use ARCH=single, ARCH=cais, ARCH=native, or ARCH=perf)
endif
ifneq ($(filter $(ARCH),single cais native perf),$(ARCH))
$(error ARCH must be one of: single cais native perf)
endif
ifneq ($(SUPPORTED_ARCHES),)
ifeq ($(filter $(ARCH),$(SUPPORTED_ARCHES)),)
$(error ARCH=$(ARCH) not supported in $(CURDIR); supported: $(SUPPORTED_ARCHES))
endif
endif
endif

REPO_ROOT ?= $(shell git rev-parse --show-toplevel)
CONF ?= GV100
VALID_CONFS := GV100 GH100 A100
ifeq ($(filter $(CONF),$(VALID_CONFS)),)
$(error CONF must be one of: $(VALID_CONFS))
endif
CUDA_INSTALL_PATH ?= /usr/local/cuda-12
CUDA9_INSTALL_PATH ?= /usr/local/cuda-9.0
GCC54_SOURCE ?= $(HOME)/tools/opt/source-gcc54
A100_SIM_HOME ?= $(HOME)/htm-workbench/accel-sim-framework/gpgpu-sim
PYTHON ?= python3
BIN_NAME ?= app
RUN_ROOT ?= run-$(ARCH)
RUN_SUBDIR ?= $(CONF)
BUILD_DIR ?= build/$(CONF)/$(ARCH)
RUN_DIR ?= $(RUN_ROOT)/$(RUN_SUBDIR)
PROG ?= $(BUILD_DIR)/$(BIN_NAME)
RUN_BIN ?= $(RUN_DIR)/$(BIN_NAME)
DEFAULT_CONF_NAME := $(if $(filter A100,$(CONF)),SM8_A100,SM7_$(CONF))
CONF_NAME ?= $(DEFAULT_CONF_NAME)
NATIVE_KERNEL_NAME ?= .*
RUN_TIMEOUT ?=
REPORT_KIND ?= generic
RUN_ARGS ?=
NCU_RUN_ARGS ?=
NCU_GROUPS ?= timing_clock,l1_l2,dram,execution,scheduler_stall
NCU_GPU ?= 0
NCU_VISIBLE_DEVICES ?=
NCU_GPC_CLOCK_MHZ ?= 1590
NCU_CACHE_CONTROL ?= $(if $(findstring --cache-control none,$(NCU_RUN_ARGS)),none,all)
NCU_SUDO ?= auto
NCU_GROUP_RUNNER ?= $(REPO_ROOT)/cal_kernels/common/ncu_group_runner.py
NCU_WORKFLOW_VERSION ?= 2
SIM_NUM_DEVICES ?=
SIM_CONFIG_SED ?=
SIM_CONFIG_APPEND ?=
SIM_LIBCUDART_SO ?=
NATIVE_METRICS ?=
NATIVE_REPORT_PATTERNS ?=
SIM_STATS ?= $(RUN_DIR)/stats.txt
NATIVE_CSV ?= $(RUN_DIR)/ncu.csv
RUN_DIR_ABS := $(abspath $(RUN_DIR))
NATIVE_CSV_ABS := $(abspath $(NATIVE_CSV))
NCU_STATS_ABS := $(abspath $(RUN_DIR)/stats.txt)
NCU_RUNNER_LOG_ABS := $(abspath $(RUN_DIR)/ncu_runner.log)
NCU_RUNNER_PLAN_LOG_ABS := $(abspath $(RUN_DIR)/ncu_runner_plan.log)
PRETTY_CSV_AWK := awk "BEGIN{seen_csv=0; ht=sprintf(\"%c\",9); dt=ht ht} /,/{if(!seen_csv){gsub(\",\",\",\" ht); seen_csv=1}else{gsub(\",\",\",\" dt)}} {print}"

ifneq ($(filter $(ARCH),native perf),)
CUDA_HOME := $(CUDA_INSTALL_PATH)
else ifeq ($(CONF),A100)
CUDA_HOME := $(CUDA_INSTALL_PATH)
else
CUDA_HOME := $(CUDA9_INSTALL_PATH)
endif

NVCC ?= $(CUDA_HOME)/bin/nvcc
CUOBJDUMP ?= $(CUDA_HOME)/bin/cuobjdump
NCU ?= $(CUDA_INSTALL_PATH)/bin/ncu
NATIVE_PTX_DUMP ?= $(RUN_DIR)/$(BIN_NAME).ptx
NATIVE_SASS_DUMP ?= $(RUN_DIR)/$(BIN_NAME).sass
NATIVE_SM ?= $(if $(filter GH100,$(CONF)),sm_90,$(if $(filter A100,$(CONF)),sm_80,sm_70))
HW_DEFINE := $(if $(filter A100,$(CONF)),-DHW_A100 -DHW_H100,$(if $(filter GH100,$(CONF)),-DHW_H100,-DHW_V100))
SIM_SM ?= $(if $(filter A100,$(CONF)),sm_80,sm_70)
ifneq ($(filter $(ARCH),native perf),)
BUILD_NVCC := $(NVCC)
else ifeq ($(CONF),A100)
BUILD_NVCC := $(if $(NVCC_CCBIN),$(NVCC) -ccbin $(NVCC_CCBIN),$(NVCC))
else
BUILD_NVCC := bash -lc 'set -euo pipefail; source "$(GCC54_SOURCE)"; exec "$(NVCC)" -ccbin "$${NVCC_CCBIN}" "$$@"' --
endif

CXXFLAGS ?= -I $(REPO_ROOT)/test_kernels/include -I $(REPO_ROOT)/cal_kernels/common -Xcompiler -fopenmp --expt-relaxed-constexpr -O3 -lineinfo
CUFLAGS ?= -std=c++14
CUFLAGS += $(HW_DEFINE)
CUFLAGS += -DHTM_CONF_$(CONF)=1
ifneq ($(filter $(ARCH),native perf),)
CUFLAGS += -arch=$(NATIVE_SM)
CUFLAGS += -DHTM_ARCH_NATIVE=1
else
CUFLAGS += --cudart shared -arch=$(SIM_SM) -DGPGPU_SIM
endif
ifeq ($(ARCH),perf)
CUFLAGS += -DPERF_RUN
endif
CXXFLAGS += $(CUFLAGS)
SRCS := $(wildcard src/*.cu)
OBJS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(SRCS))

ifeq ($(ARCH),single)
ifneq ($(CONF),A100)
SIM_HOME := $(REPO_ROOT)/gpgpu-sim-single
SOURCE_FN := source_single
else
SIM_HOME := $(A100_SIM_HOME)
endif
endif
ifeq ($(ARCH),cais)
ifneq ($(CONF),A100)
SIM_HOME := $(REPO_ROOT)/gpgpu-sim-cais
SOURCE_FN := source_cais
else
SIM_HOME := $(A100_SIM_HOME)
endif
endif
ifneq ($(filter single cais,$(ARCH)),)
ifneq ($(CONF),A100)
SRC_CONF := $(SIM_HOME)/configs/tested-cfgs/$(CONF_NAME)
else
SRC_CONF := $(REPO_ROOT)/cal_kernels/configs/tested-cfgs/$(CONF_NAME)
endif
endif

.PHONY: all clean pre_build stage_run run report rmdump show

default: all

$(BUILD_DIR)/%.o: src/%.cu
	@mkdir -p $(BUILD_DIR)
	$(BUILD_NVCC) $(CXXFLAGS) -c $< -o $@

$(PROG): $(OBJS)
	$(BUILD_NVCC) $(CXXFLAGS) $^ -o $@

pre_build:
	@mkdir -p $(BUILD_DIR) $(RUN_DIR)
	@echo TARGET $(PROG)
	@echo ARCH $(ARCH)
	@echo RUN_ARGS $(RUN_ARGS)
	@echo NCU_RUN_ARGS $(NCU_RUN_ARGS)
	@echo SIM_NUM_DEVICES $(SIM_NUM_DEVICES)
	@printf 'SIM_CONFIG_SED %s\n' "$(SIM_CONFIG_SED)"
	@printf 'SIM_CONFIG_APPEND %s\n' "$${SIM_CONFIG_APPEND-}"
	@printf 'SIM_LIBCUDART_SO %s\n' "$(SIM_LIBCUDART_SO)"

all: pre_build $(PROG)

clean:
	@if [ -z "$(ARCH)" ]; then echo "ERROR: ARCH is required for clean; use make clean ARCH=<arch> RUN_ROOT=<path>"; exit 1; fi
	@rm -rf "$(RUN_DIR)"

rmdump:
	@rm -f $(RUN_DIR)/_app_cuda_* $(RUN_DIR)/_cuobjdump_* $(RUN_DIR)/SystemConfig $(RUN_DIR)/KernelConfig $(RUN_DIR)/TB_schedule_* $(RUN_DIR)/data_placement_* $(SIM_STATS) $(NATIVE_CSV) $(RUN_DIR)/nvprof.txt $(RUN_DIR)/report.txt $(RUN_DIR)/report.csv $(RUN_DIR)/ncu_group_summary.csv $(RUN_DIR)/ncu_runner.log $(RUN_DIR)/ncu_runner_plan.log
	@rm -rf $(RUN_DIR)/ncu_groups

show:
	@echo REPO_ROOT=$(REPO_ROOT)
	@echo ARCH=$(ARCH)
	@echo CONF=$(CONF)
	@echo CUDA_INSTALL_PATH=$(CUDA_INSTALL_PATH)
	@echo CUDA9_INSTALL_PATH=$(CUDA9_INSTALL_PATH)
	@echo CUDA_HOME=$(CUDA_HOME)
	@echo NVCC=$(NVCC)
	@echo BUILD_NVCC=$(BUILD_NVCC)
	@echo GCC54_SOURCE=$(GCC54_SOURCE)
	@echo A100_SIM_HOME=$(A100_SIM_HOME)
	@echo NATIVE_SM=$(NATIVE_SM)
	@echo HW_DEFINE=$(HW_DEFINE)
	@echo SIM_SM=$(SIM_SM)
	@echo SIM_HOME=$(SIM_HOME)
	@echo SRC_CONF=$(SRC_CONF)
	@echo PROG=$(PROG)
	@echo RUN_ROOT=$(RUN_ROOT)
	@echo RUN_SUBDIR=$(RUN_SUBDIR)
	@echo RUN_DIR=$(RUN_DIR)
	@echo NCU_GROUPS=$(NCU_GROUPS)
	@echo NCU_GPU=$(NCU_GPU)
	@echo NCU_VISIBLE_DEVICES=$(NCU_VISIBLE_DEVICES)
	@echo NCU_GPC_CLOCK_MHZ=$(NCU_GPC_CLOCK_MHZ)
	@echo NCU_CACHE_CONTROL=$(NCU_CACHE_CONTROL)
	@echo NCU_SUDO=$(NCU_SUDO)
	@echo NCU_WORKFLOW_VERSION=$(NCU_WORKFLOW_VERSION)
	@echo SIM_LIBCUDART_SO=$(SIM_LIBCUDART_SO)

stage_run: pre_build all rmdump
ifneq ($(filter $(ARCH),native perf),)
	@cp -f $(PROG) $(RUN_BIN)
	@"$(CUOBJDUMP)" --dump-ptx "$(RUN_BIN)" > "$(NATIVE_PTX_DUMP)"
	@"$(CUOBJDUMP)" --dump-sass "$(RUN_BIN)" > "$(NATIVE_SASS_DUMP)"
else
	@if [ ! -d "$(SRC_CONF)" ]; then echo "ERROR: missing config $(SRC_CONF)"; exit 1; fi
	@cp -f $(PROG) $(RUN_BIN)
	@cp -f $(SRC_CONF)/* $(RUN_DIR)/
	@if [ -n "$(SIM_NUM_DEVICES)" ]; then cfg="$(RUN_DIR)/gpgpusim.config"; if [ ! -f "$$cfg" ]; then echo "ERROR: missing $$cfg"; exit 1; fi; tmp="$${cfg}.tmp"; sed '/^[[:space:]]*-gpgpu_num_devices[[:space:]]/d' "$$cfg" > "$$tmp"; printf '%s\n' "-gpgpu_num_devices $(SIM_NUM_DEVICES)" >> "$$tmp"; mv "$$tmp" "$$cfg"; fi
	if [ -n "$(SIM_CONFIG_SED)" ]; then cfg="$(RUN_DIR)/gpgpusim.config"; if [ ! -f "$$cfg" ]; then echo "ERROR: missing $$cfg"; exit 1; fi; sed -i $(SIM_CONFIG_SED) "$$cfg"; fi
	@if [ -n "$${SIM_CONFIG_APPEND-}" ]; then cfg="$(RUN_DIR)/gpgpusim.config"; if [ ! -f "$$cfg" ]; then echo "ERROR: missing $$cfg"; exit 1; fi; printf '\n# SIM_CONFIG_APPEND begin\n%s\n# SIM_CONFIG_APPEND end\n' "$${SIM_CONFIG_APPEND}" >> "$$cfg"; fi
endif

run: stage_run
ifeq ($(ARCH),native)
	@bash -lc 'set -euo pipefail; export CUDA_INSTALL_PATH="$(CUDA_INSTALL_PATH)"; export CUDA_HOME="$(CUDA_INSTALL_PATH)"; export LD_LIBRARY_PATH="$(CUDA_INSTALL_PATH)/lib64:$${LD_LIBRARY_PATH-}"; export PATH="$(CUDA_INSTALL_PATH)/bin:$${PATH-}"; cd "$(CURDIR)"; cd "$(RUN_DIR)"; if [ -n "$(RUN_TIMEOUT)" ]; then timeout $(RUN_TIMEOUT) stdbuf -oL -eL ./$(BIN_NAME) $(RUN_ARGS) 2>&1 | tee stats.txt; else stdbuf -oL -eL ./$(BIN_NAME) $(RUN_ARGS) 2>&1 | tee stats.txt; fi'
else ifeq ($(ARCH),perf)
	@bash -lc 'set -euo pipefail; export CUDA_INSTALL_PATH="$(CUDA_INSTALL_PATH)"; export CUDA_HOME="$(CUDA_INSTALL_PATH)"; export LD_LIBRARY_PATH="$(CUDA_INSTALL_PATH)/lib64:$${LD_LIBRARY_PATH-}"; export PATH="$(CUDA_INSTALL_PATH)/bin:$${PATH-}"; cd "$(CURDIR)"; cd "$(RUN_DIR)"; "$(PYTHON)" "$(NCU_GROUP_RUNNER)" --ncu "$(NCU)" --run-dir "$(RUN_DIR_ABS)" --merged-csv "$(NATIVE_CSV_ABS)" --stats-file "$(NCU_STATS_ABS)" --runner-log "$(NCU_RUNNER_PLAN_LOG_ABS)" --binary-name "$(BIN_NAME)" --run-args "$(RUN_ARGS)" --ncu-run-args "$(NCU_RUN_ARGS)" --extra-metrics "$(NATIVE_METRICS)" --kernel-name "$(NATIVE_KERNEL_NAME)" --groups "$(NCU_GROUPS)" --cache-control "$(NCU_CACHE_CONTROL)" --sudo-mode "$(NCU_SUDO)" --gpu-index "$(NCU_GPU)" --visible-devices "$(NCU_VISIBLE_DEVICES)" --target-gpc-mhz "$(NCU_GPC_CLOCK_MHZ)" $(if $(RUN_TIMEOUT),--timeout "$(RUN_TIMEOUT)",) 2>&1 | tee "$(NCU_RUNNER_LOG_ABS)"'
else
	@bash -lc 'set -eo pipefail; export CUDA_INSTALL_PATH="$(CUDA_INSTALL_PATH)"; export CUDA9_INSTALL_PATH="$(CUDA9_INSTALL_PATH)"; export OPENCL_REMOTE_GPU_HOST="${OPENCL_REMOTE_GPU_HOST-}"; set +u; if [ "$(CONF)" = "A100" ]; then export GPGPUSIM_ROOT="$(SIM_HOME)"; source "$(SIM_HOME)/setup_environment" >/dev/null; else source "$(REPO_ROOT)/sourceme" >/dev/null; $(SOURCE_FN) >/dev/null; fi; set -u; expected_libcudart=""; if [ -n "$(SIM_LIBCUDART_SO)" ]; then override_so="$(SIM_LIBCUDART_SO)"; if [ ! -f "$$override_so" ]; then echo "ERROR: missing SIM_LIBCUDART_SO=$$override_so" | tee "$(CURDIR)/$(RUN_DIR)/stats.txt"; exit 1; fi; shim_dir="$(CURDIR)/$(RUN_DIR)/.sim-libcudart"; mkdir -p "$$shim_dir"; rm -f "$$shim_dir"/libcudart.so "$$shim_dir"/libcudart.so.12 "$$shim_dir"/libcudart.so.11 "$$shim_dir"/libcudart.so.9.0; ln -sf "$$override_so" "$$shim_dir/libcudart.so"; ln -sf "$$override_so" "$$shim_dir/libcudart.so.12"; ln -sf "$$override_so" "$$shim_dir/libcudart.so.11"; ln -sf "$$override_so" "$$shim_dir/libcudart.so.9.0"; export LD_LIBRARY_PATH="$$shim_dir:$$(dirname "$$override_so"):$${LD_LIBRARY_PATH-}"; expected_libcudart=$$(readlink -f "$$override_so"); else if [ ! -f "$(SIM_HOME)/lib/$${GPGPUSIM_CONFIG}/libcudart.so" ]; then for fallback_config in gcc-11.4.0/cuda-12040/release gcc-11.4.0/cuda-11000/release gcc-5.4.0/cuda-9000/release gcc-/cuda-9000/release; do if [ -f "$(SIM_HOME)/lib/$${fallback_config}/libcudart.so" ]; then export GPGPUSIM_CONFIG="$${fallback_config}"; export LD_LIBRARY_PATH="$(SIM_HOME)/lib/$${fallback_config}:$${LD_LIBRARY_PATH-}"; break; fi; done; fi; fi; resolved_libcudart=$$(ldd "$(CURDIR)/$(RUN_BIN)" | sed -n "/libcudart\\.so/{s/.*=> //; s/ (.*//; p; q;}"); resolved_real=$$(readlink -f "$${resolved_libcudart:-/nonexistent}" 2>/dev/null || true); using_libcudart="$${resolved_real:-$${resolved_libcudart:-<missing>}}"; if [ -n "$$expected_libcudart" ]; then ok=0; [ "$$resolved_real" = "$$expected_libcudart" ] && ok=1; else ok=0; printf "%s\n" "$$resolved_libcudart" | grep -q "^$(SIM_HOME)/lib/" && ok=1; fi; if [ "$$ok" -ne 1 ]; then { echo "Using libcudart.so -> $$using_libcudart"; echo "ERROR: $(ARCH) run is not using the expected GPGPU-Sim libcudart"; echo "Resolved libcudart: $${resolved_libcudart:-<missing>}"; echo "Resolved realpath: $${resolved_real:-<missing>}"; echo "Expected libcudart: $${expected_libcudart:-$(SIM_HOME)/lib/...}"; echo "GPGPUSIM_CONFIG=$${GPGPUSIM_CONFIG:-<unset>}"; ldd "$(CURDIR)/$(RUN_BIN)"; } 2>&1 | tee "$(CURDIR)/$(RUN_DIR)/stats.txt"; exit 1; fi; cd "$(CURDIR)/$(RUN_DIR)"; { echo "Using libcudart.so -> $$using_libcudart"; echo "Resolved libcudart: $$resolved_libcudart"; echo "Resolved libcudart realpath: $$resolved_real"; echo "Expected libcudart: $${expected_libcudart:-<default from $(SIM_HOME)/lib>}"; echo "GPGPUSIM_CONFIG=$${GPGPUSIM_CONFIG:-<unset>}"; if [ -n "$(RUN_TIMEOUT)" ]; then timeout $(RUN_TIMEOUT) stdbuf -oL -eL ./$(BIN_NAME) $(RUN_ARGS); else stdbuf -oL -eL ./$(BIN_NAME) $(RUN_ARGS); fi; } 2>&1 | tee stats.txt; rc=$$?; tail -n 120 stats.txt; exit $$rc'
endif


# ncu/GPGPU-Sim alignment comments live in each kernel Makefile near REPORT_KIND/NATIVE_METRICS.
report:
ifeq ($(ARCH),native)
	@:
else
	@$(PYTHON) $(REPO_ROOT)/cal_kernels/common/report.py $(if $(filter perf,$(ARCH)),native,gpgpusim) $(REPORT_KIND) $(if $(filter perf,$(ARCH)),$(NATIVE_CSV),$(SIM_STATS)) "$(NATIVE_REPORT_PATTERNS)" > $(RUN_DIR)/report.csv
	@cp $(RUN_DIR)/report.csv $(RUN_DIR)/report.txt
	@$(PYTHON) -c 'import csv, pathlib, sys; rows=list(csv.reader(pathlib.Path(sys.argv[1]).open(newline=""))); print("" if not rows else " ".join(rows[0]));  [print(" ".join((r + [""] * (len(rows[0]) - len(r)))[:len(rows[0])])) for r in rows[1:]]' "$(RUN_DIR)/report.csv"
endif
