# Copilot instructions for `htm-parallel-kernels`

## Build, run, and report commands

This repository is benchmark-driven; there is no separate unit-test or lint framework at the repo root.

| Task | Command |
| --- | --- |
| Build one workload | `make -C cal_kernels/a_float_op_latency ARCH=native CONF=GV100 all` |
| Run one workload (single-benchmark smoke test) | `make -C cal_kernels/a_float_op_latency ARCH=native CONF=GV100 run` |
| Profile one workload with Nsight Compute and emit report CSV | `make -C cal_kernels/f_hbm_bank_parallel ARCH=perf CONF=GH100 run report` |
| Run simulator mode (single) | `make -C cal_kernels/REAL_spmm ARCH=single CONF=GV100 run report` |
| Run simulator mode (CAIS) | `make -C cal_kernels/REAL_spmm ARCH=cais CONF=GV100 run report` |
| Run H100 suite serially | `./cal_kernels/RUN_H100.sh` |
| Run one kernel through H100 runner | `ARCH=perf ./cal_kernels/RUN_H100.sh REAL_reduce` |
| Run one multi-GPU benchmark | `make -C mgpu_kernels/all_gather ARCH=native CONF=GV100 RUN_ARGS='4 1024' run` |
| Pack run artifacts | `python3 cal_kernels/pack_runs.py packed-runs` |

**Required flags and defaults:**
- `ARCH` is required by `cal_kernels/common.mk` (`single`, `cais`, `native`, `perf`).
- `CONF` defaults to `GV100`; valid values are `GV100` and `GH100`.
- Per-benchmark `SUPPORTED_ARCHES` can narrow allowed `ARCH` values.

## High-level architecture

- The repo is organized into two benchmark families:
  - `cal_kernels/`: single-GPU calibration + validation + REAL workloads.
  - `mgpu_kernels/`: multi-GPU real-CUDA validation workloads.
- Each benchmark directory is intentionally thin (`Makefile` + `src/*.cu`) and delegates orchestration to `cal_kernels/common.mk`.
- `cal_kernels/common.mk` is the control plane for:
  - toolchain selection (CUDA 12 for native/perf, CUDA 9 + GPGPU-Sim setup for single/cais),
  - architecture macro wiring (`HW_V100` / `HW_H100`, `sm_70` / `sm_90`),
  - staging into `run-<CONF>/<arch>/`,
  - execution (`run`) and post-processing (`report`),
  - PTX/SASS dump generation for native/perf.
- `cal_kernels/common/report.py` parses either Nsight CSV or GPGPU-Sim `stats.txt` and emits `report.csv` with:
  - first row = columns,
  - second row = statistic type metadata (`per_kernel`, `cumulative`, `app_per_kernel`, `text`),
  - remaining rows = data.

## Key repository-specific conventions

- Keep per-workload Makefiles declarative: set `BIN_NAME`, `REPORT_KIND`, `SUPPORTED_ARCHES`, `NATIVE_METRICS`/`NATIVE_REPORT_PATTERNS` as needed, then `include $(REPO_ROOT)/cal_kernels/common.mk`.
- Do not reintroduce legacy `run/<arch>` paths; outputs must stay under `run-<CONF>/<arch>`.
- CUDA sources are expected to enforce hardware macro selection at compile time:
  - `#if defined(HW_H100) ... #elif defined(HW_V100) ... #else #error ... #endif`.
- Workload classes have different invariants:
  - lower-case microbenchmarks prioritize native measurement quality and may shrink simulator work size under `GPGPU_SIM`,
  - `VAL_*` workloads preserve access pattern while splitting warm/measure launches,
  - `REAL_*` workloads keep native/simulator problem sizes aligned; only repeat/iteration counts are reduced for simulator.
- HBM-facing kernels (`e_hbm_hit_latency`, `f_hbm_bank_parallel`, `VAL_hbm_parallel`, `REAL_*` HBM-facing workloads) follow cold-run defaults:
  - warm launches default to `0`,
  - native repeat defaults to `10`,
  - perf/simulator repeat defaults to `1`,
  - perf mode uses Nsight cache control (`--cache-control all`) in benchmark Makefiles where enabled.
- For CAIS sweeps, patch staged configs via `SIM_CONFIG_SED` / `SIM_CONFIG_APPEND` instead of editing checked-in simulator configs.
- App-side CSV-style rows parsed from `stats.txt` are exported with `APP_` prefixes to keep them distinct from simulator counters.
- `mgpu_kernels/*` follow host CUDA semantics strictly (`cudaSetDevice`, per-device allocations, per-device launches); in CAIS runs, `SIM_NUM_DEVICES` is derived from the first `RUN_ARGS` field.
