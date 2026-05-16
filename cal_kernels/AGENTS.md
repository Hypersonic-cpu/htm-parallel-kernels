# `cal_kernels`: Hopper Calibration Workloads

This directory is on the `cal-hopper` refactor path. Follow
[HOPPER_REFACTOR_PLAN.md](/home/simuser/htm-gpgpu-sim/cal_kernels/HOPPER_REFACTOR_PLAN.md)
unless the user gives newer instructions.

## Active Layout

- Lower-case microbenchmarks:
  - `a_float_op_latency`
  - `b_l1_hit_latency`
  - `b2_l1_size_sweep`
  - `b3_l1_warp_parallel`
  - `c_smem_latency`
  - `d_l2_hit_latency`
  - `d2_l2_size_assoc`
  - `d3_l2_bank_parallel`
  - `e_hbm_hit_latency`
  - `f_hbm_bank_parallel`
- Validation microbenchmarks:
  - `VAL_fp_op`
  - `VAL_l1_parallel`
  - `VAL_smem_parallel`
  - `VAL_l2_parallel`
  - `VAL_hbm_parallel`
- Real workloads:
  - `REAL_reduce`
  - `REAL_spmm`
  - `REAL_gather`
  - `REAL_scatter`
  - `REAL_softmax`
- Shared infrastructure:
  - `common.mk`
  - `common/report.py`
  - `pack_runs.py`
  - `TUNE_HBM.md`

## Tuning Log

- Keep ongoing HBM calibration notes in `TUNE_HBM.md`.
- Update that file whenever a calibration direction is ruled in or ruled out.
- For GH100 HBM tuning, record both:
  - the exact staged/swept config knobs
  - the resulting worst-case read-BW and total-time errors across the sweep

## CAIS Staging Rule

- CAIS injection helpers were removed from `cal_kernels`:
  - `common/cais_artifacts.py` is deleted.
  - `common.mk` no longer supports `EXTRA_STAGE_CMD` in `stage_run`.
- `ARCH=cais` runs now stage only the benchmark binary plus simulator config files.

## Build And Run Rules

Every invocation must set `ARCH`. `CONF` defaults to `GV100`.

```bash
make -C cal_kernels/a_float_op_latency ARCH=native CONF=GV100 run
make -C cal_kernels/a_float_op_latency ARCH=perf CONF=GV100 run report
make -C cal_kernels/a_float_op_latency ARCH=native CONF=GH100 run
make -C cal_kernels/a_float_op_latency ARCH=perf CONF=GH100 run report
make -C cal_kernels/a_float_op_latency ARCH=single CONF=GV100 run report
make -C cal_kernels/a_float_op_latency ARCH=cais CONF=GV100 run report
```

- `ARCH=native CONF=GV100` uses CUDA 12, `-arch=sm_70`, and runs directly.
- `ARCH=native CONF=GH100` uses CUDA 12, `-arch=sm_90`, and runs directly.
- `ARCH=perf CONF=GV100` uses CUDA 12, `-arch=sm_70`, and profiles with `ncu`.
- `ARCH=perf CONF=GH100` uses CUDA 12, `-arch=sm_90`, and profiles with `ncu`.
- `ARCH=single` uses CUDA 9, `-arch=sm_70`, `GPGPU_SIM`, and `gpgpu-sim-single`.
- `ARCH=cais` uses CUDA 9, `-arch=sm_70`, `GPGPU_SIM`, and `gpgpu-sim-cais`.
- `CONF=GV100` stages `SM7_GV100`; `CONF=GH100` stages `SM7_GH100`.
- GH100 simulator configs are compatibility configs, not validated Hopper models.
- `CONF=GV100` adds `-DHW_V100`; `CONF=GH100` adds `-DHW_H100`.
- `SIM_CONFIG_SED` can patch the staged `gpgpusim.config` copy at run time via
  `sed -i -e "<expr>"` (supports semicolon-separated expressions in one string).
  Use this for sweep experiments instead of editing simulator source configs.
- Every CUDA source must reject missing hardware macros with `#error` and print
  a compile-time `#pragma message` identifying Volta V100 or Hopper H100.

Run directories are arch-rooted:

```text
run-native/GV100
run-perf/GV100
run-single/GV100
run-cais/GV100
run-native/GH100
run-perf/GH100
run-single/GH100
run-cais/GH100
```

Do not reintroduce bare `run/<arch>` paths or `run-<CONF>/<arch>` paths.

## GCC 5.4 Rule

CUDA 9 workload compilation sources this external file:

```text
/home/simuser/tools/opt/source-gcc54
```

Do not put GCC 5.4 binaries, wrapper scripts, or header overlays back into this
repo. The external setup owns those details. The repo only calls it through
`GCC54_SOURCE` in `common.mk`.

## Source Rules

For lower-case microbenchmarks:

- Native measurement quality is the priority.
- Native and simulator work sizes do not need to match.
- Use `#ifdef GPGPU_SIM` / `#ifndef GPGPU_SIM` to reduce simulator work.
- Keep simulator warmup and measured kernels as separate launches when possible
  because GPGPU-Sim reports per-kernel aggregate statistics.
- Any warm used for hit-latency measurement must walk the entire working set.
  Do not use fixed warm iteration counts that are independent of working-set
  size.
- For `e_hbm_hit_latency` and `f_hbm_bank_parallel`, default runs are now
  cold-run sweeps over three L2-relative sizes (`lt_l2`, `approx_l2`, `ge_l2`)
  with hardware-specific byte values for V100 vs H100.
  - `ARCH=native`: default measured iteration count is 10 and warm iteration is 0.
  - `ARCH=perf`: default measured iteration count is 1 and warm iteration is 0.
  - `ARCH=cais`/`ARCH=single`: default measured iteration count is 1.
  - GPGPU-Sim report parsing for these kernels should treat each
    `kernel_launch_uid` block in `stats.txt` as one launch and use the raw
    `L2_BW` value from that block as the launch bandwidth.
  - `report.py` should now act as a raw extractor only:
    - grep the requested fields from each launch chunk
    - emit CSV directly
    - do not compute launch deltas or synthetic miss rates
  - The generated CSV must include a second row that describes each column's
    statistic type, for example `per_kernel`, `cumulative`, `app_per_kernel`,
    or `text`.
  - `L2_BW_total` and the raw printed `L2_total_cache_miss_rate` line are
    cumulative run-to-date views. Do not present them as per-kernel HBM
    bandwidth or per-kernel miss rate.
  - Do not emit `L2_total_cache_miss_rate` in the CSV for these HBM kernels.
    It is too easy to misread as a per-launch cold-stream metric.
  - Any CUDA-application CSV row data parsed from `stats.txt` should be printed
    with an `APP_` prefix on every field name so simulator counters remain
    visually distinct from app-side measurements.
  - `a_float_op_latency` specifically:
  - native/perf keeps the long chain (`32768`) for measurement quality
  - `GPGPU_SIM` uses a shorter chain (`4096`) so `ARCH=cais` completes in about 50 seconds on the current CAIS build while preserving the dependent-chain instruction pattern

For `VAL_*` microbenchmarks:

- These are warm/measure split adaptations of the lower-case kernels.
- Under `GPGPU_SIM`, keep warmup as a separate kernel launch so `report.py`
  can reason about per-launch deltas cleanly.
- Keep the source kernel’s access pattern and thread/block philosophy; only
  split warm from measurement and trim simulator work to practical sizes.
- `VAL_hbm_parallel` also follows cold-run defaults now:
  - default warm iteration count is 0 (variable retained as an optional runtime
    argument)
  - default measured iteration count is 10 on native and 1 on perf/simulator
  - default no-arg execution sweeps `lt_l2`, `approx_l2`, and `ge_l2`
- For `VAL_l1_parallel` on `ARCH=cais CONF=GH100`, queue-side config-only probes
  on the current CAIS tree did not materially change the curve when toggling:
  - `-gpgpu_cache:dl1` data-port width
  - `-gpgpu_l1_banks`
  - `-gpgpu_l1_banks_byte_interleaving`
  - `-gpgpu_simt_core_sim_order`
- CAIS did not support PTX `%smid` before this calibration pass; same-SM
  warm/measure validation needed a simulator patch in
  `gpgpu-sim-cais/src/cuda-sim/ptx_sim.cc` so `SMID_REG` returns
  `get_hw_sid()`.
- `VAL_l1_parallel` now launches one CTA per SM under `GPGPU_SIM` and gates the
  real work to one CTA on hardware SM `0`, so warm and measured phases hit the
  same SM-local L1D instead of rotating across different SMs.
- Raising `-gpgpu_l1_latency` mostly shifts the whole curve upward by an almost
  constant offset rather than changing its shape.

For `REAL_*` workloads:

- Native and simulator problem sizes must match.
- Under `GPGPU_SIM`, only reduce repeat or iteration count.
- Keep deterministic correctness checks and emit CSV-like summary rows.
- Emit multiple named cases when possible. Native `GV100` cases should cross the
  6 MiB V100 L2 point (`<`, approximate, and `>>`). Native `GH100` cases should
  cross the roughly 50 MiB H100 L2 point.
- For HBM-facing `REAL_*` kernels (`REAL_reduce`, `REAL_spmm`, `REAL_gather`,
  `REAL_scatter`, `REAL_softmax`), use cold-run defaults:
  - warm launch count stays explicit but defaults to 0
  - native launch repeat defaults to 10
  - perf/simulator launch repeat defaults to 1

## Validation Rules

- Build-test `ARCH=native CONF=GV100`, `ARCH=perf CONF=GV100`, and
  `ARCH=single CONF=GV100` after touching shared infrastructure or source.
- Smoke-test `ARCH=native CONF=GH100` and `ARCH=perf CONF=GH100` compilation
  after changing native arch selection.
- Always re-check the dumped SASS and PTX after compilation to make sure logics
  are not optimized out.
- Check NCU perf result and see if cache/memory transaction and miss rate is reasonable.
  E.g. HBM test should have very low L2 hit rate.

## Benchmark Correctness
- `for` loop should `#pragma unroll 128` to reduce branch overhead.
- Use SINGLE THREAD to test the hit latency of L1/L2/HBM.
- `c_smem_latency` should use a SINGLE THREAD dependent shared-memory chain.
- For `ARCH=native`, kernel should run multiple times and record at least
  `min`, `median`, and `avg`.
  For `ARCH=perf`, no reptitive run is required since NCU will run it multiple times.
