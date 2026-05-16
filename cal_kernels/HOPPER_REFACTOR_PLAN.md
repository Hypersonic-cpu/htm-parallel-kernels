# Hopper Calibration Refactor Plan

Working branch: `cal-hopper`

Editable plan path:

```text
cal_kernels/HOPPER_REFACTOR_PLAN.md
```

## Goals

- Refactor the numbered Volta calibration kernels into a Hopper-oriented suite.
- The existing V100 artifacts can be deleted - I've backup-ed them.
- Make future run directories config-specific: `run-<CONF>/<arch>/`, e.g.
  `run-GV100/native`, `run-GH100/native`.
- Add a `GH100` config option for native collection now.
- Do not require or test `run-GH100/single` until a GH100 GPGPU-Sim config
  exists.
- Keep native workloads focused on precise hardware measurement. For lower-case
  prefixed micro-benchmarks (e.g. `a_*` through `f_*`),
  native/simulator work sizes no longer need to match.
- Keep `REAL_*` workloads native/simulator consistent except for reduced
  iteration count under `GPGPU_SIM`; problem size must remain the same.

## Directory Layout

Refactor the numbered directories under `cal_kernels` into this target layout:

```text
cal_kernels/
  a_float_op_latency/
  b_l1_hit_latency/
  b2_l1_size_sweep/
  c_smem_latency/
  d_l2_hit_latency/
  d2_l2_size_assoc/
  d3_l2_bank_parallel/
  e_hbm_hit_latency/
  f_hbm_bank_parallel/
  REAL_reduce/
  REAL_spmm/
```

Current numbered kernels may be reused as implementation hints, but each new
kernel should be reviewed against the intended Hopper measurement.

Mapping from current kernels:

```text
01_dep_chain                  -> a_float_op_latency
02_l1_ptr_chase               -> b_l1_hit_latency
03_l1_assoc                   -> b2_l1_size_sweep
07_smem_bank                  -> c_smem_latency
04_l2_ptr_chase               -> d_l2_hit_latency
11_l2_capacity_sweep          -> d2_l2_size_assoc
12_l2_hot_parallel            -> d3_l2_bank_parallel
14_hbm_miss_latency           -> e_hbm_hit_latency
16_hbm_bank_parallel          -> f_hbm_bank_parallel
validate_single or new source -> REAL_reduce
validate_single or new source -> REAL_spmm
```

Directories not in the target Hopper suite should either be archived in a
clearly named legacy folder or removed from the active `cal_kernels` run list
after confirming with the user.

## Run Directory Refactor

Change `common.mk` variables:

```make
CONF ?= GV100
RUN_ROOT ?= run-$(CONF)
BUILD_DIR ?= build/$(CONF)/$(ARCH)
RUN_DIR ?= $(RUN_ROOT)/$(ARCH)
```

Expected outputs:

```text
cal_kernels/a_float_op_latency/run-GV100/native/
cal_kernels/a_float_op_latency/run-GV100/single/
cal_kernels/a_float_op_latency/run-GH100/native/
```

Rules:

- Existing `run/` directories must be migrated to `run-GV100/`.
- Existing `run.bak/` directories should be left alone unless explicitly needed.
- `ARCH=native CONF=GH100 run` is allowed.
- `ARCH=single CONF=GH100 run` must fail early with a clear message until a
  GH100 simulator config exists.
- `CONF=GV100 ARCH=single` keeps using current Volta simulator config.
- `CONF=GH100 ARCH=native` uses CUDA 12 and runs directly.
- `CONF=GH100 ARCH=perf` uses CUDA 12 and `ncu`.

Update scripts that assume `run/`:

- `cal_kernels/common.mk`
- `cal_kernels/common/report.py`
- `cal_kernels/pack_runs.py`
- `cal_kernels/README.md`
- `cal_kernels/AGENTS.md`
- repo root `README.md` if it documents calibration commands

`pack_runs.py` target layout should become:

```text
SPECIFIED_NAME/a_float_op_latency/run-GV100/native
SPECIFIED_NAME/a_float_op_latency/run-GV100/single
SPECIFIED_NAME/a_float_op_latency/run-GH100/native
```

It should copy immediate `run-*` directories, not bare `run/`.

## Workload Rules

For `a_*` through `f_*`:

- Native should prioritize clean, precise H100 measurement.
- Use `#ifndef GPGPU_SIM` / `#ifdef GPGPU_SIM` blocks in `.cu` files.
- Under `GPGPU_SIM`, reduce work size aggressively if needed.
- Add explicit simulator warmup kernels or warmup phases that can be excluded or
  separated from measured kernels where possible.
- Avoid TLB influence in simulator measurements by warming the touched address
  range before the measured kernel.
- Because GPGPU-Sim reports per-kernel aggregate stats, measured kernels should
  be separate launches from warmup kernels.

For `REAL_*`:

- Native and simulator problem size must match.
- Under `GPGPU_SIM`, only reduce iteration count or repeat count.
- Validate functionality with deterministic checksums.
- Keep work relatively small so native `ncu` and future simulator runs are both
  manageable.

## Native Hopper Measurement Set
### `a_float_op_latency`

- Dependent chain for `fadd`, `fmul`, and optionally `fma`.
- Use proper `#pragma unroll 128` to reduce loop branch overhead.
- On V100, FADD should have latency around 9 cycles. Do not mark this workload
  done if measured FADD latency is far from that without understanding the SASS
  and timing method.
- Report cycles per op from `clock64` and NCU instruction counts.
- Native size can be large enough to reduce timer noise.
- Simulator size can be small.

### `b_l1_hit_latency`

- SINGLE Pointer-chase chain targeting L1 hits.
- Direct `ptr = *ptr`, no arithmetic addr calc required.
- Use cache operators appropriate for Hopper native measurement.
- Volta reference hint: L1 latency was around 28 cycles.

### `b2_l1_size_sweep`

- Sweep working set around expected L1 capacity and associativity boundaries.
- Emit CSV rows containing working-set size, stride, and measured latency.

### `c_smem_latency`

- Dependent shared-memory load chain.
- Include bank-conflict variants only if they directly help calibration.

### `d_l2_hit_latency`

- Single-thread or low-parallelism dependent global load chain targeting L2 hits.
- Volta reference hint: single-thread L2 hit latency was around 200 cycles.

### `d2_l2_size_assoc`

- Sweep L2 working-set size and conflict pattern.
- Output capacity/associativity-sensitive latency and hit-rate metrics.

### `d3_l2_bank_parallel`

- High-parallelism L2 bank/partition test. A multi-thread version of `d_l2_hit_latency`.
- Intended to calibrate interconnect queue size and bandwidth-related effects.
- Report achieved bandwidth, kernel time, and relevant NCU L2 metrics.

### `e_hbm_hit_latency`

- DRAM/HBM latency probe with L2 bypass or large enough footprint to miss L2.
- Separate warmup from measured launch to reduce TLB/noise.

### `f_hbm_bank_parallel`

- High-parallelism HBM bandwidth/bank-level parallelism test.
- Intended for L2-HBM bandwidth calibration.
- Volta reference hint: measured HBM was about 900 GiB/s peak, about 750 GiB/s
  practically achievable for this suite.

## REAL Workloads

### `REAL_reduce`

- Deterministic reduction over a modest vector.
- Fixed problem size across native and simulator.
- `GPGPU_SIM` may reduce repeat count only.
- Emit checksum and timing/stat rows.

### `REAL_spmm`

- Deterministic sparse matrix times dense vector or small dense matrix.
- Use a fixed generated sparse pattern checked on host.
- Generate CSR format matrix and dense matrix on host. Randomized with fixed seed.
- Fixed problem size across native and simulator.
- `GPGPU_SIM` may reduce repeat count only.
- Emit checksum and timing/stat rows.

### `REAL_gather`
 - Single card gather sample kernel, `y[i] = x[idx[i]]`.
 - Test read request coalescing.
 - `GPGPU_SIM` may reduce repeat count only.

### `REAL_scatter`
 - Single card scatter sample kernel, `y[idx[i]] = x[i]`.
 - Test write request coalescing.
 - `GPGPU_SIM` may reduce repeat count only.


## Makefile and NCU Test Requirement

On the current V100 machine, test `ARCH=native` direct execution and `ARCH=perf`
NCU collection for every new Hopper workload that can run on V100:

```bash
make -C cal_kernels/a_float_op_latency ARCH=native CONF=GV100 run report
make -C cal_kernels/a_float_op_latency ARCH=perf CONF=GV100 run report
make -C cal_kernels/b_l1_hit_latency ARCH=native CONF=GV100 run report
make -C cal_kernels/b_l1_hit_latency ARCH=perf CONF=GV100 run report
...
make -C cal_kernels/REAL_spmm ARCH=native CONF=GV100 run report
make -C cal_kernels/REAL_spmm ARCH=perf CONF=GV100 run report
```

This validates CUDA 12, NCU invocation, PTX/SASS dump, report parsing, and CSV
format. It does not validate Hopper numerical targets.

For later H100 collection:

```bash
make -C cal_kernels/a_float_op_latency ARCH=native CONF=GH100 run
make -C cal_kernels/a_float_op_latency ARCH=perf CONF=GH100 run report
```

The target `run` should use `tee` to print output to screen as well as store
to a .txt run directory. `ARCH=native report` intentionally prints nothing.
`ARCH=perf report` should grep information from ncu outputs and re-arrange them
into CSV; `ARCH=single report` should do the same for GPGPU-Sim outputs.

## Implementation Phases

1. Preserve V100 artifacts:
   - Move current numbered `run/` directories to `run-GV100/`.
   - Confirm no current run artifacts are lost.

2. Refactor shared infrastructure:
   - Add `CONF`.
   - Add `run-$(CONF)` layout.
   - Add `GH100` native option.
   - Make `ARCH=single CONF=GH100` fail clearly.
   - Update report and pack scripts.

3. Replace numbered kernels with Hopper suite:
   - Create `a_*` through `f_*` directories.
   - Use current kernels only as starting points.
   - Add explicit `GPGPU_SIM` source guards.
   - Keep warmup launches separate from measured launches.

4. Add `REAL_*` workloads:
   - Implement `REAL_reduce`.
   - Implement `REAL_spmm`.
   - Add deterministic correctness checks.

5. Validate native on V100:
   - Build and run all new workloads with `ARCH=native CONF=GV100`.
   - Ensure each produces NCU CSV, PTX, SASS, `stats.txt`, `report.txt`, and
     `report.csv`.
   - Ensure statistics looks resonable and close to reference value on V100. Give brief report for each kernel at last. Make sure they're not influenced by compilter optimization, bypassing etc. 
   - If there's any problem, check SASS and PTX for debugging. If you cannot solve this question, do NOT mark it as DONE at the bottom of this file. Also report to me at last.
   - Make sure CSR is generated. 

6. Document:
   - Update `cal_kernels/README.md`.
   - Update `cal_kernels/AGENTS.md`.
   - Update root docs if command examples changed.

## Other Questions Answered by User
- About Multi-GPU benchmarks: Not used so far, removed.
- Should old numbered source directories: Unused kernels could be deleted directly. 
  All `run*/` data are backup-ed. 
- For H100, the native should explicitly use `sm_90` (NOT `SM_90A`) 
  with current native CUDA version(12).
- For V100, use `sm_70` with native CUDA. 
- For all simulation, use SM70+CUDA9.

## Progress Tracking:

Only mark `[x]` after test-pass.
- [x] `a_float_op_latency`
- [x] `b_l1_hit_latency`
- [x] `b2_l1_size_sweep`
- [x] `c_smem_latency`
- [x] `d_l2_hit_latency`
- [x] `d2_l2_size_assoc`
- [x] `d3_l2_bank_parallel`
- [x] `e_hbm_hit_latency`
- [x] `f_hbm_bank_parallel`
- [x] `REAL_reduce`
- [x] `REAL_spmm`
- [x] `REAL_gather`
- [x] `REAL_scatter`
