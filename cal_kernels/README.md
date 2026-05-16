# Hopper calibration kernels

This directory is being refactored on branch `cal-hopper` for Hopper native
measurement and later GPGPU-Sim calibration.

## Common invocation

Every command must specify `ARCH`; `CONF` defaults to `GV100`.

```bash
make -C cal_kernels/a_float_op_latency ARCH=native CONF=GV100 run
make -C cal_kernels/a_float_op_latency ARCH=perf CONF=GV100 run report
make -C cal_kernels/a_float_op_latency ARCH=native CONF=GH100 run
make -C cal_kernels/a_float_op_latency ARCH=perf CONF=GH100 run report
make -C cal_kernels/a_float_op_latency ARCH=single CONF=GV100 run report
make -C cal_kernels/a_float_op_latency ARCH=cais CONF=GV100 run report
```

- `ARCH=native CONF=GV100` builds `sm_70`, runs directly with CUDA 12, and
  writes application output under `run-native/GV100/`. Its `report` target is
  intentionally empty.
- `ARCH=native CONF=GH100` builds `sm_90`, runs directly with CUDA 12, and
  writes application output under `run-native/GH100/`.
- `ARCH=perf CONF=GV100` builds `sm_70`, profiles with `ncu`, and writes
  artifacts under `run-perf/GV100/`.
- `ARCH=perf CONF=GH100` builds `sm_90`, profiles with `ncu`, and writes
  artifacts under `run-perf/GH100/`.
- `ARCH=single` builds `sm_70` with CUDA 9 and runs through GPGPU-Sim single
  mode under `run-single/<CONF>/`.
- `ARCH=cais` builds `sm_70` with CUDA 9 and runs through the CAIS simulator
  under `run-cais/<CONF>/`.

Simulator workload compilation uses the external GCC 5.4 setup:

```bash
/home/simuser/tools/opt/source-gcc54
```

The repo does not vendor GCC 5.4 wrappers or compiler binaries.

## Active workloads

- `a_float_op_latency`: dependent floating-point operation latency.
- `b_l1_hit_latency`: single pointer-chase chain targeting L1 hits.
- `b2_l1_size_sweep`: L1 working-set size sweep.
- `b3_l1_warp_parallel`: one-warp L1 hit probe with active lane sweep.
- `c_smem_latency`: dependent shared-memory latency / bank behavior.
- `d_l2_hit_latency`: single-thread L2 hit latency.
- `d2_l2_size_assoc`: L2 size and associativity sweep.
- `d3_l2_bank_parallel`: parallel L2 pressure / bank behavior.
- `e_hbm_hit_latency`: HBM/DRAM latency probe.
- `f_hbm_bank_parallel`: HBM bandwidth / bank-level parallelism.
- `REAL_reduce`: deterministic reduction.
- `REAL_spmm`: deterministic CSR SpMM.
- `REAL_gather`: deterministic gather.
- `REAL_scatter`: deterministic scatter.
- `REAL_softmax`: deterministic row-wise softmax.

Lower-case microbenchmarks may use different native and simulator work sizes.
`REAL_*` workloads must keep the same problem size between native and simulator;
only repeat/iteration counts may be reduced under `GPGPU_SIM`.

`REAL_*` workloads emit multiple named cases. The native `GV100` cases are sized
to be below, near, and above the 6 MiB V100 L2 cache. The native `GH100` cases
are sized to be below, near, and above the roughly 50 MiB H100 L2 cache.

HBM-facing kernels now default to cold-run behavior:

- `e_hbm_hit_latency`, `f_hbm_bank_parallel`, and `VAL_hbm_parallel` run
  `lt_l2`, `approx_l2`, `ge_l2` size sweeps by default (hardware-specific byte
  values for V100 vs H100), with warm iteration defaulting to `0`.
- Native defaults to measured iteration/repeat count `10`; perf and simulator
  defaults to `1`.
- `ARCH=perf` uses Nsight Compute cache control (`--cache-control all`) in the
  benchmark Makefiles for these HBM workloads.

## Artifacts

Typical outputs:

- `run-<arch>/<subdir>/stats.txt`: stdout/stderr from the run, saved through
  `tee`.
- `run-perf/<subdir>/ncu.csv`: raw Nsight Compute CSV.
- `run-{native,perf}/<subdir>/<bin>.ptx`: PTX dump from `cuobjdump`.
- `run-{native,perf}/<subdir>/<bin>.sass`: SASS dump from `cuobjdump`.
- `run-<arch>/<subdir>/report.txt`: parsed report text.
- `run-<arch>/<subdir>/report.csv`: explainable CSV generated from the report.

To pack run artifacts:

```bash
cal_kernels/pack_runs.py packed-runs
cal_kernels/pack_runs.py --compress packed-runs
```

The packer copies immediate `run-*` directories and preserves paths such as:

```text
packed-runs/a_float_op_latency/run-GV100/native
packed-runs/a_float_op_latency/run-GV100/perf
packed-runs/a_float_op_latency/run-GH100/native
```
