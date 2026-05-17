# `mgpu_kernels`: Real-CUDA Multi-GPU Validation

This subtree is for lightweight multi-GPU CUDA workloads used to validate the
CAIS `real_cuda` execution model.

## Layout Rules

- Mirror `cal_kernels` structure:
  - one directory per benchmark
  - `Makefile`
  - `src/`
  - `build/<CONF>/<ARCH>/`
  - `run-<ARCH>/<subdir>/`
- Reuse `cal_kernels/common.mk` unless there is a benchmark-specific reason not
  to.

## Behavioral Rules

- Treat host-side CUDA semantics as the reference:
  - `cudaSetDevice()` selects the executing GPU
  - each `cudaMalloc()` belongs to exactly one GPU
  - each kernel launch runs on exactly one GPU
  - peer traffic comes from ordinary CUDA peer pointers and peer access enable
- Prefer deterministic validation and small simulator work sizes.
- Use direct peer reads/writes for v1. Do not depend on in-switch computing.
- Reuse `cal_kernels/common.mk` directly. Avoid benchmark-local build logic
  unless the validation really needs it.

## Current Benchmarks

- `all_gather`: each GPU reads all peer inputs and writes a gathered output.
- `all_reduce`: each GPU reads all peer inputs and writes its own reduced
  output.
- `peer_copy`: each GPU reads another GPU's memory and stores the result
  locally to validate forward routing correctness.

## Current Status

- `ARCH=native CONF=GV100` is passing for `all_gather`, `all_reduce`, and
  `peer_copy` at 1, 2, and 4 GPUs.
- `ARCH=cais CONF=GV100` is also passing for the same benchmark/count matrix.
- `ARCH=cais CONF=GH100` `all_gather` now launches and passes at 1 GPU with the
  40-channel GH100 config.
  - The working GH100 rule is: keep the 64-slot intra-partition mapping string,
    avoid partition IPOLY for 40 channels, and choose an L2 set geometry that
    stays inside the implemented cache IPOLY set counts.

## Measurement Notes

- `all_gather` defaults differ by build mode:
  - default profile is `approx_l2` on native and `test` on simulator
  - supported size profiles via `RUN_ARGS='<num_gpus> <profile>'`:
    - `test` (small, quick sim)
    - `le_l2`
    - `approx_l2`
    - `gt_l2`
  - you can still pass explicit numeric element count as the second field.
- `all_reduce` and `peer_copy` use the same profile names in `RUN_ARGS`.
- For profile `test`, use multi-GPU runs (`num_gpus >= 2`) for validation.
- For `ARCH=cais`, the first `RUN_ARGS` field is also staged into the copied
  `gpgpusim.config` as `-gpgpu_num_devices` through `SIM_NUM_DEVICES`.
  Example: `RUN_ARGS='4 1024'` requests four visible CUDA devices and a
  1024-element chunk.
- Before the measured mgpu kernel, each GPU runs one L2 flush launch and then
  all GPUs are synchronized:
  - native: real flush over the full flush buffer
  - simulator: placeholder flush kernel (sim-side config controls cache state)
- Do not run multiple CAIS mgpu invocations concurrently into the same
  `RUN_ROOT`/`RUN_DIR`; staging rewrites the copied run config. Use distinct
  `RUN_ROOT`s for parallel experiments.
- In CAIS 1-GPU `all_gather`, it is expected to see `L2_total_cache_miss_rate > 0`
  while `total dram reads/writes = 0`:
  - misses are mainly write-side (`GLOBAL_ACC_W` miss/sector-miss),
  - L2 policy is lazy fetch-on-read, so those misses do not force DRAM reads,
  - and the small simulator working set avoids writeback evictions.
- For `all_gather` 1-GPU alignment against NCU:
  - `-gpgpu_perf_sim_memcpy 1` can over-warm L2 and suppress DRAM reads.
  - `-gpgpu_perf_sim_memcpy 0` restores non-zero DRAM reads and usually gives a closer HBM-access/BW match to NCU for this case.
  - Changing L2 write-allocate `L -> F` alone is not sufficient here because the kernel stores are full-sector writes.

#### all_gather (1GPU): Why L2 miss rate != 0 but HBM<->L2 traffic is 0

Reproduced runs:

- `make -C mgpu_kernels/all_gather ARCH=perf ... RUN_ARGS='1 4096' run`
- `make -C mgpu_kernels/all_gather ARCH=cais ... RUN_ARGS='1' run` (sim default chunk is 4096)

Observed in CAIS run (`run-investigate-1g/cais/stats.txt`):

- `L2_total_cache_miss_rate = 0.4923`
- `total dram reads = 0`
- `total dram writes = 0`
- `L2_BW = 11.5277 GB/s` (this is subpartition->SM reply traffic, not HBM traffic)
- L2 breakdown: reads are hits, misses are from `GLOBAL_ACC_W` (`MISS` + `SECTOR_MISS`)

Root cause:

1. The L2 write-allocate policy is `LAZY_FETCH_ON_READ` (`-gpgpu_cache:dl2 ... L:B:m:L:P`).
2. Under this policy, write misses allocate/mark modified in L2 without issuing a DRAM read (`wr_miss_wa_lazy_fetch_on_read`).
3. DRAM traffic counters are updated only when requests are pushed into DRAM (`dram_t::push -> memlatstat_dram_access`), which never happens here.
4. Working set is tiny (sim chunk 4096), so there are no L2 evictions/writebacks to DRAM.
5. `-gpgpu_perf_sim_memcpy 1` pre-fills L2 tags on `cudaMemcpy`, so global reads hit in L2.

Extra parsing caveat:

- `cal_kernels/common/report.py` currently sums `dram_reads[i]` / `dram_writes[i]` patterns; CAIS `stats.txt` now prints `total dram reads/writes` tables. So report-side `dram_*_bytes` can appear as zero even when future workloads do generate DRAM traffic.
