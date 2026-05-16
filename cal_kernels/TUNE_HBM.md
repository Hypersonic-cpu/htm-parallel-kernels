# GH100 HBM Tuning Log

This file tracks ongoing GH100 CAIS HBM calibration work in `cal_kernels`,
with current priority:

- prioritize `Read BW`
- prioritize total kernel time (`ns`)
- evaluate sweeps by worst-case error, not MAE
- temporarily ignore write BW because it is entangled with flush / accounting

## Ground Truth

Real-H100 reference data currently comes from:

- `/home/simuser/Archive/h100runs-v5/f_hbm_bank_parallel/run-GH100-AFlush-Seq/perf/ncu.csv`

For `f_hbm_bank_parallel`, compare only the probe kernel against:

- `dram__bytes_read.sum.per_second`
- `gpu__time_duration.sum`

## Important Findings So Far

### 1. The main visible `L2_BW` gain came from address striping

At `nblk=512`, the baseline GH100-compatible config (`dramid@8`) produced about:

- DRAM-read BW `982 GB/s`
- time `19754 ns`

Changing only the channel-striping direction to a finer interleave improved the
probe substantially:

- `dramid@7`
- plus `-dram_bnk_indexing_policy 1`

At `nblk=512`, that moved the simulator to about:

- `L2_BW` `1526 GB/s`
- time `12714 ns`

But this was later found to be **not equal** to DRAM read bandwidth.

### 2. That improvement was not a complete solution

The `dramid@7 + bank_idx=1` line improves mid-range read BW, but the full sweep
still has large worst-case error:

- low `nblk` total time is too small
- high `nblk` read BW overshoots

Observed on the `run-GH100-DRAMFix-Parallel-*` line:

- worst read-BW error: about `67.5%`
- worst total-time error: about `44.9%`

### 3. Writeback-accounting fixes are real, but not the current objective

The CAIS tree now distinguishes:

- `flush_*_cache`: write back dirty lines
- `force_invalidate_*_cache`: invalidate contents

`memory_sub_partition::flushL2()` now emits explicit `L2_WRBK_ACC` traffic for
dirty L2 lines when writeback is requested. This fixed the previous situation
where some probe runs showed near-zero DRAM writes even though the kernel does
perform global stores.

For the current phase, this matters only as context. Write BW is temporarily not
the calibration target.

## Key Tried Config Directions

### Baseline GH100-compatible path

- `-gpgpu_mem_addr_mapping dramid@8;...`
- `-dram_bnk_indexing_policy 0`
- `-dram_bnkgrp_indexing_policy 1`
- `-gpgpu_cache:dl2 S:64:128:80,L:B:m:L:P,A:192:4,32:0,32`

Representative `nblk=512` result:

- read BW `982.0 GB/s`
- time `19753.6 ns`
- vs real `2108.6 GB/s`, `15936 ns`

### Read-focused mapping improvement

- `-gpgpu_mem_addr_mapping dramid@7;...`
- `-dram_bnk_indexing_policy 1`
- `-dram_bnkgrp_indexing_policy 1`
- keep current L2 latency setting unchanged

Representative `nblk=512` result:

- `L2_BW` `1525.8 GB/s`
- time `12713.7 ns`

This is currently the best `L2_BW` point that had been observed, but it is not a
valid DRAM-read-BW result.

Recorded example run:

- `cal_kernels/f_hbm_bank_parallel/run-GH100-Debug-Map-Dr7-BK1/cais/report.csv`

For that run, the probe kernel (`kernel_launch_uid=2`) reported:

- `gpu_mem_bw = 1525.8119 GB/s` (`stats.txt` `L2_BW`)
- `gpu_sim_cycle = 23266`
- converted time at `1.83 GHz` core clock: `12713.7 ns`
- `total dram reads = 390810`
- recomputed DRAM read BW = `390810 * 32B / 12713.7 ns = 983.66 GB/s`

Against the current real-H100 reference at `nblk=512`:

- real read BW `2108.6 GB/s`
- real time `15936 ns`

So the previous `1.5 TB/s read BW` claim was wrong. The correct interpretation
is:

- `L2_BW` error vs real read BW: not a valid comparison
- DRAM read BW error `-53.3%`
- time error `-20.2%`

This point does **not** cross the `1.5 TB/s` DRAM-read-BW threshold.

## Metric Correction

For this benchmark:

- `report.csv` `gpu_mem_bw` comes from `stats.txt` `L2_BW`
- `L2_BW` is printed from `partiton_replys_in_parallel * 32B / time`
- it is a reply-path throughput metric, not the same thing as DRAM read BW
- calibration against real H100 read bandwidth must instead use:
  - simulator `total dram reads (delta) * 32B / kernel time`
  - real `dram__bytes_read.sum.per_second`

All future BW comparisons in this file should use DRAM-read BW unless explicitly
labeled as `L2_BW`.

### Queue-depth sweeps

Tried increasing queueing capacity:

- `-gpgpu_dram_partition_queues 128:128:128:128`
- `-gpgpu_frfcfs_dram_sched_queue_size 128`
- `-gpgpu_dram_return_queue_size 256`

Effect:

- little to no meaningful improvement in read BW
- did not fix worst-case total time

### DRAM timing sweeps on the old baseline line

Tried timing-only changes such as:

- `RRD=4`
- reduced `RCD/RP/RC/RAS`
- changed `CCDL`

Effect:

- these did not improve the calibration target
- representative 512-block results were worse than baseline in both read BW and
  total time

### L2 write-policy variants

Tried alternatives on top of the `dramid@7 + bank_idx=1` line, for example:

- `L:B:m:N:P`
- `L:E:m:N:P`
- `L:L:m:N:P`
- `L:L:m:L:P`

Effect:

- some variants brought write BW closer
- but the full sweep often lost read BW and/or total time badly
- this is why the previously observed `1.5 TB/s` point and the later
  `1.08 TB/s` point both existed: they came from different L2 write-policy
  settings with different tradeoffs

## Current Working Hypothesis

The next useful direction is:

1. stay on the `dramid@7 + bank_idx=1` line because that is the only strong
   read-BW gain found so far
2. tune read-side memory latency / throughput balance without changing:
   - frequency
   - `-gpgpu_l2_rop_latency`
3. focus on reducing:
   - low-`nblk` total-time underestimation
   - high-`nblk` read-BW overshoot

## In-Progress Direction

Started a fixed-latency probe on top of the current `dramid@7 + bank_idx=1`
line:

- `SIM_CONFIG_SED='s#^-dram_latency .*#-dram_latency 160#'`
- `SIM_CONFIG_SED='s#^-dram_latency .*#-dram_latency 220#'`

Targeted runs:

- `run-GH100-Lat160-16`
- `run-GH100-Lat160-64`
- `run-GH100-Lat160-8192`
- `run-GH100-Lat220-16`
- `run-GH100-Lat220-64`
- `run-GH100-Lat220-8192`

Why this direction:

- the worst total-time error is currently at low `nblk`, where the simulator is
  too fast
- `dram_latency` is a fixed-latency style knob, so it is more likely to raise
  low-concurrency time than to completely destroy the 512-block `1.5 TB/s`
  read-BW point

Observed result:

- `dram_latency=160`
  - `nblk=16`: read `405.2 GB/s` (`-17.0%`), time `41605.5 ns` (`-39.5%`)
  - `nblk=64`: read `1108.9 GB/s` (`-21.4%`), time `15425.7 ns` (`-35.2%`)
  - `nblk=8192`: read `2351.6 GB/s` (`+67.1%`), time `24970.5 ns` (`+4.6%`)
- `dram_latency=220`
  - `nblk=16`: read `368.7 GB/s` (`-24.5%`), time `45720.2 ns` (`-33.5%`)
  - `nblk=64`: read `1073.1 GB/s` (`-24.0%`), time `15939.3 ns` (`-33.1%`)
  - `nblk=8192`: read `2316.1 GB/s` (`+64.5%`), time `25353.6 ns` (`+6.2%`)

Conclusion:

- raising fixed `dram_latency` alone is not enough
- it only modestly improves low-`nblk` time
- it does not address the large high-`nblk` read-BW overshoot

## Current Review Snapshot

Using the real-H100 reference from `run-GH100-AFlush-Seq`, the four most useful
anchor points for the probe kernel are:

| nblk | real read BW (GB/s) | real time (ns) |
|---|---:|---:|
| 16 | 488.43 | 68800 |
| 64 | 1411.43 | 23808 |
| 512 | 2108.58 | 15936 |
| 8192 | 1407.57 | 23872 |

For the current best read-focused completed line
`dramid@7 + dram_bnk_indexing_policy=1 + dram_bnkgrp_indexing_policy=1`
(`run-GH100-DRAMFix-Parallel-*`):

| nblk | sim read BW (GB/s) | sim time (ns) | read error | time error |
|---|---:|---:|---:|---:|
| 16 | 444.83 | 37900.5 | -8.9% | -44.9% |
| 64 | 1132.20 | 15107.7 | -19.8% | -36.5% |
| 512 | 1525.81 | 12713.7 | -27.6% | -20.2% |
| 8192 | 2357.93 | 24903.3 | +67.5% | +4.3% |

This makes the current problem very concrete:

- the `512` point is the first one to cross `1.5 TB/s`
- but the worst read-BW error is still the `8192` overshoot
- the worst total-time error is still on low `nblk`

## Pending Probe

Started a nearby conservative variant:

- `-gpgpu_mem_addr_mapping dramid@7;...`
- `-dram_bnk_indexing_policy 1`
- `-dram_bnkgrp_indexing_policy 0`

Run roots:

- `run-GH100-Dr7BG0BK1-16`
- `run-GH100-Dr7BG0BK1-64`
- `run-GH100-Dr7BG0BK1-512`
- `run-GH100-Dr7BG0BK1-8192`

Status at last review:

- these runs have not produced `report.csv` yet
- `stats.txt` in all four directories currently stops after the probe kernel is
  launched, so they are still in-flight and must not be treated as validated
  results

## Next Direction

Since timing-only changes did not fix the shape, the next targeted comparison is
to stay on `dramid@7` and reduce service parallelism slightly:

1. `dramid@7` only
2. `dramid@7 + dram_bnkgrp_indexing_policy=0`

The goal is to preserve as much of the `512` read-BW gain as possible while
reducing:

- low-`nblk` time underestimation
- high-`nblk` read-BW overshoot

## 2026-05-06: Recalibration Restart (DRAM-read metric)

After correcting the metric definition, all read-BW comparisons below use:

- `DRAM Read BW = (total dram reads delta) * 32B / kernel_time`

### Completed checks

1. `dramid@7 + bank_idx=1 + bankgrp_idx=1` (same config as previous `Read15`)
   at `nblk=512`:
   - DRAM read BW `983.66 GB/s` (not `1.53 TB/s`)
   - time `12713.7 ns`
   - vs real (`2108.58 GB/s`, `15936 ns`): read `-53.35%`, time `-20.22%`

2. `Memcpy warmup off`: `-gpgpu_perf_sim_memcpy 0`
   - `nblk=16`: DRAM read BW `442.67 GB/s`, time `37900.5 ns`
   - `nblk=64`: DRAM read BW `1110.51 GB/s`, time `15107.7 ns`
   - `nblk=512`: DRAM read BW `1120.16 GB/s`, time `14977.6 ns`
   - `nblk=512` vs real: read error improved from `-53.35%` to `-46.88%`,
     time error improved from `-20.22%` to `-6.01%`
   - note: `nblk=16/64` stayed close to prior values, so this knob mainly helps
     mid/high-concurrency points

### Failed / blocked checks

1. `-gpgpu_cache:dl2 S:64:64:80,...`
   - aborts at cache init assertion:
     `m_line_sz / SECTOR_SIZE == SECTOR_CHUNCK_SIZE`
   - this line-size combination is invalid for the sector-cache path

2. `-gpgpu_cache:dl2_texture_only 1` (`nblk=64/512`)
   - currently repeatedly stuck at probe-kernel phase and not producing
     `report.csv` within timeout windows
   - treated as blocked until root cause is diagnosed

### Current practical state

- best confirmed improvement so far on corrected metric is
  `-gpgpu_perf_sim_memcpy 0` at `nblk=512`
- low-concurrency (`nblk=16/64`) time is still too fast
- additional config lines that significantly alter L2 behavior are currently
  prone to hangs/non-termination in this benchmark workflow

## Current Best Sweep Under The New Objective

No existing sweep meets the current objective yet.

Among the existing sweep families:

- `run-GH100-FullRun-Parallel-*`
  - worst read-BW error: `54.2%`
  - worst total-time error: `44.2%`
- `run-GH100-DRAMFix-Parallel-*`
  - worst read-BW error: `67.5%`
  - worst total-time error: `44.9%`
- `run-GH100-TuneW-B-Parallel-*`
  - worst read-BW error: `48.6%`
  - worst total-time error: `67.6%`

So the current task is still open.
