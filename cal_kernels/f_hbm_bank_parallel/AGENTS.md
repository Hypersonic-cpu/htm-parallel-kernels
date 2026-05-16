# `cal_kernels/f_hbm_bank_parallel`: HBM bank-parallel calibration

This folder inherits the rules from [../AGENTS.md](../AGENTS.md).

## Report parsing rule

- Treat each `kernel_launch_uid` block in `run-*/cais/stats.txt` as one launch.
- Ignore unrelated flush/debug lines before parsing:
  - `not ready`
  - `Dirty lines flushed from L2`
  - `Flushed L2`
- The benchmark CSV summary table in `stats.txt` starts with the `case,...`
  header. Any fields printed from that table must be emitted as `APP_*` names.
- For `ARCH=cais`, the launch-level HBM report must use the raw values from the launch chunk:
  - `L2_BW`
- In CAIS `stats.txt`, `total dram reads`, `total dram writes`,
  `L2_total_cache_accesses`, `L2_total_cache_misses`, `L2_BW_total`, and the
  printed `L2_total_cache_miss_rate` are run-to-date cumulative views.
- `report.py` must not difference those counters for this benchmark anymore.
  It should only export the raw chunk values to CSV.
- The CSV must contain a second row with each column's statistic type.
- Do not emit `L2_total_cache_miss_rate` in the CSV. It is a cumulative mixed
  cache statistic and should not be shown as if it were the per-kernel cold-read
  miss rate.
- This benchmark must avoid same-launch line reuse. A contiguous `float4` pattern lets multiple 16B loads land in the same 128B L2 line and creates false L2 hits even after a correct flush. Keep simulator reads spaced by one full 128B cache line per access.
- For CAIS config sweeps, prefer `make ... SIM_CONFIG_SED='<expr>' run` so the
  staged `run-*/cais/gpgpusim.config` is patched in place without touching
  `gpgpu-sim-cais/configs/tested-cfgs/...`.
- For GH100/H100 read-BW calibration, keep the benchmark access pattern exactly
  as one `float4` every `128B`.
  - In the 64 MiB case this creates `524,288` cold accesses.
  - Real H100 NCU shows about `1,050,104` `dram__sectors_read.sum`, so the
    hardware is effectively pulling about `64B` from DRAM per cold access.
  - CAIS default sector-L2 miss handling only sends one `32B` atom to DRAM per
    cold access. If `total dram reads` lands near `524,289`, that is the
    modeling gap to fix first, not a front-end issue.
  - For this kernel, the best current modeling split is:
    - `-gpgpu_l2_read_fetch_granularity 64`: widen the L2 miss request to DRAM.
    - `-gpgpu_dram_read_burst_granularity 64`: serve that `64B` with one DRAM
      READ command rather than two serialized `32B` commands.
