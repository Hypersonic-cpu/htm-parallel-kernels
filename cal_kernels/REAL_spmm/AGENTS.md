# `REAL_spmm` Run Guide

Use this directory for SpMM calibration runs and sweep helpers.

## Script workflow

- `scripts/launch_cais_no_timeout.sh`
  - stages and runs one GH100 CAIS case immediately
  - usage: `<run_root> <run_arg> <profile>`
- `scripts/prepare_cais_profile_run.sh`
  - stages one run directory without launching it
- `scripts/submit_cais_profile_run.sh`
  - background-launches one staged run through `nohup`
- `scripts/stage_and_submit_cais_profile_run.sh`
  - stages, then launches in one step

## Runtime-library isolation

- For branch-local simulator experiments, do not overwrite the default CAIS
  `libcudart.so`.
- Build or install a variant `.so` somewhere else, then launch with:
  - `SIM_LIBCUDART_SO=/abs/path/to/libcudart.so.9.0`
- `cal_kernels/common.mk` will validate the resolved runtime library before
  the simulator starts and will fail fast if the wrong `libcudart` is linked.

## Profiles

- `mu4rq16` is the lightest SpMM throughput profile here and is a good first
  launch-smoke choice on this branch.
- `tuneA` through `tuneJ` and `bypassA/B` are staged config sweeps from the
  issue/admission debugging track. Keep them script-only; do not treat them as
  architectural fixes by themselves.
