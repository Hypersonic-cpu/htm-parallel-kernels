# `cal_kernels/REAL_gather` notes

This folder inherits the shared rules in [../AGENTS.md](../AGENTS.md).

- Use `REAL_GATHER_CASE=<case_name>` to run a single case (for example
  `REAL_GATHER_CASE=ll_l2`) without editing source.
- `run_case()` now performs an additional CPU-side correctness guard for
  simulator cache-maintenance work:
  - it writes a deterministic pattern to `dev_x` in a dedicated kernel launch
  - runs gather as usual
  - checks both gathered outputs and the final `dev_x` contents after D2H copy
    to catch unexpected dirty-line loss across kernel boundaries.
