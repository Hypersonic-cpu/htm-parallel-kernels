#!/bin/zsh

set -euo pipefail

for CASENAME in  1 4 16 32 64 132 256 512 $((3*256)) 1024 2048 4096 $((3*2048)) 8192; do make ARCH=cais CONF=GH100 RUN_ROOT=run-GH100-ReadBWStats-${CASENAME} RUN_ARGS="${CASENAME}" run 1> /dev/null& done; wait
