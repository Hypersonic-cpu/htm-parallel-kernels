#!/usr/bin/env python3
"""Run small, independent Nsight Compute metric groups for one workload.

The helper intentionally owns the whole native/perf profiling transaction:
metric discovery, device/clock validation, one external GPC lock, independent
NCU invocations, raw result preservation, and final CSV merging.  It does not
change CUDA application arguments or simulator behavior.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


WORKFLOW_VERSION = 2
DEFAULT_CHIP = "GH100"
DEFAULT_GPC_CLOCK_MHZ = 1590
GPC_LOCK_TOLERANCE_MHZ = 5.0

# These are runtime frequency counters.  They are deliberately not CUDA
# device attributes, boost limits, or theoretical clock values.
RUNTIME_CLOCK_METRICS = [
    "gpc__cycles_elapsed.avg.per_second",
    "sm__cycles_elapsed.avg.per_second",
    "lts__cycles_elapsed.avg.per_second",
    "dram__cycles_elapsed.avg.per_second",
    # This is an optional NCU counter only.  It is never called the Accel-Sim
    # interconnect clock; ICNT remains unavailable unless a separate suitable
    # counter is added deliberately.
    "ltcfabric__cycles_elapsed.avg.per_second",
]

COMMON_RUNTIME_METRICS = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    *RUNTIME_CLOCK_METRICS,
]

GROUP_METRICS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        (
            "timing_clock",
            [
                *COMMON_RUNTIME_METRICS,
            ],
        ),
        (
            "l1_l2",
            [
                *COMMON_RUNTIME_METRICS,
                "l1tex__t_sector_hit_rate.pct",
                "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
                "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
                "lts__t_sector_hit_rate.pct",
                "lts__t_sector_op_read_hit_rate.pct",
                "lts__t_sector_op_write_hit_rate.pct",
                "lts__t_sectors_op_read.sum",
                "lts__t_sectors_op_write.sum",
                "lts__t_bytes.sum",
            ],
        ),
        (
            "dram",
            [
                *COMMON_RUNTIME_METRICS,
                "dram__bytes_read.sum",
                "dram__bytes_write.sum",
                "dram__sectors_read.sum",
                "dram__sectors_write.sum",
                "dram__bytes_read.sum.per_second",
                "dram__bytes_write.sum.per_second",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            ],
        ),
        (
            "execution",
            [
                *COMMON_RUNTIME_METRICS,
                "sm__inst_executed.sum",
                "sm__warps_launched.sum",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "sm__maximum_warps_avg_per_active_cycle",
                "sm__maximum_warps_per_active_cycle_pct",
                "launch__block_dim_x",
                "launch__block_dim_y",
                "launch__block_dim_z",
                "launch__block_size",
                "launch__grid_dim_x",
                "launch__grid_dim_y",
                "launch__grid_dim_z",
                "launch__grid_size",
            ],
        ),
        (
            "scheduler_stall",
            [
                *COMMON_RUNTIME_METRICS,
                "smsp__warps_eligible.avg",
                "smsp__warps_issue_stalled_barrier.avg",
                "smsp__warps_issue_stalled_dispatch_stall.avg",
                "smsp__warps_issue_stalled_long_scoreboard.avg",
                "smsp__warps_issue_stalled_math_pipe_throttle.avg",
                "smsp__warps_issue_stalled_memory_dependency.avg",
                "smsp__warps_issue_stalled_not_selected.avg",
                "smsp__warps_issue_stalled_short_scoreboard.avg",
                "smsp__warps_issue_stalled_wait.avg",
            ],
        ),
    ]
)

DEFAULT_GROUPS = list(GROUP_METRICS.keys())
FORBIDDEN_MEASURED_CLOCK_METRICS = {
    "device__attribute_clock_rate",
    "device__attribute_max_gpu_frequency_khz",
    "device__attribute_memory_clock_rate",
    "device__attribute_max_mem_frequency_khz",
}

SUMMARY_FIELDS = [
    "group",
    "status",
    "returncode",
    "pass_count",
    "attempt_count",
    "ncu_invocation_mode",
    "transient_retry",
    "launch_count",
    "duration_ns_avg",
    "sm_cycles_avg",
    "gpc_clock_hz_avg",
    "sm_clock_hz_avg",
    "lts_clock_hz_avg",
    "dram_clock_hz_avg",
    "icnt_clock_hz",
    "gpc_lock_status",
    "external_gpc_lock_status",
    "l1_hit_rate_pct_avg",
    "l1_global_load_hit_rate_pct_avg",
    "l1_global_store_hit_rate_pct_avg",
    "l1_load_sectors_sum",
    "l1_store_sectors_sum",
    "l2_hit_rate_pct_avg",
    "l2_read_hit_rate_pct_avg",
    "l2_write_hit_rate_pct_avg",
    "l2_read_sectors_sum",
    "l2_write_sectors_sum",
    "l2_bytes_sum",
    "dram_read_bytes_sum",
    "dram_write_bytes_sum",
    "dram_read_bytes_per_second_avg",
    "dram_write_bytes_per_second_avg",
    "dram_throughput_pct_avg",
    "instructions_sum",
    "warps_launched_sum",
    "occupancy_pct_avg",
    "error",
]


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def parse_metric_list(value: str) -> List[str]:
    return unique(part.strip() for part in value.split(","))


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"N/A", "NA", "NAN", "NOT collected".upper()}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def fmt_number(value: Any) -> str:
    number = parse_float(value)
    if number is None:
        return "N/A"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return "{:.6f}".format(number)


def fmt_optional(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "N/A"
    return str(value)


def mean_metric(rows: Sequence[Mapping[str, str]], name: str) -> Optional[float]:
    values = [parse_float(row.get(name)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def sum_metric(rows: Sequence[Mapping[str, str]], name: str) -> Optional[float]:
    values = [parse_float(row.get(name)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def run_capture(command: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "command failed ({}): {}\n{}".format(
                result.returncode, shlex.join(command), result.stdout.strip()
            )
        )
    return result


def query_metric_catalog(ncu: str, chip: str) -> Tuple[set[str], Dict[str, str]]:
    """Return exact metric names from profiling and launch collections."""

    available: set[str] = set()
    output_by_collection: Dict[str, str] = {}
    for collection in ("profiling", "launch"):
        command = [
            ncu,
            "--query-metrics",
            "--query-metrics-mode",
            "all",
            "--query-metrics-collection",
            collection,
            "--chips",
            chip,
        ]
        result = run_capture(command, check=False)
        output_by_collection[collection] = result.stdout
        if result.returncode != 0:
            raise RuntimeError(
                "ncu metric query failed for {} ({}): {}".format(
                    collection, result.returncode, result.stdout.strip()
                )
            )
        for line in result.stdout.splitlines():
            token = line.strip().split(maxsplit=1)
            if token and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", token[0]):
                available.add(token[0])
    return available, output_by_collection


def query_gpu_info(gpu: str) -> Dict[str, Any]:
    query = [
        "nvidia-smi",
        "-i",
        gpu,
        "--query-gpu=name,memory.total,pci.bus_id,uuid",
        "--format=csv,noheader,nounits",
    ]
    result = run_capture(query, check=False)
    if result.returncode != 0:
        raise RuntimeError("unable to query GPU {} with nvidia-smi: {}".format(gpu, result.stdout.strip()))
    line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        raise RuntimeError("unexpected nvidia-smi GPU query output: {}".format(line))
    name, memory, bus_id, uuid = parts[:4]
    memory_mib = parse_float(memory)
    if "H100" not in name.upper() or memory_mib is None or memory_mib < 75000:
        raise RuntimeError(
            "selected GPU is not an H100 80GB-class device: name={!r}, memory_mib={!r}".format(
                name, memory_mib
            )
        )

    full = run_capture(["nvidia-smi", "-i", gpu, "-q"], check=False)
    full_text = full.stdout if full.returncode == 0 else ""
    form_factor_match = re.search(r"(?im)^\s*(?:Product Name|Board Part Number|Board Type).*?(SXM\s*\d*)", full_text)
    form_factor = form_factor_match.group(1).replace(" ", "") if form_factor_match else "not_reported"
    if form_factor not in {"not_reported", "SXM", "SXM5"}:
        raise RuntimeError("selected H100 form factor is not SXM5: {}".format(form_factor))
    return {
        "index": gpu,
        "name": name,
        "memory_mib": memory_mib,
        "pci_bus_id": bus_id,
        "uuid": uuid,
        "form_factor": form_factor,
        "form_factor_check": "passed" if form_factor in {"SXM", "SXM5"} else "not_reported_by_nvidia_smi",
    }


def query_supported_gpc_clocks(gpu: str) -> List[int]:
    result = run_capture(["nvidia-smi", "-i", gpu, "-q", "-d", "SUPPORTED_CLOCKS"], check=False)
    if result.returncode != 0:
        raise RuntimeError("unable to query supported clocks for GPU {}: {}".format(gpu, result.stdout.strip()))
    values = []
    for line in result.stdout.splitlines():
        if re.search(r"(?i)\b(?:graphics|gpc)\b", line):
            match = re.search(r"([0-9]+)\s*MHz", line)
            if match:
                values.append(int(match.group(1)))
    values = sorted(set(values))
    if not values:
        raise RuntimeError("nvidia-smi did not report supported Graphics/GPC clocks")
    return values


def query_clock_snapshot(gpu: str) -> Dict[str, str]:
    command = [
        "nvidia-smi",
        "-i",
        gpu,
        "--query-gpu=clocks.current.graphics,clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ]
    result = run_capture(command, check=False)
    if result.returncode != 0:
        return {"gpc_mhz": "N/A", "sm_mhz": "N/A", "dram_mhz": "N/A", "error": result.stdout.strip()}
    line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    values = [part.strip() for part in line.split(",")]
    if len(values) < 3:
        return {"gpc_mhz": "N/A", "sm_mhz": "N/A", "dram_mhz": "N/A", "error": line}
    return {"gpc_mhz": values[0], "sm_mhz": values[1], "dram_mhz": values[2]}


def parse_duration(value: str) -> Optional[float]:
    if not value:
        return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([smhdSMHD]?)\s*", value)
    if not match:
        raise ValueError("unsupported timeout value {!r}; use seconds or a suffix s/m/h/d".format(value))
    number = float(match.group(1))
    scale = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[match.group(2).lower()]
    return number * scale


def remove_option(tokens: List[str], option: str, takes_value: bool = True) -> List[str]:
    out = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == option:
            index += 2 if takes_value else 1
            continue
        if takes_value and token.startswith(option + "="):
            index += 1
            continue
        out.append(token)
        index += 1
    return out


def sanitize_ncu_args(value: str) -> List[str]:
    tokens = shlex.split(value) if value else []
    for option in ("--cache-control", "--clock-control", "--metrics", "--log-file", "--page"):
        tokens = remove_option(tokens, option)
    tokens = remove_option(tokens, "--csv", takes_value=False)
    tokens = remove_option(tokens, "--force-overwrite", takes_value=False)
    return tokens


def privileged_ncu_command(command: Sequence[str]) -> List[str]:
    """Run NCU as root while retaining the caller's CUDA visibility/env.

    Hopper performance-counter access is commonly restricted to root on
    shared machines.  ``-E`` is intentional: CUDA_VISIBLE_DEVICES and the
    caller's CUDA library environment must describe the same logical device
    to NCU and to the application.
    """

    return ["sudo", "-n", "-E", *command]


def has_counter_permission_error(raw_csv: Path, log_text: str) -> bool:
    text = log_text
    if raw_csv.is_file():
        text += "\n" + raw_csv.read_text(encoding="utf-8", errors="replace")
    return "ERR_NVGPUCTRPERM" in text or "does not have permission to access NVIDIA GPU Performance Counters" in text


def has_application_busy_error(raw_csv: Path, log_text: str) -> bool:
    text = log_text
    if raw_csv.is_file():
        text += "\n" + raw_csv.read_text(encoding="utf-8", errors="replace")
    return "CUDA-capable device(s) is/are busy or unavailable" in text or "device(s) is/are busy" in text


def read_raw_csv(path: Path) -> Tuple[List[str], Dict[str, str], List[Dict[str, str]]]:
    if not path.is_file() or path.stat().st_size == 0:
        return [], {}, []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))
    header_index = None
    for index, row in enumerate(rows):
        if "Kernel Name" in row or "Kernel" in row or "Name" in row:
            header_index = index
            break
    if header_index is None:
        return [], {}, []
    header = [cell.strip() for cell in rows[header_index]]
    units: Dict[str, str] = {}
    unit_index = header_index + 1
    if unit_index < len(rows):
        unit_row = rows[unit_index]
        units = {
            name: unit_row[index].strip() if index < len(unit_row) else ""
            for index, name in enumerate(header)
        }
    data: List[Dict[str, str]] = []
    for raw in rows[unit_index + 1 :]:
        if not raw:
            continue
        padded = list(raw) + [""] * max(0, len(header) - len(raw))
        row = {name: padded[index].strip() for index, name in enumerate(header)}
        kernel = row.get("Kernel Name") or row.get("Kernel") or row.get("Name")
        if kernel:
            data.append(row)
    return header, units, data


def launch_key(row: Mapping[str, str], ordinal: int, occurrence: int) -> Tuple[str, ...]:
    launch_id = row.get("ID") or row.get("Invocations") or "ordinal:{}".format(ordinal)
    kernel = row.get("Kernel Name") or row.get("Kernel") or row.get("Name") or ""
    return (
        str(launch_id),
        row.get("Device", ""),
        row.get("Context", ""),
        row.get("Stream", ""),
        kernel,
        str(occurrence),
    )


def merge_group_csvs(group_records: Sequence[Mapping[str, Any]], report_metrics: Sequence[str], output: Path) -> int:
    columns: List[str] = []
    units: Dict[str, str] = {}
    merged: "OrderedDict[Tuple[str, ...], Dict[str, str]]" = OrderedDict()

    for record in group_records:
        raw_path = Path(record["raw_csv"])
        header, raw_units, rows = read_raw_csv(raw_path)
        if not header:
            continue
        for column in header:
            if column in FORBIDDEN_MEASURED_CLOCK_METRICS:
                continue
            if column not in columns:
                columns.append(column)
                units[column] = raw_units.get(column, "")
        occurrences: Dict[Tuple[str, ...], int] = {}
        for ordinal, row in enumerate(rows):
            base = launch_key(row, ordinal, 0)
            occurrence = occurrences.get(base, 0)
            occurrences[base] = occurrence + 1
            key = launch_key(row, ordinal, occurrence)
            if key not in merged:
                merged[key] = {}
            merged[key].update(
                {
                    column: value
                    for column, value in row.items()
                    if value != "" and column not in FORBIDDEN_MEASURED_CLOCK_METRICS
                }
            )

    for metric in report_metrics:
        if metric in FORBIDDEN_MEASURED_CLOCK_METRICS:
            continue
        if metric not in columns:
            columns.append(metric)
            units[metric] = ""

    if not columns or not merged:
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerow([units.get(column, "") for column in columns])
        for row in merged.values():
            writer.writerow(
                [
                    row.get(column, "") if row.get(column, "") != "" else ("N/A" if column in report_metrics else "")
                    for column in columns
                ]
            )
    return len(merged)


def extract_pass_count(rows: Sequence[Mapping[str, str]], log_text: str) -> Optional[int]:
    values = [parse_float(row.get("profiler__replayer_passes")) for row in rows]
    values = [int(round(value)) for value in values if value is not None]
    if values:
        return max(values)
    matches = re.findall(r"(?i)(\d+)\s+pass(?:es)?\b", log_text)
    return max((int(value) for value in matches), default=None)


def lock_status(gpc_hz: Optional[float], target_mhz: float) -> str:
    if gpc_hz is None:
        return "N/A"
    return "maintained" if abs(gpc_hz / 1e6 - target_mhz) <= GPC_LOCK_TOLERANCE_MHZ else "not_maintained"


def snapshot_lock_status(
    before: Mapping[str, str], after: Mapping[str, str], target_mhz: float
) -> str:
    values = [parse_float(before.get("gpc_mhz")), parse_float(after.get("gpc_mhz"))]
    if any(value is None for value in values):
        return "N/A"
    return (
        "maintained"
        if all(abs(value - target_mhz) <= 0.5 for value in values if value is not None)
        else "not_maintained"
    )


def build_summary(rows: Sequence[Mapping[str, str]], status: str, returncode: Any, pass_count: Any, error: str, target_mhz: float) -> Dict[str, Any]:
    gpc_hz = mean_metric(rows, RUNTIME_CLOCK_METRICS[0])
    sm_hz = mean_metric(rows, RUNTIME_CLOCK_METRICS[1])
    lts_hz = mean_metric(rows, RUNTIME_CLOCK_METRICS[2])
    dram_hz = mean_metric(rows, RUNTIME_CLOCK_METRICS[3])
    return {
        "status": status,
        "returncode": returncode,
        "pass_count": pass_count,
        "launch_count": len(rows),
        "duration_ns_avg": mean_metric(rows, "gpu__time_duration.sum"),
        "sm_cycles_avg": mean_metric(rows, "sm__cycles_elapsed.avg"),
        "gpc_clock_hz_avg": gpc_hz,
        "sm_clock_hz_avg": sm_hz,
        "lts_clock_hz_avg": lts_hz,
        "dram_clock_hz_avg": dram_hz,
        "icnt_clock_hz": "N/A",
        "gpc_lock_status": lock_status(gpc_hz, target_mhz),
        "l1_hit_rate_pct_avg": mean_metric(rows, "l1tex__t_sector_hit_rate.pct"),
        "l1_global_load_hit_rate_pct_avg": mean_metric(rows, "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"),
        "l1_global_store_hit_rate_pct_avg": mean_metric(rows, "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct"),
        "l1_load_sectors_sum": sum_metric(rows, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
        "l1_store_sectors_sum": sum_metric(rows, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"),
        "l2_hit_rate_pct_avg": mean_metric(rows, "lts__t_sector_hit_rate.pct"),
        "l2_read_hit_rate_pct_avg": mean_metric(rows, "lts__t_sector_op_read_hit_rate.pct"),
        "l2_write_hit_rate_pct_avg": mean_metric(rows, "lts__t_sector_op_write_hit_rate.pct"),
        "l2_read_sectors_sum": sum_metric(rows, "lts__t_sectors_op_read.sum"),
        "l2_write_sectors_sum": sum_metric(rows, "lts__t_sectors_op_write.sum"),
        "l2_bytes_sum": sum_metric(rows, "lts__t_bytes.sum"),
        "dram_read_bytes_sum": sum_metric(rows, "dram__bytes_read.sum"),
        "dram_write_bytes_sum": sum_metric(rows, "dram__bytes_write.sum"),
        "dram_read_bytes_per_second_avg": mean_metric(rows, "dram__bytes_read.sum.per_second"),
        "dram_write_bytes_per_second_avg": mean_metric(rows, "dram__bytes_write.sum.per_second"),
        "dram_throughput_pct_avg": mean_metric(rows, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "instructions_sum": sum_metric(rows, "sm__inst_executed.sum"),
        "warps_launched_sum": sum_metric(rows, "sm__warps_launched.sum"),
        "occupancy_pct_avg": mean_metric(rows, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        "error": error,
    }


def spread_percent(records: Sequence[Mapping[str, Any]], field: str) -> str:
    values = [parse_float(record.get(field)) for record in records if record.get("status") == "passed"]
    values = [value for value in values if value is not None]
    if not values:
        return "N/A"
    average = sum(values) / len(values)
    if average == 0:
        return "N/A"
    return "{:.6f}".format((max(values) - min(values)) / average * 100.0)


def write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt_optional(row.get(field)) for field in SUMMARY_FIELDS})


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--merged-csv", required=True)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--runner-log", required=True)
    parser.add_argument("--run-args", default="")
    parser.add_argument("--ncu-run-args", default="")
    parser.add_argument("--extra-metrics", default="")
    parser.add_argument("--kernel-name", default=".*")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--cache-control", choices=["all", "none"], default="all")
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--visible-devices", default="")
    parser.add_argument(
        "--ncu-devices",
        default="",
        help="CUDA logical device ordinals to profile; all visible devices still run the application",
    )
    parser.add_argument("--target-gpc-mhz", type=float, default=DEFAULT_GPC_CLOCK_MHZ)
    parser.add_argument("--chip", default=DEFAULT_CHIP)
    parser.add_argument(
        "--clock-lock-mode",
        choices=["internal", "external"],
        default="internal",
        help="internal locks/resets the selected GPU; external requires the caller to keep the lock",
    )
    parser.add_argument("--timeout", default="")
    parser.add_argument(
        "--sudo-mode",
        choices=["auto", "never", "always"],
        default=os.environ.get("NCU_SUDO", "auto"),
        help="run NCU as root always, never, or retry as root on ERR_NVGPUCTRPERM",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def group_plan(group_names: Sequence[str], extra_metrics: Sequence[str], available: set[str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    plan: Dict[str, Dict[str, Any]] = OrderedDict()
    all_report_metrics: List[str] = []
    for group in group_names:
        if group not in GROUP_METRICS:
            raise ValueError("unknown NCU metric group {!r}; available: {}".format(group, ", ".join(GROUP_METRICS)))
        requested = list(GROUP_METRICS[group])
        if group == "execution":
            requested.extend(extra_metrics)
        requested = [
            metric
            for metric in unique(requested)
            if metric not in FORBIDDEN_MEASURED_CLOCK_METRICS
        ]
        available_metrics = [metric for metric in requested if metric in available]
        unavailable_metrics = [metric for metric in requested if metric not in available]
        plan[group] = {
            "requested_metrics": requested,
            "available_metrics": available_metrics,
            "unavailable_metrics": unavailable_metrics,
        }
        all_report_metrics.extend(requested)
    return plan, unique(all_report_metrics)


def ncu_command(args: argparse.Namespace, group: str, metrics: Sequence[str], raw_csv: Path) -> List[str]:
    command = [
        args.ncu,
        "--target-processes",
        "all",
        *sanitize_ncu_args(args.ncu_run_args),
        "--clock-control",
        "none",
        "--cache-control",
        args.cache_control,
        "--csv",
        "--page",
        "raw",
        "--force-overwrite",
        "--log-file",
        str(raw_csv),
        "--metrics",
        ",".join(metrics),
    ]
    if args.ncu_devices:
        command.extend(["--devices", args.ncu_devices])
    kernel_name = args.kernel_name.strip()
    if kernel_name and kernel_name != ".*":
        if not kernel_name.startswith("regex:"):
            kernel_name = "regex:" + kernel_name
        command.extend(["--kernel-name-base", "demangled", "--kernel-name", kernel_name])
    command.extend(["./" + Path(args.run_dir).joinpath("__NCU_BINARY__").name, *shlex.split(args.run_args)])
    return command


def replace_binary(command: List[str], binary: str) -> List[str]:
    # The sentinel keeps the argument construction independent of the binary
    # basename and avoids shell interpolation of RUN_ARGS.
    return [binary if token in {"__NCU_BINARY__", "./__NCU_BINARY__"} else token for token in command]


def run_profile(args: argparse.Namespace, plan: Mapping[str, Mapping[str, Any]], report_metrics: Sequence[str], manifest: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    run_dir = Path(args.run_dir).resolve()
    group_root = run_dir / "ncu_groups"
    group_root.mkdir(parents=True, exist_ok=True)
    timeout_seconds = parse_duration(args.timeout)
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    group_rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    locked = False
    lock_error = ""

    try:
        supported = query_supported_gpc_clocks(args.gpu_index)
        manifest["supported_gpc_clocks_mhz"] = supported
        if int(round(args.target_gpc_mhz)) not in supported:
            raise RuntimeError(
                "requested GPC clock {:.0f} MHz is not supported; supported values include {}".format(
                    args.target_gpc_mhz, ", ".join(str(value) for value in supported)
                )
            )

        if args.clock_lock_mode == "internal":
            lock_command = ["sudo", "nvidia-smi", "-i", args.gpu_index, "-lgc", "{0:.0f},{0:.0f}".format(args.target_gpc_mhz)]
            lock_result = run_capture(lock_command, check=False)
            manifest["clock_lock_command"] = shlex.join(lock_command)
            manifest["clock_lock_output"] = lock_result.stdout.strip()
            if lock_result.returncode != 0:
                raise RuntimeError("GPC clock lock failed: {}".format(lock_result.stdout.strip()))
            locked = True
            manifest["clock_lock_status"] = "applied"
        else:
            manifest["clock_lock_command"] = "external caller lock"
            manifest["clock_lock_status"] = "external_prelocked"

        for group, specification in plan.items():
            group_dir = group_root / group
            group_dir.mkdir(parents=True, exist_ok=True)
            raw_csv = group_dir / "ncu.csv"
            group_log = group_dir / "run.log"
            before = query_clock_snapshot(args.gpu_index)
            available_metrics = list(specification["available_metrics"])
            start = time.monotonic()
            status = "skipped_unavailable"
            returncode: Any = "N/A"
            error = ""
            rows: List[Dict[str, str]] = []
            pass_count: Any = "N/A"
            attempt_count = 0
            invocation_mode = "N/A"
            final_command = "N/A"
            permission_fallback = False
            transient_retry = False

            if not available_metrics:
                error = "no requested metrics are available"
            elif deadline is not None and time.monotonic() >= deadline:
                status = "timeout"
                error = "case timeout reached before group"
            else:
                base_command = ncu_command(args, group, available_metrics, raw_csv)
                binary = str(run_dir / args.binary_name)
                base_command = replace_binary(base_command, binary)
                remaining = None if deadline is None else max(1.0, deadline - time.monotonic())
                if args.sudo_mode == "always":
                    attempts = [("sudo", privileged_ncu_command(base_command))]
                else:
                    attempts = [("user", base_command)]

                for attempt_index, (candidate_mode, command) in enumerate(attempts):
                    attempt_count += 1
                    invocation_mode = candidate_mode
                    final_command = shlex.join(command)
                    try:
                        with group_log.open("w", encoding="utf-8") as log:
                            completed = subprocess.run(
                                command,
                                cwd=str(run_dir),
                                env=profile_environment(args),
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=remaining,
                                check=False,
                            )
                        returncode = completed.returncode
                        _, _, rows = read_raw_csv(raw_csv)
                        log_text = group_log.read_text(encoding="utf-8", errors="replace") if group_log.exists() else ""
                        pass_count = extract_pass_count(rows, log_text)
                        if completed.returncode == 0 and rows:
                            status = "passed"
                            break

                        permission_error = has_counter_permission_error(raw_csv, log_text)
                        if args.sudo_mode == "auto" and not permission_fallback and permission_error:
                            permission_fallback = True
                            user_error_csv = group_dir / "ncu.user-error.csv"
                            if raw_csv.is_file():
                                shutil.copyfile(raw_csv, user_error_csv)
                            user_log = group_dir / "run.user.log"
                            if group_log.is_file():
                                shutil.copyfile(group_log, user_log)
                            attempts.append(("sudo", privileged_ncu_command(base_command)))
                            continue

                        busy_error = has_application_busy_error(raw_csv, log_text)
                        if busy_error and not transient_retry:
                            # A concurrent CUDA context can disappear between
                            # the preflight and application attach. Retry this
                            # group once without changing its metric set or
                            # cache/clock policy; other groups remain intact.
                            transient_retry = True
                            time.sleep(1.0)
                            attempts.append((candidate_mode, command))
                            continue

                        status = "failed"
                        if completed.returncode == 0:
                            error = "NCU succeeded but produced no kernel rows"
                        elif permission_error:
                            error = "NCU performance-counter permission denied (ERR_NVGPUCTRPERM)"
                        elif busy_error:
                            error = "application reported CUDA device busy or unavailable"
                        else:
                            error = "NCU exit code {}".format(completed.returncode)
                        break
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                        returncode = 124
                        error = "group timeout"
                        if raw_csv.exists():
                            _, _, rows = read_raw_csv(raw_csv)
                        break
                    except OSError as exc:
                        status = "failed"
                        returncode = 127
                        error = str(exc)
                        break

            after = query_clock_snapshot(args.gpu_index)
            summary = build_summary(rows, status, returncode, pass_count, error, args.target_gpc_mhz)
            summary.update(
                {
                    "group": group,
                    "raw_csv": str(raw_csv),
                    "run_log": str(group_log),
                    "requested_metrics": list(specification["requested_metrics"]),
                    "available_metrics": available_metrics,
                    "unavailable_metrics": list(specification["unavailable_metrics"]),
                    "attempt_count": attempt_count,
                    "ncu_invocation_mode": invocation_mode,
                    "ncu_command": final_command,
                    "permission_fallback": permission_fallback,
                    "transient_retry": transient_retry,
                    "clock_snapshot_before": before,
                    "clock_snapshot_after": after,
                    "external_gpc_lock_status": snapshot_lock_status(
                        before, after, args.target_gpc_mhz
                    ),
                    "elapsed_seconds": time.monotonic() - start,
                }
            )
            records.append(summary)
            group_rows.append(summary)
            print(
                "[ncu-group] {} status={} passes={} launches={} returncode={}".format(
                    group, status, pass_count, len(rows), returncode
                ),
                flush=True,
            )

        successful_records = [record for record in records if record["status"] == "passed"]
        if successful_records:
            first_log = Path(successful_records[0]["run_log"])
            if first_log.exists():
                Path(args.stats_file).write_text(first_log.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        merged_rows = merge_group_csvs(records, report_metrics, Path(args.merged_csv))
        manifest["merged_launch_count"] = merged_rows
        manifest["cross_group_spread_pct"] = {
            "duration_ns_avg": spread_percent(records, "duration_ns_avg"),
            "sm_cycles_avg": spread_percent(records, "sm_cycles_avg"),
            "gpc_clock_hz_avg": spread_percent(records, "gpc_clock_hz_avg"),
            "sm_clock_hz_avg": spread_percent(records, "sm_clock_hz_avg"),
            "lts_clock_hz_avg": spread_percent(records, "lts_clock_hz_avg"),
            "dram_clock_hz_avg": spread_percent(records, "dram_clock_hz_avg"),
        }
        manifest["external_gpc_lock_snapshots_verified"] = bool(records) and all(
            record.get("external_gpc_lock_status") == "maintained"
            for record in records
        )
        manifest["runtime_gpc_lock_verified"] = (
            bool(successful_records)
            and all(record.get("gpc_lock_status") == "maintained" for record in successful_records)
        )
        # Keep the historical key as the strict runtime-counter result.
        manifest["gpc_lock_verified"] = manifest["runtime_gpc_lock_verified"]
        return group_rows, all(record["status"] in {"passed", "skipped_unavailable"} for record in records)
    finally:
        if locked:
            reset_command = ["sudo", "nvidia-smi", "-i", args.gpu_index, "-rgc"]
            reset_result = run_capture(reset_command, check=False)
            manifest["clock_reset_command"] = shlex.join(reset_command)
            manifest["clock_reset_output"] = reset_result.stdout.strip()
            manifest["clock_reset_status"] = "reset" if reset_result.returncode == 0 else "failed"
            if reset_result.returncode != 0:
                lock_error = "GPC clock reset failed: {}".format(reset_result.stdout.strip())
                manifest["clock_reset_error"] = lock_error
        elif args.clock_lock_mode == "external":
            manifest["clock_reset_status"] = "not_applicable_external"


def profile_environment(args: argparse.Namespace) -> Dict[str, str]:
    environment = os.environ.copy()
    if args.visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    return environment


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    parser.add_argument("--binary-name", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    merged_csv = Path(args.merged_csv).resolve()
    stats_file = Path(args.stats_file).resolve()
    runner_log = Path(args.runner_log).resolve()
    group_names = [name.strip() for name in args.groups.split(",") if name.strip()]
    raw_extra_metrics = parse_metric_list(args.extra_metrics)
    extra_metrics = [
        metric
        for metric in raw_extra_metrics
        if metric not in FORBIDDEN_MEASURED_CLOCK_METRICS
    ]
    manifest: Dict[str, Any] = {
        "workflow_version": WORKFLOW_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ncu": str(Path(args.ncu).resolve()),
        "run_dir": str(run_dir),
        "binary": str(run_dir / args.binary_name),
        "run_args": args.run_args,
        "ncu_run_args": args.ncu_run_args,
        "ncu_sudo_mode": args.sudo_mode,
        "cache_control": args.cache_control,
        "clock_control": "none",
        "gpu_index": args.gpu_index,
        "ncu_devices": args.ncu_devices or "all_visible",
        "visible_devices": args.visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "chip": args.chip,
        "target_gpc_clock_mhz": args.target_gpc_mhz,
        "clock_lock_mode": args.clock_lock_mode,
        "icnt_clock_hz": "N/A",
        "icnt_clock_status": "unavailable_no_suitable_ncu_counter",
        "requested_groups": group_names,
        "extra_metrics": extra_metrics,
        "forbidden_metrics_ignored": [
            metric
            for metric in raw_extra_metrics
            if metric in FORBIDDEN_MEASURED_CLOCK_METRICS
        ],
        "report_metrics": [],
        "groups": {},
    }
    manifest_path = run_dir / "ncu_groups" / "manifest.json"
    summary_path = run_dir / "ncu_group_summary.csv"
    runner_log.parent.mkdir(parents=True, exist_ok=True)

    output_lines: List[str] = []
    result_code = 1
    try:
        available, _ = query_metric_catalog(args.ncu, args.chip)
        plan, report_metrics = group_plan(group_names, extra_metrics, available)
        manifest["report_metrics"] = report_metrics
        manifest["metric_catalog_check"] = "passed"
        manifest["metric_catalog_available_count"] = len(available)
        for group, specification in plan.items():
            manifest["groups"][group] = {
                "requested_metrics": specification["requested_metrics"],
                "available_metrics": specification["available_metrics"],
                "unavailable_metrics": specification["unavailable_metrics"],
            }

        if args.dry_run:
            for group, specification in plan.items():
                available_metrics = specification["available_metrics"]
                if not available_metrics:
                    output_lines.append("[dry-run] {} skipped: no available metrics".format(group))
                    continue
                raw_csv = run_dir / "ncu_groups" / group / "ncu.csv"
                command = replace_binary(
                    ncu_command(args, group, available_metrics, raw_csv),
                    str(run_dir / args.binary_name),
                )
                output_lines.append("[dry-run] {}: {}".format(group, shlex.join(command)))
                if args.sudo_mode in {"auto", "always"}:
                    output_lines.append(
                        "[dry-run] {} sudo fallback: {}".format(
                            group, shlex.join(privileged_ncu_command(command))
                        )
                    )
            result_code = 0
        else:
            manifest["device"] = query_gpu_info(args.gpu_index)
            records, ok = run_profile(args, plan, report_metrics, manifest)
            manifest["groups"].update(
                {
                    record["group"]: {
                        key: value
                        for key, value in record.items()
                        if key not in {"raw_csv", "run_log"}
                    }
                    for record in records
                }
            )
            write_summary(summary_path, records)
            clock_cleanup_ok = (
                manifest.get("clock_reset_status") == "reset"
                if args.clock_lock_mode == "internal"
                else manifest.get("clock_reset_status") == "not_applicable_external"
            )
            result_code = 0 if ok and clock_cleanup_ok else 1
    except Exception as exc:  # Preserve a manifest explaining preflight failures.
        manifest["workflow_status"] = "failed"
        manifest["error"] = str(exc)
        eprint("[ncu-group] ERROR: {}".format(exc))
        result_code = 1
    finally:
        if output_lines:
            print("\n".join(output_lines))
        manifest.setdefault("workflow_status", "passed" if result_code == 0 else "failed")
        write_manifest(manifest_path, manifest)
        runner_log.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
