#!/usr/bin/env python3
import argparse
import csv
import math
import pathlib
import re
import sys

SIM_PATTERNS = {
    "gpu_ipc": re.compile(r"gpu_ipc =\s*([0-9.+-eE]+)"),
    "gpu_sim_cycle": re.compile(r"gpu_sim_cycle =\s*([0-9.+-eE]+)"),
    "gpu_sim_insn": re.compile(r"gpu_sim_insn =\s*([0-9.+-eE]+)"),
    "gpu_tot_sim_cycle": re.compile(r"gpu_tot_sim_cycle =\s*([0-9.+-eE]+)"),
    "gpu_tot_sim_insn": re.compile(r"gpu_tot_sim_insn =\s*([0-9.+-eE]+)"),
    "gpu_tot_ipc": re.compile(r"gpu_tot_ipc =\s*([0-9.+-eE]+)"),
    "gpgpu_n_load_insn": re.compile(r"gpgpu_n_load_insn =\s*([0-9.+-eE]+)"),
    "gpgpu_n_store_insn": re.compile(r"gpgpu_n_store_insn =\s*([0-9.+-eE]+)"),
    "gpgpu_n_mem_read_global": re.compile(r"gpgpu_n_mem_read_global =\s*([0-9.+-eE]+)"),
    "gpgpu_n_mem_write_global": re.compile(
        r"gpgpu_n_mem_write_global =\s*([0-9.+-eE]+)"
    ),
    "gpu_mem_bw": re.compile(r"L2_BW\s*=\s*([0-9.+-eE]+)"),
    "gpu_tot_mem_bw": re.compile(
        r"(?:gpu_tot_mem_bw|L2_BW_total)\s*=\s*([0-9.+-eE]+)"
    ),
    "L1D_total_cache_accesses": re.compile(
        r"L1D_total_cache_accesses =\s*([0-9.+-eE]+)"
    ),
    "L1D_total_cache_hits": re.compile(r"L1D_total_cache_hits =\s*([0-9.+-eE]+)"),
    "L1D_total_cache_misses": re.compile(r"L1D_total_cache_misses =\s*([0-9.+-eE]+)"),
    "L1D_total_cache_miss_rate": re.compile(
        r"L1D_total_cache_miss_rate =\s*([0-9.+-eE]+)"
    ),
    "L2_total_cache_accesses": re.compile(r"L2_total_cache_accesses =\s*([0-9.+-eE]+)"),
    "L2_total_cache_hits": re.compile(r"L2_total_cache_hits =\s*([0-9.+-eE]+)"),
    "L2_total_cache_misses": re.compile(r"L2_total_cache_misses =\s*([0-9.+-eE]+)"),
    "L2_total_cache_miss_rate": re.compile(
        r"L2_total_cache_miss_rate =\s*([0-9.+-eE]+)"
    ),
    "total_dram_reads": re.compile(r"total dram reads =\s*([0-9.+-eE]+)"),
    "total_dram_writes": re.compile(r"total dram writes =\s*([0-9.+-eE]+)"),
    "gpu_shared_mem_bank_conflict_reads": re.compile(
        r"gpu_shared_mem_bank_conflict_reads\s*=\s*([0-9.+-eE]+)"
    ),
    "gpu_shared_mem_bank_conflict_writes": re.compile(
        r"gpu_shared_mem_bank_conflict_writes\s*=\s*([0-9.+-eE]+)"
    ),
    "icnt_traffic_injected": re.compile(r"icnt_traffic_injected\s*=\s*([0-9.+-eE]+)"),
    "icnt_traffic_received": re.compile(r"icnt_traffic_received\s*=\s*([0-9.+-eE]+)"),
    "icnt_throughput_avg": re.compile(r"icnt_throughput_avg\s*=\s*([0-9.+-eE]+)"),
    "icnt_total_pkts_mem_to_simt": re.compile(
        r"icnt_total_pkts_mem_to_simt\s*=\s*([0-9.+-eE]+)"
    ),
    "icnt_total_pkts_simt_to_mem": re.compile(
        r"icnt_total_pkts_simt_to_mem\s*=\s*([0-9.+-eE]+)"
    ),
    "Req_Network_injected_packets_per_cycle": re.compile(
        r"Req_Network_injected_packets_per_cycle\s*=\s*([0-9.+-eE]+)"
    ),
    "Reply_Network_injected_packets_per_cycle": re.compile(
        r"Reply_Network_injected_packets_per_cycle\s*=\s*([0-9.+-eE]+)"
    ),
}
SIM_L1_PATTERNS = {
    "L1_global_read_hits": re.compile(
        r"^\s*Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*([0-9.+-eE]+)\s*$"
    ),
    "L1_global_read_misses": re.compile(
        r"^\s*Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[MISS\]\s*=\s*([0-9.+-eE]+)\s*$"
    ),
    "L1_global_read_sector_misses": re.compile(
        r"^\s*Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[SECTOR_MISS\]\s*=\s*([0-9.+-eE]+)\s*$"
    ),
    "L1_global_read_accesses": re.compile(
        r"^\s*Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*([0-9.+-eE]+)\s*$"
    ),
}
SECTOR_PAT = re.compile(r"dram_reads\[(\d+)\]\s*=\s*([0-9.+-eE]+)")
WRITE_PAT = re.compile(r"dram_writes\[(\d+)\]\s*=\s*([0-9.+-eE]+)")
KERNEL_NAME_PAT = re.compile(r"^kernel_name =\s*(.+?)\s*$")
KERNEL_UID_PAT = re.compile(r"^kernel_launch_uid =\s*([0-9.+-eE]+)\s*$")
CSV_HEADER_FIELD_PAT = re.compile(r"^[A-Za-z_][A-Za-z0-9_()./%-]*$")
TAIL_KV_PAT = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*),(?P<value>[^,\s][^,]*)\s*$")
IGNORED_STATS_PATTERNS = (
    "not ready",
    "Dirty lines flushed from L2",
    "Flushed L2",
)
APP_ROW_HEADER_NAMES = {"case", "kernel", "kernel_name", "launch_id", "name"}
APP_LATENCY_PATTERNS = {
    "fadd_raw_cycles_per_op": re.compile(r"FADD chain latency:\s*([0-9.+-eE]+)"),
    "iadd_raw_cycles_per_op": re.compile(r"IADD chain latency:\s*([0-9.+-eE]+)"),
    "fadd_adjusted_cycles_per_op": re.compile(
        r"FADD adjusted latency:\s*([0-9.+-eE]+)"
    ),
}

SIM_KIND_COLUMNS = {
    "dep_chain": [
        "kernel_name",
        "kernel_launch_uid",
        "gpu_ipc",
        "gpu_sim_cycle",
        "gpu_sim_insn",
        "gpu_tot_sim_cycle",
        "gpu_tot_sim_insn",
        "gpu_tot_ipc",
    ],
    "l1": [
        "kernel_name",
        "kernel_launch_uid",
        "gpu_ipc",
        "gpu_sim_cycle",
        "L1_global_read_accesses",
        "L1_global_read_hits",
        "L1_global_read_misses",
        "L1_global_read_sector_misses",
        "L2_total_cache_accesses",
        "L2_total_cache_hits",
        "L2_total_cache_misses",
        "total_dram_reads",
        "total_dram_writes",
    ],
    "l2": [
        "kernel_name",
        "kernel_launch_uid",
        "gpu_ipc",
        "gpu_sim_cycle",
        "L2_total_cache_accesses",
        "L2_total_cache_hits",
        "L2_total_cache_misses",
        "total_dram_reads",
        "total_dram_writes",
    ],
    "l2_parallel": [
        "kernel_name",
        "kernel_launch_uid",
        "gpu_ipc",
        "gpu_sim_cycle",
        "L2_total_cache_accesses",
        "L2_total_cache_hits",
        "L2_total_cache_misses",
        "total_dram_reads",
        "total_dram_writes",
    ],
    "hbm": [
        "kernel_name",
        "kernel_launch_uid",
        "gpu_ipc",
        "gpu_sim_cycle",
        "gpu_sim_insn",
        "gpu_tot_sim_cycle",
        "gpu_tot_sim_insn",
        "gpu_tot_ipc",
        "gpu_mem_bw",
        "gpu_tot_mem_bw",
        "L2_total_cache_accesses",
        "L2_total_cache_hits",
        "L2_total_cache_misses",
        "total_dram_reads",
        "total_dram_writes",
    ],
    "smem": [
        "kernel_name",
        "kernel_launch_uid",
        "gpu_ipc",
        "gpu_sim_cycle",
        "gpu_shared_mem_bank_conflict_reads",
        "gpu_shared_mem_bank_conflict_writes",
        "total_dram_reads",
        "total_dram_writes",
    ],
    "validation_single": [
        "kernel_name",
        "kernel_launch_uid",
        "gpu_ipc",
        "gpu_sim_cycle",
        "L1D_total_cache_accesses",
        "L1D_total_cache_hits",
        "L1D_total_cache_misses",
        "L2_total_cache_accesses",
        "L2_total_cache_hits",
        "L2_total_cache_misses",
        "total_dram_reads",
        "total_dram_writes",
    ],
    "nvlink": [
        "kernel_name",
        "kernel_launch_uid",
        "icnt_total_pkts_simt_to_mem",
        "icnt_total_pkts_mem_to_simt",
        "Req_Network_injected_packets_per_cycle",
        "Reply_Network_injected_packets_per_cycle",
        "total_dram_reads",
        "total_dram_writes",
    ],
}

SIM_COLUMN_TYPES = {
    "kernel_name": "text",
    "kernel_launch_uid": "per_kernel",
    "gpu_ipc": "per_kernel",
    "gpu_sim_cycle": "per_kernel",
    "gpu_sim_insn": "per_kernel",
    "gpu_tot_sim_cycle": "cumulative",
    "gpu_tot_sim_insn": "cumulative",
    "gpu_tot_ipc": "cumulative",
    "gpu_mem_bw": "per_kernel",
    "gpu_tot_mem_bw": "cumulative",
    "L1_global_read_accesses": "cumulative",
    "L1_global_read_hits": "cumulative",
    "L1_global_read_misses": "cumulative",
    "L1_global_read_sector_misses": "cumulative",
    "L1D_total_cache_accesses": "cumulative",
    "L1D_total_cache_hits": "cumulative",
    "L1D_total_cache_misses": "cumulative",
    "L2_total_cache_accesses": "cumulative",
    "L2_total_cache_hits": "cumulative",
    "L2_total_cache_misses": "cumulative",
    "total_dram_reads": "cumulative",
    "total_dram_writes": "cumulative",
    "gpu_shared_mem_bank_conflict_reads": "cumulative",
    "gpu_shared_mem_bank_conflict_writes": "cumulative",
    "icnt_total_pkts_simt_to_mem": "cumulative",
    "icnt_total_pkts_mem_to_simt": "cumulative",
    "Req_Network_injected_packets_per_cycle": "per_kernel",
    "Reply_Network_injected_packets_per_cycle": "per_kernel",
}


def parse_sim(path: pathlib.Path):
    text = path.read_text(errors="ignore")
    values = {}
    for key, pat in SIM_PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            values[key] = matches[-1]
    dram_reads = sum(float(m.group(2)) for m in SECTOR_PAT.finditer(text))
    dram_writes = sum(float(m.group(2)) for m in WRITE_PAT.finditer(text))
    if dram_reads:
        values["dram_read_sectors"] = dram_reads
        values["dram_read_bytes"] = dram_reads * 32.0
    if dram_writes:
        values["dram_write_sectors"] = dram_writes
        values["dram_write_bytes"] = dram_writes * 32.0
    if "total_dram_reads" in values and "dram_read_bytes" not in values:
        values["dram_read_bytes"] = float(values["total_dram_reads"]) * 32.0
    if "total_dram_writes" in values and "dram_write_bytes" not in values:
        values["dram_write_bytes"] = float(values["total_dram_writes"]) * 32.0
    return values


def parse_sim_launches(path: pathlib.Path):
    launches = []
    current = None
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.rstrip()
        if any(pattern in line for pattern in IGNORED_STATS_PATTERNS):
            continue
        m_name = KERNEL_NAME_PAT.match(line)
        if m_name:
            if current and current.get("kernel_launch_uid") is not None:
                launches.append(current)
            current = {
                "kernel_name": m_name.group(1),
                "metrics": {},
            }
            continue
        if current is None:
            continue
        m_uid = KERNEL_UID_PAT.match(line)
        if m_uid:
            current["kernel_launch_uid"] = m_uid.group(1)
            continue
        for key, pat in {**SIM_PATTERNS, **SIM_L1_PATTERNS}.items():
            m = pat.match(line)
            if m:
                value = m.group(1)
                current["metrics"][key] = value
    if current:
        launches.append(current)
    return launches


def parse_csv_table_rows(path: pathlib.Path):
    rows = []
    header = None
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or "GPGPU-Sim" in line or " = " in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if header is None:
            if (
                len(parts) >= 3
                and parts[0] in APP_ROW_HEADER_NAMES
                and all(CSV_HEADER_FIELD_PAT.match(part) for part in parts)
            ):
                header = parts
            continue
        if len(parts) != len(header):
            continue
        if any(not part or " " in part for part in parts):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def parse_tail_kv_pairs(path: pathlib.Path):
    pairs = []
    lines = path.read_text(errors="ignore").splitlines()
    start = 0
    for idx, raw_line in enumerate(lines):
        if raw_line.strip() == "GPGPU-Sim: detected inactive GPU simulation thread":
            start = idx + 1
    for raw_line in lines[start:]:
        line = raw_line.strip()
        if not line or line == "GPGPU-Sim: *** exit detected ***":
            continue
        m = TAIL_KV_PAT.match(line)
        if not m:
            if pairs:
                break
            continue
        pairs.append((m.group("key"), m.group("value")))
    return pairs


def parse_app_latencies(path: pathlib.Path):
    values = {}
    if not path.exists():
        return values
    text = path.read_text(errors="ignore")
    for key, pattern in APP_LATENCY_PATTERNS.items():
        match = pattern.search(text)
        if match:
            values[key] = match.group(1)
    return values


def print_pairs(title, pairs):
    print(title)
    for key, value in pairs:
        print("{}: {}".format(key, value))


def print_prefixed_pairs(title, pairs, prefix):
    print(title)
    for key, value in pairs:
        print("{}{}: {}".format(prefix, key, value))


def parse_float(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def fmt_number(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return "{:.6f}".format(value)


def fmt_ratio(numer, denom):
    if numer is None or denom is None or denom == 0.0:
        return "n/a"
    return fmt_number(numer / denom)


def fmt_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return fmt_number(value)
    numeric = parse_float(value)
    if numeric is not None:
        return fmt_number(numeric)
    return value


def avg_value(values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / float(len(filtered))


def ratio_value(numer, denom):
    if numer is None or denom in (None, 0.0):
        return None
    return numer / denom


def launch_cycle_value(metrics, prev_metrics):
    launch_cycles = parse_float(metrics.get("gpu_sim_cycle"))
    if launch_cycles is not None:
        return launch_cycles
    if prev_metrics is None:
        return parse_float(metrics.get("gpu_tot_sim_cycle"))
    total_cycles = parse_float(metrics.get("gpu_tot_sim_cycle"))
    prev_total_cycles = parse_float(prev_metrics.get("gpu_tot_sim_cycle"))
    if total_cycles is None or prev_total_cycles is None:
        return total_cycles
    if total_cycles < prev_total_cycles:
        return total_cycles
    return total_cycles - prev_total_cycles


def native_total_cycles(metrics):
    elapsed = parse_float(metrics.get("elapsed_cycles_sm"))
    if elapsed is not None:
        return elapsed
    ipc = parse_float(metrics.get("ipc"))
    inst = parse_float(metrics.get("inst_executed"))
    if ipc is None or inst is None or ipc == 0.0:
        return None
    return inst / ipc


def map_rows_to_launches(launches, rows):
    if not launches or not rows:
        return {}
    if len(rows) == len(launches):
        return dict(enumerate(rows))
    interesting_name = launches[-1].get("kernel_name")
    interesting_indices = [
        idx
        for idx, launch in enumerate(launches)
        if launch.get("kernel_name") == interesting_name
    ]
    if len(rows) == len(interesting_indices):
        return {idx: row for idx, row in zip(interesting_indices, rows)}
    return {}


def delta_metric(metrics, prev_metrics, key):
    current = parse_float(metrics.get(key))
    if current is None:
        return None
    if prev_metrics is None:
        return current
    previous = parse_float(prev_metrics.get(key))
    if previous is None or current < previous:
        return current
    return current - previous


def print_launch(launch, row=None, row_prefix="APP_"):
    print("kernel_name: {}".format(launch.get("kernel_name", "n/a")))
    print("kernel_launch_uid: {}".format(launch.get("kernel_launch_uid", "n/a")))
    if row:
        for key, value in row.items():
            print("{}{}: {}".format(row_prefix, key, value))


def report_launch_metrics(kind: str, launch, prev_launch=None, row=None):
    metrics = launch["metrics"]
    prev_metrics = prev_launch["metrics"] if prev_launch else None
    print_launch(launch, row)
    if kind == "dep_chain":
        reported_insn = delta_metric(metrics, prev_metrics, "gpu_sim_insn")
        total_cycles = launch_cycle_value(metrics, prev_metrics)
        print("gpu_ipc: {}".format(fmt_value(metrics.get("gpu_ipc"))))
        print(
            "gpu_sim_insn_reported: {}".format(fmt_value(metrics.get("gpu_sim_insn")))
        )
        print("per_warp_inst: {}".format(fmt_number(reported_insn)))
        print("total_cycles: {}".format(fmt_number(total_cycles)))
        return
    if kind == "l1":
        total_cycles = launch_cycle_value(metrics, prev_metrics)
        l1_accesses = delta_metric(metrics, prev_metrics, "L1_global_read_accesses")
        l1_hits = delta_metric(metrics, prev_metrics, "L1_global_read_hits")
        l1_misses = delta_metric(metrics, prev_metrics, "L1_global_read_misses")
        l1_sector_misses = delta_metric(
            metrics, prev_metrics, "L1_global_read_sector_misses"
        )
        l2_accesses = delta_metric(metrics, prev_metrics, "L2_total_cache_accesses")
        l2_misses = delta_metric(metrics, prev_metrics, "L2_total_cache_misses")
        dram_read_bytes = delta_metric(metrics, prev_metrics, "dram_read_bytes")
        dram_write_bytes = delta_metric(metrics, prev_metrics, "dram_write_bytes")
        print("gpu_ipc: {}".format(fmt_value(metrics.get("gpu_ipc"))))
        print("total_cycles: {}".format(fmt_value(total_cycles)))
        print("L1_global_read_accesses: {}".format(fmt_value(l1_accesses)))
        print("L1_global_read_hits: {}".format(fmt_value(l1_hits)))
        print("L1_global_read_misses: {}".format(fmt_value(l1_misses)))
        print("L1_global_read_sector_misses: {}".format(fmt_value(l1_sector_misses)))
        print("L1_global_read_miss_rate: {}".format(fmt_ratio(l1_misses, l1_accesses)))
        print("L2_total_cache_accesses: {}".format(fmt_value(l2_accesses)))
        print("L2_total_cache_misses: {}".format(fmt_value(l2_misses)))
        print("L2_total_cache_miss_rate: {}".format(fmt_ratio(l2_misses, l2_accesses)))
        print("dram_read_bytes: {}".format(fmt_value(dram_read_bytes)))
        print("dram_write_bytes: {}".format(fmt_value(dram_write_bytes)))
        return
    if kind == "l2":
        total_cycles = launch_cycle_value(metrics, prev_metrics)
        l2_accesses = delta_metric(metrics, prev_metrics, "L2_total_cache_accesses")
        l2_misses = delta_metric(metrics, prev_metrics, "L2_total_cache_misses")
        dram_read_bytes = delta_metric(metrics, prev_metrics, "dram_read_bytes")
        dram_write_bytes = delta_metric(metrics, prev_metrics, "dram_write_bytes")
        print("gpu_ipc: {}".format(fmt_value(metrics.get("gpu_ipc"))))
        print("total_cycles: {}".format(fmt_value(total_cycles)))
        print("L2_total_cache_accesses: {}".format(fmt_value(l2_accesses)))
        print("L2_total_cache_misses: {}".format(fmt_value(l2_misses)))
        print("L2_total_cache_miss_rate: {}".format(fmt_ratio(l2_misses, l2_accesses)))
        print("dram_read_bytes: {}".format(fmt_value(dram_read_bytes)))
        print("dram_write_bytes: {}".format(fmt_value(dram_write_bytes)))
        return
    if kind == "hbm":
        total_cycles = launch_cycle_value(metrics, prev_metrics)
        l2_accesses = delta_metric(metrics, prev_metrics, "L2_total_cache_accesses")
        l2_misses = delta_metric(metrics, prev_metrics, "L2_total_cache_misses")
        dram_read_bytes = delta_metric(metrics, prev_metrics, "dram_read_bytes")
        dram_write_bytes = delta_metric(metrics, prev_metrics, "dram_write_bytes")
        app_requested_read_bytes = None
        if row is not None:
            app_requested_read_bytes = parse_float(
                row.get("requested_read_bytes") or row.get("APP_requested_read_bytes")
            )
        gpu_mem_bw = metrics.get("gpu_mem_bw")
        if gpu_mem_bw is None:
            gpu_mem_bw = metrics.get("gpu_tot_mem_bw")
        print("gpu_ipc: {}".format(fmt_value(metrics.get("gpu_ipc"))))
        print("total_cycles: {}".format(fmt_value(total_cycles)))
        print("dram_read_bytes: {}".format(fmt_value(dram_read_bytes)))
        print("dram_write_bytes: {}".format(fmt_value(dram_write_bytes)))
        print("gpu_mem_bw: {}".format(fmt_value(gpu_mem_bw)))
        print("gpu_tot_mem_bw: {}".format(fmt_value(metrics.get("gpu_tot_mem_bw"))))
        print("L2_total_cache_accesses: {}".format(fmt_value(l2_accesses)))
        print("L2_total_cache_misses: {}".format(fmt_value(l2_misses)))
        print("L2_total_cache_miss_rate: {}".format(fmt_ratio(l2_misses, l2_accesses)))
        print(
            "dram_read_bytes_over_APP_requested_read_bytes: {}".format(
                fmt_ratio(dram_read_bytes, app_requested_read_bytes)
            )
        )
        return
    if kind == "smem":
        total_cycles = launch_cycle_value(metrics, prev_metrics)
        bank_conflict_reads = delta_metric(
            metrics, prev_metrics, "gpu_shared_mem_bank_conflict_reads"
        )
        bank_conflict_writes = delta_metric(
            metrics, prev_metrics, "gpu_shared_mem_bank_conflict_writes"
        )
        dram_read_bytes = delta_metric(metrics, prev_metrics, "dram_read_bytes")
        dram_write_bytes = delta_metric(metrics, prev_metrics, "dram_write_bytes")
        print("gpu_ipc: {}".format(fmt_value(metrics.get("gpu_ipc"))))
        print("total_cycles: {}".format(fmt_value(total_cycles)))
        print(
            "gpu_shared_mem_bank_conflict_reads: {}".format(
                fmt_value(bank_conflict_reads)
            )
        )
        print(
            "gpu_shared_mem_bank_conflict_writes: {}".format(
                fmt_value(bank_conflict_writes)
            )
        )
        print("dram_read_bytes: {}".format(fmt_value(dram_read_bytes)))
        print("dram_write_bytes: {}".format(fmt_value(dram_write_bytes)))
        return
    if kind == "validation_single":
        l1_accesses = delta_metric(metrics, prev_metrics, "L1D_total_cache_accesses")
        l1_misses = delta_metric(metrics, prev_metrics, "L1D_total_cache_misses")
        l2_accesses = delta_metric(metrics, prev_metrics, "L2_total_cache_accesses")
        l2_misses = delta_metric(metrics, prev_metrics, "L2_total_cache_misses")
        dram_read_bytes = delta_metric(metrics, prev_metrics, "dram_read_bytes")
        dram_write_bytes = delta_metric(metrics, prev_metrics, "dram_write_bytes")
        print("gpu_ipc: {}".format(fmt_value(metrics.get("gpu_ipc"))))
        print("gpu_sim_cycle: {}".format(fmt_value(metrics.get("gpu_sim_cycle"))))
        print("L1D_total_cache_accesses: {}".format(fmt_value(l1_accesses)))
        print("L1D_total_cache_misses: {}".format(fmt_value(l1_misses)))
        print("L1D_total_cache_miss_rate: {}".format(fmt_ratio(l1_misses, l1_accesses)))
        print("L2_total_cache_accesses: {}".format(fmt_value(l2_accesses)))
        print("L2_total_cache_misses: {}".format(fmt_value(l2_misses)))
        print("L2_total_cache_miss_rate: {}".format(fmt_ratio(l2_misses, l2_accesses)))
        print("dram_read_bytes: {}".format(fmt_value(dram_read_bytes)))
        print("dram_write_bytes: {}".format(fmt_value(dram_write_bytes)))
        return
    if kind == "nvlink":
        simt_to_mem = delta_metric(metrics, prev_metrics, "icnt_total_pkts_simt_to_mem")
        mem_to_simt = delta_metric(metrics, prev_metrics, "icnt_total_pkts_mem_to_simt")
        dram_read_bytes = delta_metric(metrics, prev_metrics, "dram_read_bytes")
        dram_write_bytes = delta_metric(metrics, prev_metrics, "dram_write_bytes")
        print("icnt_total_pkts_simt_to_mem: {}".format(fmt_value(simt_to_mem)))
        print("icnt_total_pkts_mem_to_simt: {}".format(fmt_value(mem_to_simt)))
        print(
            "Req_Network_injected_packets_per_cycle: {}".format(
                fmt_value(metrics.get("Req_Network_injected_packets_per_cycle"))
            )
        )
        print(
            "Reply_Network_injected_packets_per_cycle: {}".format(
                fmt_value(metrics.get("Reply_Network_injected_packets_per_cycle"))
            )
        )
        print("dram_read_bytes: {}".format(fmt_value(dram_read_bytes)))
        print("dram_write_bytes: {}".format(fmt_value(dram_write_bytes)))
        return
    for key, value in sorted(metrics.items()):
        print("{}: {}".format(key, fmt_value(value)))


def iter_launch_groups(launches):
    start = 0
    while start < len(launches):
        end = start + 1
        kernel_uid = launches[start].get("kernel_launch_uid")
        while (
            end < len(launches)
            and launches[end].get("kernel_launch_uid") == kernel_uid
        ):
            end += 1
        yield start, end
        start = end


def print_launch_group_summary(kind: str, launches, start: int, end: int):
    if end - start < 2:
        return
    measured_indices = list(range(start + 1, end))
    if not measured_indices:
        return

    def launch_delta(idx, key):
        prev_metrics = launches[idx - 1]["metrics"] if idx > 0 else None
        return delta_metric(launches[idx]["metrics"], prev_metrics, key)

    if kind == "hbm":
        total_cycles = [launch_delta(idx, "gpu_sim_cycle") for idx in measured_indices]
        dram_read_bytes = [
            launch_delta(idx, "dram_read_bytes") for idx in measured_indices
        ]
        dram_write_bytes = [
            launch_delta(idx, "dram_write_bytes") for idx in measured_indices
        ]
    else:
        total_cycles = [launch_delta(idx, "gpu_sim_cycle") for idx in measured_indices]
        dram_read_bytes = [
            launch_delta(idx, "dram_read_bytes") for idx in measured_indices
        ]
        dram_write_bytes = [
            launch_delta(idx, "dram_write_bytes") for idx in measured_indices
        ]
    l1_hit_rates = []
    l2_hit_rates = []
    smem_conflict_reads = []
    smem_conflict_writes = []
    total_bw_bytes_per_cycle = []

    for idx in measured_indices:
        l1_accesses = launch_delta(idx, "L1_global_read_accesses")
        l1_hits = launch_delta(idx, "L1_global_read_hits")
        l2_accesses = launch_delta(idx, "L2_total_cache_accesses")
        l2_hits = launch_delta(idx, "L2_total_cache_hits")

        if kind == "hbm":
            cycle_value = launch_delta(idx, "gpu_sim_cycle")
            read_value = launch_delta(idx, "dram_read_bytes")
            write_value = launch_delta(idx, "dram_write_bytes")
            l2_hit_rates.append(ratio_value(l2_hits, l2_accesses))
            total_bw_bytes_per_cycle.append(
                ratio_value((read_value or 0.0) + (write_value or 0.0), cycle_value)
            )
        else:
            cycle_value = launch_delta(idx, "gpu_sim_cycle")
            read_value = launch_delta(idx, "dram_read_bytes")
            write_value = launch_delta(idx, "dram_write_bytes")
            l1_hit_rates.append(ratio_value(l1_hits, l1_accesses))
            l2_hit_rates.append(ratio_value(l2_hits, l2_accesses))
            smem_conflict_reads.append(
                launch_delta(idx, "gpu_shared_mem_bank_conflict_reads")
            )
            smem_conflict_writes.append(
                launch_delta(idx, "gpu_shared_mem_bank_conflict_writes")
            )
            total_bw_bytes_per_cycle.append(
                ratio_value(
                    (read_value or 0.0) + (write_value or 0.0),
                    cycle_value,
                )
            )

    print("kernel_group_summary")
    print("kernel_name: {}".format(launches[start].get("kernel_name", "n/a")))
    print("launch_count: {}".format(end - start))
    print("dropped_first_launches: 1")
    print("averaged_launches: {}".format(len(measured_indices)))
    print("avg_total_cycles: {}".format(fmt_value(avg_value(total_cycles))))

    if kind == "l1":
        print("avg_L1_global_read_hit_rate: {}".format(fmt_value(avg_value(l1_hit_rates))))
        print("avg_L2_total_cache_hit_rate: {}".format(fmt_value(avg_value(l2_hit_rates))))
        print("avg_dram_read_bytes: {}".format(fmt_value(avg_value(dram_read_bytes))))
        print("avg_dram_write_bytes: {}".format(fmt_value(avg_value(dram_write_bytes))))
        return
    if kind in {"l2", "l2_parallel"}:
        print("avg_L2_total_cache_hit_rate: {}".format(fmt_value(avg_value(l2_hit_rates))))
        print("avg_dram_read_bytes: {}".format(fmt_value(avg_value(dram_read_bytes))))
        print("avg_dram_write_bytes: {}".format(fmt_value(avg_value(dram_write_bytes))))
        return
    if kind == "hbm":
        print("avg_L2_total_cache_hit_rate: {}".format(fmt_value(avg_value(l2_hit_rates))))
        print("avg_dram_read_bytes: {}".format(fmt_value(avg_value(dram_read_bytes))))
        print("avg_dram_write_bytes: {}".format(fmt_value(avg_value(dram_write_bytes))))
        print(
            "avg_dram_total_bytes_per_cycle: {}".format(
                fmt_value(avg_value(total_bw_bytes_per_cycle))
            )
        )
        return
    if kind == "smem":
        print(
            "avg_gpu_shared_mem_bank_conflict_reads: {}".format(
                fmt_value(avg_value(smem_conflict_reads))
            )
        )
        print(
            "avg_gpu_shared_mem_bank_conflict_writes: {}".format(
                fmt_value(avg_value(smem_conflict_writes))
            )
        )
        return


def report_gpgpusim(kind: str, stats: pathlib.Path):
    launches = parse_sim_launches(stats)
    if launches:
        row_map = map_rows_to_launches(launches, parse_csv_table_rows(stats))
        columns = ["record_type"] + SIM_KIND_COLUMNS.get(
            kind, ["kernel_name", "kernel_launch_uid"]
        )
        app_columns = ordered_app_columns(
            [row_map[idx] for idx in sorted(row_map.keys()) if row_map.get(idx)]
        )
        columns.extend([col for col in app_columns if col not in columns])
        rows = []
        for idx, launch in enumerate(launches):
            metrics = launch["metrics"]
            row = {
                "record_type": "data",
                "kernel_name": launch.get("kernel_name", ""),
                "kernel_launch_uid": launch.get("kernel_launch_uid", ""),
            }
            for key in columns:
                if key in {"record_type", "kernel_name", "kernel_launch_uid"}:
                    continue
                if key.startswith("APP_"):
                    continue
                row[key] = fmt_value(metrics.get(key))
            app_row = row_map.get(idx)
            if app_row:
                for key, value in app_row.items():
                    row["APP_{}".format(key)] = value
            rows.append(row)
        emit_csv(columns, rows, SIM_COLUMN_TYPES)
        return
    values = parse_sim(stats)
    columns = ["record_type"] + SIM_KIND_COLUMNS.get(kind, ["kernel_name"])
    row = {"record_type": "data", "kernel_name": "run_total"}
    for key in columns:
        if key in {"record_type", "kernel_name"}:
            continue
        row[key] = fmt_value(values.get(key))
    emit_csv(columns, [row], SIM_COLUMN_TYPES)


def parse_native_csv(path: pathlib.Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = None
        header_has_metric_rows = False
        for raw in reader:
            if not raw:
                continue
            row = [cell.strip() for cell in raw]
            if row[0].startswith("=="):
                continue
            if "Kernel Name" in row or "Kernel" in row or "Name" in row:
                header = {name: idx for idx, name in enumerate(row)}
                header_has_metric_rows = "Metric Name" in row
                continue
            if header is None:
                continue

            def get(name, default=""):
                idx = header.get(name)
                if idx is None or idx >= len(row):
                    return default
                return row[idx]

            kernel = get("Kernel Name") or get("Kernel") or get("Name")
            device = get("Device")
            invocations = get("Invocations") or get("ID")
            if not kernel:
                continue
            if header_has_metric_rows:
                metric = get("Metric Name")
                value = get("Metric Value") or get("Avg") or get("Value")
                if not metric or not value:
                    continue
                rows.append(
                    {
                        "device": device,
                        "kernel": kernel,
                        "metric": metric,
                        "value": value,
                        "invocations": invocations,
                    }
                )
                continue

            metrics = {}
            for name, idx in header.items():
                if idx >= len(row):
                    continue
                if name in {
                    "ID",
                    "Process ID",
                    "Process Name",
                    "Host Name",
                    "Kernel Name",
                    "Kernel",
                    "Name",
                    "Context",
                    "Stream",
                    "Block Size",
                    "Grid Size",
                    "Device",
                    "CC",
                }:
                    continue
                value = row[idx]
                if value == "":
                    continue
                metrics[name] = value
            rows.append(
                {
                    "device": device,
                    "kernel": kernel,
                    "metric": None,
                    "value": None,
                    "invocations": invocations,
                    "metrics": metrics,
                }
            )
    grouped = {}
    for row in rows:
        label_parts = []
        if row["invocations"]:
            label_parts.append(str(row["invocations"]))
        if row["device"]:
            label_parts.append(row["device"])
        label_parts.append(row["kernel"])
        label = " :: ".join(label_parts)
        entry = grouped.setdefault(
            label,
            {
                "kernel": row["kernel"],
                "device": row["device"],
                "invocations": row["invocations"],
                "metrics": {},
            },
        )
        if row["invocations"]:
            entry["invocations"] = row["invocations"]
        if row.get("metric") is None:
            entry["metrics"].update(row["metrics"])
        else:
            entry["metrics"][row["metric"]] = row["value"]
    return grouped


def wanted_metrics(kind):
    return {
        "dep_chain": [
            "sm__inst_executed.sum",
            "sm__cycles_elapsed.avg",
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
        ],
        "l1": [
            "l1tex__t_sector_hit_rate.pct",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "lts__t_sectors_op_read.sum",
        ],
        "l2": [
            "lts__t_sector_hit_rate.pct",
            "lts__t_sectors_op_read.sum",
            "lts__t_sectors_op_write.sum",
        ],
        "l2_parallel": [
            "lts__t_sector_hit_rate.pct",
            "lts__t_sectors_op_read.sum",
            "lts__t_sectors_op_write.sum",
            "gpu__time_duration.sum",
        ],
        "hbm": [
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "dram__sectors_read.sum",
            "dram__sectors_write.sum",
            "dram__bytes_read.sum.per_second",
            "dram__bytes_write.sum.per_second",
            "gpu__time_duration.sum",
            "lts__t_sector_hit_rate.pct",
        ],
        "smem": [
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        ],
        "validation_single": [
            "l1tex__t_sector_hit_rate.pct",
            "lts__t_sector_hit_rate.pct",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "sm__inst_executed.sum",
            "sm__cycles_elapsed.avg",
            "gpu__time_duration.sum",
        ],
        "nvlink": [
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "nvlrx__bytes.sum",
            "nvltx__bytes.sum",
            "lts__t_sector_hit_rate.pct",
            "gpu__time_duration.sum",
        ],
    }.get(kind)


def filter_kernel_entries(entries, patterns):
    if not patterns:
        return entries
    filtered = []
    for label, entry in entries:
        kernel = entry["kernel"]
        if any(pat in kernel for pat in patterns):
            filtered.append((label, entry))
    return filtered


def map_rows_to_entries(entries, rows):
    if not entries or not rows:
        return {}
    if len(rows) == len(entries):
        return {idx: row for idx, row in enumerate(rows)}
    return {}


def csv_stat_type(column, sim_types=None):
    if column == "record_type":
        return "meta"
    if column.startswith("APP_"):
        return "app_per_kernel"
    if sim_types and column in sim_types:
        return sim_types[column]
    if column in {"kernel", "kernel_name", "device"}:
        return "text"
    if column in {"launch_id", "kernel_launch_uid", "invocations"}:
        return "per_kernel"
    return "per_kernel"


def ordered_app_columns(rows):
    if not rows:
        return []
    return ["APP_{}".format(key) for key in rows[0].keys()]


def emit_csv(columns, rows, sim_types=None):
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(columns)
    writer.writerow([csv_stat_type(col, sim_types) for col in columns])
    for row in rows:
        writer.writerow([row.get(col, "") for col in columns])


def native_metric(metrics, *names):
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def report_native(kind: str, native_csv: pathlib.Path, pattern_arg: str):
    entries = parse_native_csv(native_csv)
    patterns = [p for p in pattern_arg.split(",") if p]
    selected = filter_kernel_entries(list(entries.items()), patterns)
    if not selected:
        selected = list(entries.items())
    app_rows = parse_csv_table_rows(native_csv.parent / "stats.txt")
    app_latencies = parse_app_latencies(native_csv.parent / "stats.txt")
    row_map = map_rows_to_entries(selected, app_rows)
    wanted = wanted_metrics(kind)
    app_columns = ordered_app_columns(app_rows)
    columns = ["record_type", "kernel", "device", "launch_id"]
    columns.extend([col for col in app_columns if col not in columns])
    if kind == "dep_chain":
        columns.extend(
            [
                "APP_fadd_raw_cycles_per_op",
                "APP_fadd_adjusted_cycles_per_op",
                "APP_iadd_raw_cycles_per_op",
                "sm__inst_executed.sum",
                "sm__cycles_elapsed.avg",
                "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
            ]
        )
    else:
        metric_columns = wanted if wanted is not None else []
        columns.extend([col for col in metric_columns if col not in columns])

    rows = []
    for idx, (_, entry) in enumerate(selected):
        row = {
            "record_type": "data",
            "kernel": entry.get("kernel", ""),
            "device": entry.get("device", ""),
            "launch_id": entry.get("invocations", ""),
        }
        app_row = row_map.get(idx)
        if app_row:
            for key, value in app_row.items():
                row["APP_{}".format(key)] = value
        if kind == "dep_chain":
            kernel_name = entry.get("kernel", "")
            if "dep_chain_fp" in kernel_name:
                row["APP_fadd_raw_cycles_per_op"] = fmt_value(
                    app_latencies.get("fadd_raw_cycles_per_op")
                )
                row["APP_fadd_adjusted_cycles_per_op"] = fmt_value(
                    app_latencies.get("fadd_adjusted_cycles_per_op")
                )
            if "dep_chain_int" in kernel_name:
                row["APP_iadd_raw_cycles_per_op"] = fmt_value(
                    app_latencies.get("iadd_raw_cycles_per_op")
                )
            for key in columns:
                if key in row or key.startswith("APP_"):
                    continue
                row[key] = fmt_value(entry["metrics"].get(key))
        else:
            for key in columns:
                if key in row or key.startswith("APP_"):
                    continue
                row[key] = fmt_value(entry["metrics"].get(key))
        rows.append(row)
    emit_csv(columns, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["gpgpusim", "native"])
    parser.add_argument("kind")
    parser.add_argument("path")
    parser.add_argument("patterns", nargs="?", default="")
    args = parser.parse_args()
    path = pathlib.Path(args.path)
    if not path.exists():
        sys.exit("missing report input: {}".format(path))
    if args.mode == "gpgpusim":
        report_gpgpusim(args.kind, path)
    else:
        report_native(args.kind, path, args.patterns)


if __name__ == "__main__":
    main()
