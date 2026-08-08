#!/usr/bin/env python3
"""Serial DCGM/NSYS/NCU campaign for the collected multi-GPU cases.

The script deliberately keeps the hardware-facing policy in one place:
physical GPU IDs are never inferred from CUDA ordinals, all case executions
are serial, and the cleanup path restores every setting changed by the
campaign.  Nsight Systems is invoked directly with the repository's
``cuda_iface_full`` metric set because the local three-stage nsys-tool launch
path does not preserve CUDA activity on this host; ``check_report.py`` is run
after every report.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
KERNELS_ROOT = WORKSPACE_ROOT / "kernels"
ACCELSIM_ROOT = WORKSPACE_ROOT / "htm-accel-sim"
METRIC_ROOT = Path("/home/yjkong/Repos/metric_collection")
PROFILE_SH = METRIC_ROOT / "tool/profile.sh"
METRICS_PY = METRIC_ROOT / "tool/metrics.py"
NSYS = Path("/usr/local/cuda/bin/nsys")
NSYS_CHECK = METRIC_ROOT / "tool/nsys/check_report.py"
IFACE_SET = METRIC_ROOT / "tool/nsys/sets/iface_full.config"
NCU = Path("/usr/local/cuda-12.8/bin/ncu")
NCU_RUNNER = KERNELS_ROOT / "cal_kernels/common/ncu_group_runner.py"

TARGET_GPUS = [0, 1, 4, 5, 6, 7]
GPC_MHZ = 1590
MEMORY_MHZ_PREFERRED = 2619
GROUPS = "timing_clock,l1_l2,dram,execution,scheduler_stall"

CASES = [
    ("peer_copy", "test", "2gpu", [0, 1]),
    ("peer_copy", "small", "2gpu", [0, 1]),
    ("peer_copy", "medium", "2gpu", [0, 1]),
    ("peer_copy", "large", "2gpu", [0, 1]),
    ("peer_copy", "test", "4gpu", [4, 5, 6, 7]),
    ("peer_copy", "small", "4gpu", [4, 5, 6, 7]),
    ("peer_copy", "medium", "4gpu", [4, 5, 6, 7]),
    ("peer_copy", "large", "4gpu", [4, 5, 6, 7]),
    ("all_gather", "test", "2gpu", [0, 1]),
    ("all_gather", "small", "2gpu", [0, 1]),
    ("all_gather", "medium", "2gpu", [0, 1]),
    ("all_gather", "large", "2gpu", [0, 1]),
    ("all_gather", "test", "4gpu", [4, 5, 6, 7]),
    ("all_gather", "small", "4gpu", [4, 5, 6, 7]),
    ("all_gather", "medium", "4gpu", [4, 5, 6, 7]),
    ("all_gather", "large", "4gpu", [4, 5, 6, 7]),
    ("all_reduce", "test", "2gpu", [0, 1]),
    ("all_reduce", "small", "2gpu", [0, 1]),
    ("all_reduce", "medium", "2gpu", [0, 1]),
    ("all_reduce", "large", "2gpu", [0, 1]),
    ("all_reduce", "test", "4gpu", [4, 5, 6, 7]),
    ("all_reduce", "small", "4gpu", [4, 5, 6, 7]),
    ("all_reduce", "medium", "4gpu", [4, 5, 6, 7]),
    ("all_reduce", "large", "4gpu", [4, 5, 6, 7]),
]

# These are the measured values selected from the repeat calibration.  The
# repeated region, rather than setup/correctness time, is what is targeted.
GATHER_REPEATS = {
    "A": {
        "2gpu": {"test": 3_000_000, "small": 2_000_000, "medium": 800_000, "large": 230_000},
        "4gpu": {"test": 3_000_000, "small": 800_000, "medium": 500_000, "large": 150_000},
    },
    "B": {
        "2gpu": {"test": 1_200_000, "small": 800_000, "medium": 400_000, "large": 100_000},
        "4gpu": {"test": 1_500_000, "small": 400_000, "medium": 250_000, "large": 50_000},
    },
    "C": {
        "2gpu": {"test": 600_000, "small": 400_000, "medium": 200_000, "large": 50_000},
        "4gpu": {"test": 750_000, "small": 200_000, "medium": 125_000, "large": 25_000},
    },
}

BINARY_NAMES = {
    "peer_copy": "peer_copy",
    "all_gather": "mgather_naive",
    "all_reduce": "mreduce",
}
KERNEL_NAMES = {
    "peer_copy": "peer_copy_kernel",
    "all_gather": "allgather_peer_kernel",
    "all_reduce": "allreduce_peer_kernel",
}

MANIFEST_COLUMNS = [
    "application", "size", "gpu_count", "physical_gpu_ids", "logical_gpu_mapping",
    "pass", "profiler", "target_metric_gpus", "metric_frequency", "ncu_metric_group",
    "repeat", "cache_control", "command", "cuda_visible_devices", "git_commits",
    "profiler_versions", "gpu_model_uuid", "requested_gpc_mhz", "requested_memory_mhz",
    "observed_clock_before", "observed_clock_after", "temperature_pstate_throttle",
    "start_time", "end_time", "exit_code", "nsys_diagnose", "fallback_status",
    "output_path", "validity_status", "failure_reason",
]


class CampaignError(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def command_text(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def run_command(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    started = now()
    out = ""
    try:
        result = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        rc = result.returncode
        out = result.stdout or ""
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        out += "\n[TIMEOUT]\n"
    except OSError as exc:
        rc = 127
        out = "[OSERROR] {}\n".format(exc)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(out, encoding="utf-8")
    return {
        "command": command_text(command),
        "start_time": started,
        "end_time": now(),
        "returncode": rc,
        "output": out,
        "log_path": str(log_path) if log_path else None,
    }


def sudo_command(*args: str) -> List[str]:
    return ["sudo", "-n", *args]


def query_csv(command: Sequence[str]) -> List[List[str]]:
    result = run_command(command)
    if result["returncode"] != 0:
        raise CampaignError("command failed: {}\n{}".format(result["command"], result["output"].strip()))
    return [row for row in csv.reader(result["output"].splitlines()) if row]


def gpu_snapshot(gpus: Iterable[int]) -> Dict[str, Any]:
    ids = [str(g) for g in gpus]
    fields = [
        "index", "name", "uuid", "pci.bus_id", "compute_mode", "temperature.gpu",
        "pstate", "clocks.current.graphics", "clocks.current.sm", "clocks.current.memory",
        "clocks_throttle_reasons.active", "power.draw",
    ]
    rows = query_csv([
        "nvidia-smi", "-i", ",".join(ids), "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ])
    out: Dict[str, Any] = {}
    for row in rows:
        padded = row + [""] * max(0, len(fields) - len(row))
        out[padded[0].strip()] = {fields[i]: padded[i].strip() for i in range(len(fields))}
    return out


def supported_clock_values(gpu: int, kind: str) -> List[int]:
    text = run_command(["nvidia-smi", "-i", str(gpu), "-q", "-d", "SUPPORTED_CLOCKS"])["output"]
    values: List[int] = []
    if kind == "gpc":
        patterns = (r"(?i)\bgraphics\b\s*:\s*([0-9]+)\s*MHz", r"(?i)\bgpc\b\s*:\s*([0-9]+)\s*MHz")
    else:
        patterns = (r"(?i)\bmemory\b\s*:\s*([0-9]+)\s*MHz",)
    for line in text.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                values.append(int(match.group(1)))
                break
    return sorted(set(values))


def query_compute_apps(gpus: Iterable[int]) -> List[Dict[str, str]]:
    rows = query_csv([
        "nvidia-smi", "-i", ",".join(str(g) for g in gpus),
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    out = []
    for row in rows:
        if len(row) >= 4 and row[1].strip() not in {"", "No running processes found"}:
            out.append({"gpu_uuid": row[0].strip(), "pid": row[1].strip(), "process_name": row[2].strip(), "used_gpu_memory": row[3].strip()})
    return out


def git_commit(path: Path) -> str:
    result = run_command(["git", "-C", str(path), "rev-parse", "HEAD"])
    return result["output"].strip() if result["returncode"] == 0 else "unavailable"


def tool_version(command: Sequence[str]) -> str:
    result = run_command(command)
    return result["output"].strip()


def matrix_rows() -> List[Dict[str, Any]]:
    rows = []
    for app, size, gpu_class, physical in CASES:
        rows.append({
            "application": app,
            "size": size,
            "gpu_count": len(physical),
            "gpu_class": gpu_class,
            "physical_gpus": physical,
            "logical_mapping": {str(i): physical[i] for i in range(len(physical))},
            "case_name": "{}-{}-{}".format(app, gpu_class[0], size),
        })
    return rows


def print_matrix(rows: Sequence[Dict[str, Any]]) -> None:
    print("Complete profiling matrix: {} cases".format(len(rows)), flush=True)
    print("| # | application | size | GPUs | physical -> logical |", flush=True)
    print("|---:|---|---|---|---|", flush=True)
    for index, row in enumerate(rows, 1):
        physical = ",".join(str(g) for g in row["physical_gpus"])
        mapping = ", ".join("P{}→L{}".format(p, l) for l, p in row["logical_mapping"].items())
        print("| {} | {} | {} | {} | {} -> {} |".format(index, row["application"], row["size"], row["gpu_count"], physical, mapping), flush=True)


def app_args(app: str, gpu_count: int, size: str, pass_name: str) -> List[str]:
    args = [str(gpu_count), size]
    if app == "all_gather":
        if pass_name == "D":
            args.append("1")
        else:
            args.append(str(GATHER_REPEATS[pass_name]["{}gpu".format(gpu_count)][size]))
    return args


def env_for(physical: Sequence[int]) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in physical)
    return env


def app_binary(app: str) -> Path:
    return KERNELS_ROOT / "mgpu_kernels" / app / "build/GH100/native" / BINARY_NAMES[app]


def parse_nsys_diagnosis(path: Path) -> Dict[str, Any]:
    diagnostics = path / "diagnostics.txt"
    text = diagnostics.read_text(encoding="utf-8", errors="replace") if diagnostics.exists() else ""
    verdict = "ERROR"
    for candidate in ("PASS", "PARTIAL", "FAIL", "ERROR"):
        if "判定: {}".format(candidate) in text:
            verdict = candidate
            break
    kernel_match = re.search(r"kernel 轨:\s*([0-9]+)\s*条", text)
    return {
        "verdict": verdict,
        "kernel_count": int(kernel_match.group(1)) if kernel_match else 0,
        "diagnostics": str(diagnostics),
        "text": text,
    }


def chown_tree(path: Path) -> None:
    if not path.exists():
        return
    run_command(sudo_command("chown", "-R", "{}:{}".format(os.getuid(), os.getgid()), str(path)))


class Campaign:
    def __init__(self, output_root: Path, resume: bool) -> None:
        self.output_root = output_root.resolve()
        self.resume = resume
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_root / "campaign_manifest.json"
        self.csv_path = self.output_root / "campaign_manifest.csv"
        self.rows = matrix_rows()
        self.records: List[Dict[str, Any]] = []
        self.changed_compute_mode = False
        self.compute_mode_before: Dict[str, str] = {}
        self.gpc_lock_attempted = False
        self.memory_lock_attempted = False
        self.memory_lock_applied = False
        self.campaign: Dict[str, Any] = {
            "schema_version": 1,
            "workflow": "mgpu-hardware-calibration",
            "created_at": now(),
            "output_root": str(self.output_root),
            "matrix_count": len(self.rows),
            "matrix": self.rows,
            "physical_gpu_policy": {
                "two_gpu": [0, 1],
                "four_gpu": [4, 5, 6, 7],
                "forbidden_workload_and_counter_gpus": [2, 3],
                "logical_mapping_note": "CUDA_VISIBLE_DEVICES remaps each physical list to logical 0..N-1",
            },
            "requested_clocks": {"gpc_mhz": GPC_MHZ, "memory_mhz_preferred": MEMORY_MHZ_PREFERRED},
            "passes": {
                "A": {"profiler": "dcgm", "metric_frequency": "100ms", "fields": "pass1_core"},
                "B": {"profiler": "nsys", "metric_frequency": "10kHz", "config": "cuda_iface_full"},
                "C": {"profiler": "nsys", "metric_frequency_initial": "100kHz", "fallback": "50kHz", "config": "cuda_iface_full"},
                "D": {"profiler": "ncu", "groups": GROUPS, "clock_control": "none"},
            },
            "gather_repeat_values": GATHER_REPEATS,
            "gather_repeat_validation": {
                "status": "passed",
                "two_gpu_preflight": "/home/yjkong/htm-workspace/profiling-results/preflight/2gpu-gather/20260808-234023",
                "four_gpu_preflight": "/home/yjkong/htm-workspace/profiling-results/preflight/4gpu-gather/20260808-234119",
                "idle_baselines": [
                    "/home/yjkong/htm-workspace/profiling-results/preflight/2gpu-idle/20260808-234001",
                    "/home/yjkong/htm-workspace/profiling-results/preflight/4gpu-idle/20260808-234059",
                ],
                "path_evidence": "DCGM preflight measured approximately 375 GB/s NVLink TX/RX for 2-GPU and 250-372 GB/s per-direction NVLink traffic for 4-GPU; PCIe returned to a few MB/s after setup/control.",
            },
            "tool_versions": {},
            "gpu_inventory": {},
            "clock_lock": {},
            "cases": [],
            "records": self.records,
            "status": "initialized",
        }

    def save(self) -> None:
        self.campaign["records"] = self.records
        self.manifest_path.write_text(json.dumps(self.campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in self.records:
                writer.writerow({key: row.get(key, "") for key in MANIFEST_COLUMNS})

    def preflight(self) -> None:
        self.campaign["tool_versions"] = {
            "nvidia_smi": tool_version(["nvidia-smi", "--version"]),
            "nvcc": tool_version(["/usr/local/cuda-12.8/bin/nvcc", "--version"]),
            "nsys": tool_version([str(NSYS), "--version"]),
            "ncu": tool_version([str(NCU), "--version"]),
            "dcgmi": tool_version(["dcgmi", "--version"]),
        }
        self.campaign["git_commits"] = {
            "kernels": git_commit(KERNELS_ROOT),
            "htm_accel_sim": git_commit(ACCELSIM_ROOT),
            "metric_collection": git_commit(METRIC_ROOT),
        }
        self.campaign["gpu_inventory"] = gpu_snapshot(TARGET_GPUS)
        topology_dir = self.output_root / "preflight"
        topology_dir.mkdir(exist_ok=True)
        for name, command in {
            "topo-m.txt": ["nvidia-smi", "topo", "-m"],
            "topo-p2p-r.txt": ["nvidia-smi", "topo", "-p2p", "r"],
            "topo-p2p-w.txt": ["nvidia-smi", "topo", "-p2p", "w"],
            "topo-p2p-n.txt": ["nvidia-smi", "topo", "-p2p", "n"],
        }.items():
            result = run_command(command, log_path=topology_dir / name)
            if result["returncode"] != 0:
                raise CampaignError("topology command failed: {}".format(result["command"]))
        apps = query_compute_apps(TARGET_GPUS)
        if apps:
            raise CampaignError("target GPUs have existing compute processes; no process was killed: {}".format(apps))
        gpc_sets = [set(supported_clock_values(gpu, "gpc")) for gpu in TARGET_GPUS]
        common_gpc = set.intersection(*gpc_sets)
        self.campaign["supported_gpc_clocks_mhz"] = sorted(common_gpc)
        if GPC_MHZ not in common_gpc:
            raise CampaignError("1590 MHz GPC clock is not common across target GPUs")
        memory_sets = [set(supported_clock_values(gpu, "memory")) for gpu in TARGET_GPUS]
        common_memory = set.intersection(*memory_sets)
        self.campaign["supported_memory_clocks_mhz"] = sorted(common_memory)
        self.campaign["memory_lock_candidate_mhz"] = MEMORY_MHZ_PREFERRED if MEMORY_MHZ_PREFERRED in common_memory else (min(common_memory) if common_memory else None)
        self.campaign["clock_state_before"] = gpu_snapshot(TARGET_GPUS)
        self.compute_mode_before = {
            gpu: self.campaign["clock_state_before"].get(str(gpu), {}).get("compute_mode", "")
            for gpu in TARGET_GPUS
        }
        self.campaign["status"] = "preflight_passed"
        self.save()

    def acquire(self) -> None:
        if any(mode not in {"Exclusive_Process", "EXCLUSIVE_PROCESS"} for mode in self.compute_mode_before.values()):
            result = run_command(sudo_command("nvidia-smi", "-i", ",".join(str(g) for g in TARGET_GPUS), "-c", "EXCLUSIVE_PROCESS"))
            self.changed_compute_mode = result["returncode"] == 0
            if result["returncode"] != 0:
                raise CampaignError("could not enable EXCLUSIVE_PROCESS: {}".format(result["output"].strip()))
        self.gpc_lock_attempted = True
        gpc_result = run_command(sudo_command("nvidia-smi", "-i", ",".join(str(g) for g in TARGET_GPUS), "-lgc", "{},{}".format(GPC_MHZ, GPC_MHZ)))
        self.campaign["clock_lock"]["gpc_command"] = gpc_result["command"]
        self.campaign["clock_lock"]["gpc_output"] = gpc_result["output"]
        self.campaign["clock_lock"]["gpc_status"] = "applied" if gpc_result["returncode"] == 0 else "failed"
        if gpc_result["returncode"] != 0:
            raise CampaignError("GPC lock failed: {}".format(gpc_result["output"].strip()))
        candidate = self.campaign.get("memory_lock_candidate_mhz")
        if candidate is not None:
            self.memory_lock_attempted = True
            memory_result = run_command(sudo_command("nvidia-smi", "-i", ",".join(str(g) for g in TARGET_GPUS), "-lmc", "{},{}".format(candidate, candidate)))
            self.memory_lock_applied = memory_result["returncode"] == 0
            self.campaign["clock_lock"]["memory_command"] = memory_result["command"]
            self.campaign["clock_lock"]["memory_output"] = memory_result["output"]
            self.campaign["clock_lock"]["memory_status"] = "applied" if self.memory_lock_applied else "failed_non_deferred_lock"
        self.campaign["clock_lock"]["acquired_at"] = now()
        self.campaign["status"] = "locks_acquired"
        self.save()

    def restore(self) -> None:
        restore: Dict[str, Any] = {"start_time": now(), "commands": []}
        if self.gpc_lock_attempted:
            result = run_command(sudo_command("nvidia-smi", "-i", ",".join(str(g) for g in TARGET_GPUS), "-rgc"))
            restore["commands"].append(result)
        if self.memory_lock_attempted:
            result = run_command(sudo_command("nvidia-smi", "-i", ",".join(str(g) for g in TARGET_GPUS), "-rmc"))
            restore["commands"].append(result)
        if self.changed_compute_mode:
            mode_map = {"Default": "DEFAULT", "Exclusive_Process": "EXCLUSIVE_PROCESS", "Prohibited": "PROHIBITED", "EXCLUSIVE_PROCESS": "EXCLUSIVE_PROCESS"}
            for gpu, old_mode in self.compute_mode_before.items():
                token = mode_map.get(old_mode)
                if token:
                    restore["commands"].append(run_command(sudo_command("nvidia-smi", "-i", str(gpu), "-c", token)))
        restore["end_time"] = now()
        restore["final_target_snapshot"] = gpu_snapshot(TARGET_GPUS)
        restore["restored"] = all(item["returncode"] == 0 for item in restore["commands"])
        self.campaign["cleanup"] = restore
        self.campaign["clock_state_after_cleanup"] = restore["final_target_snapshot"]
        self.campaign["status"] = "restored" if restore["restored"] else "restore_failed"
        self.save()

    def check_idle(self, physical: Sequence[int]) -> None:
        apps = query_compute_apps(physical)
        if apps:
            raise CampaignError("unrelated compute process appeared on {}: {}".format(physical, apps))

    def base_record(self, row: Dict[str, Any], pass_name: str, output_path: Path, repeat: Any) -> Dict[str, Any]:
        visible = ",".join(str(g) for g in row["physical_gpus"])
        logical = json.dumps(row["logical_mapping"], sort_keys=True)
        return {
            "application": row["application"],
            "size": row["size"],
            "gpu_count": row["gpu_count"],
            "physical_gpu_ids": ",".join(str(g) for g in row["physical_gpus"]),
            "logical_gpu_mapping": logical,
            "pass": pass_name,
            "profiler": {"A": "dcgm", "B": "nsys", "C": "nsys", "D": "ncu"}[pass_name[0]],
            "target_metric_gpus": "",
            "metric_frequency": {"A": "100ms", "B": "10kHz", "C": "100kHz/50kHz", "D": "counter"}[pass_name[0]],
            "ncu_metric_group": "",
            "repeat": repeat,
            "cache_control": "all",
            "command": "",
            "cuda_visible_devices": visible,
            "git_commits": json.dumps(self.campaign.get("git_commits", {}), sort_keys=True),
            "profiler_versions": json.dumps(self.campaign.get("tool_versions", {}), sort_keys=True),
            "gpu_model_uuid": json.dumps({str(g): self.campaign.get("gpu_inventory", {}).get(str(g), {}) for g in row["physical_gpus"]}, sort_keys=True),
            "requested_gpc_mhz": GPC_MHZ,
            "requested_memory_mhz": self.campaign.get("memory_lock_candidate_mhz", "N/A"),
            "observed_clock_before": "",
            "observed_clock_after": "",
            "temperature_pstate_throttle": "",
            "start_time": now(),
            "end_time": "",
            "exit_code": "",
            "nsys_diagnose": "N/A",
            "fallback_status": "none",
            "output_path": str(output_path),
            "validity_status": "running",
            "failure_reason": "",
        }

    def finish_record(self, record: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any], result: Dict[str, Any], status: str, reason: str = "") -> None:
        record["observed_clock_before"] = json.dumps(before, sort_keys=True)
        record["observed_clock_after"] = json.dumps(after, sort_keys=True)
        record["temperature_pstate_throttle"] = json.dumps({"before": before, "after": after}, sort_keys=True)
        record["end_time"] = now()
        record["exit_code"] = result.get("returncode", "")
        record["command"] = result.get("command", record.get("command", ""))
        record["validity_status"] = status
        record["failure_reason"] = reason
        self.records.append(record)
        self.save()

    def run_dcgm(self, row: Dict[str, Any], case_dir: Path) -> None:
        output = case_dir / "dcgm"
        output.mkdir(parents=True, exist_ok=True)
        repeat = app_args(row["application"], row["gpu_count"], row["size"], "A")[-1] if row["application"] == "all_gather" else 1
        args = app_args(row["application"], row["gpu_count"], row["size"], "A")
        binary = app_binary(row["application"])
        record = self.base_record(row, "A", output, repeat)
        record["target_metric_gpus"] = record["physical_gpu_ids"]
        command = ["bash", str(PROFILE_SH), "--gpus", record["physical_gpu_ids"], "--fields", "pass1_core", "--interval-ms", "100", "--lead", "2", "--lag", "3", "--out", str(output), "--note", "Pass A physical/logical mapping {} / {}".format(record["physical_gpu_ids"], record["logical_gpu_mapping"]), "--", str(binary), *args]
        before = gpu_snapshot(row["physical_gpus"])
        result = run_command(command, cwd=WORKSPACE_ROOT, env=env_for(row["physical_gpus"]), log_path=output / "campaign-command.log", timeout=1800)
        run_dirs = sorted([path for path in output.iterdir() if path.is_dir() and re.fullmatch(r"[0-9]{8}-[0-9]{6}", path.name)]) if output.exists() else []
        parse_result: Dict[str, Any] = {"returncode": 0, "output": "", "command": ""}
        if run_dirs:
            raw_run = run_dirs[-1]
            parse_result = run_command([sys.executable, str(METRICS_PY), "parse", str(raw_run)], log_path=raw_run / "parse.log")
            record["output_path"] = str(raw_run)
        after = gpu_snapshot(row["physical_gpus"])
        valid = result["returncode"] == 0 and parse_result["returncode"] == 0 and bool(run_dirs)
        reason = "" if valid else "profile/parse failed or no DCGM run directory"
        self.finish_record(record, before, after, result, "PASS" if valid else "FAIL", reason)

    def nsys_command(self, row: Dict[str, Any], output: Path, frequency: int, metric_gpus: Sequence[int], pass_name: str) -> List[str]:
        binary = app_binary(row["application"])
        args = app_args(row["application"], row["gpu_count"], row["size"], pass_name)
        report_base = output / "report"
        return sudo_command("-E", "env", "CUDA_VISIBLE_DEVICES=" + ",".join(str(g) for g in row["physical_gpus"]), str(NSYS), "profile", "--force-overwrite=true", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none", "--gpu-metrics-devices=" + ",".join(str(g) for g in metric_gpus), "--gpu-metrics-set=file:" + str(IFACE_SET), "--gpu-metrics-frequency=" + str(frequency), "-o", str(report_base), "--", str(binary), *args)

    def diagnose_nsys(self, output: Path, report: Path) -> Dict[str, Any]:
        result = run_command([sys.executable, str(NSYS_CHECK), "--json", str(report)], cwd=output, log_path=output / "diagnose.json", timeout=1800)
        chown_tree(output)
        diagnosis = parse_nsys_diagnosis(output)
        diagnosis["command_result"] = {key: value for key, value in result.items() if key != "output"}
        return diagnosis

    def run_nsys(self, row: Dict[str, Any], case_dir: Path, pass_name: str, frequency: int, metric_gpus: Sequence[int], repeat_pass: str) -> Dict[str, Any]:
        output = case_dir / ("nsys-10khz" if pass_name == "B" else "gpu{}-{}khz".format(metric_gpus[0], frequency // 1000))
        output.mkdir(parents=True, exist_ok=True)
        repeat = app_args(row["application"], row["gpu_count"], row["size"], repeat_pass)[-1] if row["application"] == "all_gather" else 1
        record = self.base_record(row, pass_name, output, repeat)
        record["target_metric_gpus"] = ",".join(str(g) for g in metric_gpus)
        record["metric_frequency"] = "{}kHz".format(frequency // 1000)
        command = self.nsys_command(row, output, frequency, metric_gpus, repeat_pass)
        record["execution_backend"] = "direct_nsys_profile"
        before = gpu_snapshot(row["physical_gpus"])
        result = run_command(command, cwd=WORKSPACE_ROOT, env=env_for(row["physical_gpus"]), log_path=output / "command.log", timeout=3600)
        chown_tree(output)
        report = output / "report.nsys-rep"
        diagnosis = self.diagnose_nsys(output, report) if report.exists() else {"verdict": "ERROR", "kernel_count": 0, "text": "report missing"}
        after = gpu_snapshot(row["physical_gpus"])
        valid = result["returncode"] == 0 and diagnosis["verdict"] == "PASS" and diagnosis["kernel_count"] > 0
        record["nsys_diagnose"] = diagnosis["verdict"]
        reason = "" if valid else "nsys rc={} diagnose={} kernels={}".format(result["returncode"], diagnosis["verdict"], diagnosis["kernel_count"])
        self.finish_record(record, before, after, result, "PASS" if valid else "FAIL", reason)
        return {"valid": valid, "diagnosis": diagnosis, "record": record, "result": result}

    def run_nsys_c(self, row: Dict[str, Any], case_dir: Path, physical_gpu: int) -> None:
        first = self.run_nsys(row, case_dir, "C", 100000, [physical_gpu], "C")
        if first["valid"]:
            return
        # Keep the failed 100 kHz directory.  A 50 kHz rerun is allowed only
        # for an unusable report, and is recorded as an explicit fallback.
        if not first["diagnosis"].get("text") and first["result"].get("returncode") != 0:
            return
        fallback = self.run_nsys(row, case_dir, "C", 50000, [physical_gpu], "C")
        fallback["record"]["fallback_status"] = "accepted_50khz" if fallback["valid"] else "50khz_failed"
        self.records[-1]["fallback_status"] = fallback["record"]["fallback_status"]
        self.records[-1]["failure_reason"] = "fallback for 100 kHz unusable report"
        self.save()

    def run_ncu(self, row: Dict[str, Any], case_dir: Path, physical_gpu: int) -> None:
        output = case_dir / "ncu-gpu{}".format(physical_gpu)
        output.mkdir(parents=True, exist_ok=True)
        binary = app_binary(row["application"])
        link = output / binary.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(binary)
        args = app_args(row["application"], row["gpu_count"], row["size"], "D")
        command = [sys.executable, str(NCU_RUNNER), "--ncu", str(NCU), "--run-dir", str(output), "--merged-csv", str(output / "ncu.csv"), "--stats-file", str(output / "stats.txt"), "--runner-log", str(output / "ncu_runner.log"), "--run-args", " ".join(shlex.quote(arg) for arg in args), "--kernel-name", KERNEL_NAMES[row["application"]], "--groups", GROUPS, "--cache-control", "all", "--gpu-index", str(physical_gpu), "--ncu-devices", "0", "--visible-devices", ",".join(str(g) for g in row["physical_gpus"]), "--target-gpc-mhz", str(GPC_MHZ), "--clock-lock-mode", "external", "--sudo-mode", "always", "--timeout", "900", "--binary-name", binary.name]
        record = self.base_record(row, "D", output, 1)
        record["target_metric_gpus"] = str(physical_gpu)
        record["metric_frequency"] = "counter"
        record["ncu_metric_group"] = GROUPS
        record["command"] = command_text(command)
        before = gpu_snapshot(row["physical_gpus"])
        result = run_command(command, cwd=WORKSPACE_ROOT, env=env_for(row["physical_gpus"]), log_path=output / "campaign-command.log", timeout=3600)
        chown_tree(output)
        group_statuses: Dict[str, str] = {}
        group_manifest = output / "ncu_groups/manifest.json"
        if group_manifest.exists():
            try:
                data = json.loads(group_manifest.read_text(encoding="utf-8"))
                group_statuses = {name: value.get("status", "unknown") for name, value in data.get("groups", {}).items()}
            except (OSError, ValueError):
                pass
        after = gpu_snapshot(row["physical_gpus"])
        valid = result["returncode"] == 0 and bool(group_statuses) and all(status in {"passed", "skipped_unavailable"} for status in group_statuses.values())
        reason = "" if valid else "NCU group statuses: {}".format(group_statuses or "manifest missing")
        self.finish_record(record, before, after, result, "PASS" if valid else "FAIL", reason)

    def run_case(self, row: Dict[str, Any]) -> None:
        case_dir = self.output_root / row["application"] / row["size"] / row["gpu_class"]
        case_dir.mkdir(parents=True, exist_ok=True)
        case_key = "{}/{}/{}".format(row["application"], row["size"], row["gpu_class"])
        self.check_idle(row["physical_gpus"])
        print("[campaign] START {} physical={} logical={}".format(case_key, row["physical_gpus"], row["logical_mapping"]), flush=True)
        self.run_dcgm(row, case_dir)
        self.check_idle(row["physical_gpus"])
        self.run_nsys(row, case_dir, "B", 10000, row["physical_gpus"], "B")
        self.check_idle(row["physical_gpus"])
        c_targets = [row["physical_gpus"][0], row["physical_gpus"][1]]
        for target in c_targets:
            self.check_idle(row["physical_gpus"])
            self.run_nsys_c(row, case_dir, target)
        self.check_idle(row["physical_gpus"])
        self.run_ncu(row, case_dir, row["physical_gpus"][0])
        self.campaign["cases"].append({"key": case_key, "path": str(case_dir), "physical_gpus": row["physical_gpus"], "logical_mapping": row["logical_mapping"], "completed_at": now()})
        self.save()
        print("[campaign] END {}".format(case_key), flush=True)

    def run(self) -> None:
        self.preflight()
        self.acquire()
        try:
            self.campaign["status"] = "running"
            self.save()
            for row in self.rows:
                case_key = "{}/{}/{}".format(row["application"], row["size"], row["gpu_class"])
                if self.resume and any(record.get("application") == row["application"] and record.get("size") == row["size"] and record.get("gpu_count") == row["gpu_count"] and record.get("pass") == "D" and record.get("validity_status") == "PASS" for record in self.records):
                    print("[campaign] RESUME skip {}".format(case_key), flush=True)
                    continue
                self.run_case(row)
            self.campaign["status"] = "completed"
        except BaseException as exc:
            self.campaign["status"] = "aborted"
            self.campaign["abort_reason"] = repr(exc)
            self.save()
            raise
        finally:
            self.restore()


def install_signal_handlers() -> None:
    def handler(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("received signal {}".format(signum))
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def print_dry_run_commands(output_root: Path) -> None:
    """Print the future campaign plan without touching GPUs or starting tools."""
    output_root = output_root.resolve()
    print("DRY RUN ONLY: no GPU/tool command is being executed.", flush=True)
    print("planned output root: {}".format(output_root), flush=True)
    print("\n# Read-only preflight commands", flush=True)
    for command in (
        ["nvidia-smi", "topo", "-m"],
        ["nvidia-smi", "topo", "-p2p", "r"],
        ["nvidia-smi", "topo", "-p2p", "w"],
        ["nvidia-smi", "topo", "-p2p", "n"],
        ["nvidia-smi", "-i", "0,1,4,5,6,7", "--query-gpu=index,name,uuid,pci.bus_id,compute_mode,temperature.gpu,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory", "--format=csv,noheader,nounits"],
    ):
        print(command_text(command), flush=True)
    print("\n# Reservation, clock setup, and guaranteed cleanup", flush=True)
    print(command_text(sudo_command("nvidia-smi", "-i", "0,1,4,5,6,7", "-c", "EXCLUSIVE_PROCESS")), flush=True)
    print(command_text(sudo_command("nvidia-smi", "-i", "0,1,4,5,6,7", "-lgc", "1590,1590")), flush=True)
    print(command_text(sudo_command("nvidia-smi", "-i", "0,1,4,5,6,7", "-lmc", "2619,2619")), flush=True)
    print("# The driver restores the initial compute mode and runs:", flush=True)
    print(command_text(sudo_command("nvidia-smi", "-i", "0,1,4,5,6,7", "-rgc")), flush=True)
    print(command_text(sudo_command("nvidia-smi", "-i", "0,1,4,5,6,7", "-rmc")), flush=True)

    for index, row in enumerate(matrix_rows(), 1):
        physical = ",".join(str(gpu) for gpu in row["physical_gpus"])
        visible = "CUDA_VISIBLE_DEVICES=" + physical
        case_dir = output_root / row["application"] / row["size"] / row["gpu_class"]
        binary = app_binary(row["application"])
        print("\n# Case {:02d}: {} {} {} | physical {} -> logical {}".format(index, row["application"], row["size"], row["gpu_class"], physical, row["logical_mapping"]), flush=True)

        args_a = app_args(row["application"], row["gpu_count"], row["size"], "A")
        dcgm_command = ["bash", str(PROFILE_SH), "--gpus", physical, "--fields", "pass1_core", "--interval-ms", "100", "--lead", "2", "--lag", "3", "--out", str(case_dir / "dcgm"), "--", str(binary), *args_a]
        print(visible + " " + command_text(dcgm_command), flush=True)
        print("# then: python3 {} parse <newest {}/timestamp>".format(METRICS_PY, case_dir / "dcgm"), flush=True)

        args_b = app_args(row["application"], row["gpu_count"], row["size"], "B")
        nsys_b = sudo_command("-E", "env", "CUDA_VISIBLE_DEVICES=" + physical, str(NSYS), "profile", "--force-overwrite=true", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none", "--gpu-metrics-devices=" + physical, "--gpu-metrics-set=file:" + str(IFACE_SET), "--gpu-metrics-frequency=10000", "-o", str(case_dir / "nsys-10khz/report"), "--", str(binary), *args_b)
        print(command_text(nsys_b), flush=True)
        print("# then: python3 {} --json {}/report.nsys-rep".format(NSYS_CHECK, case_dir / "nsys-10khz"), flush=True)

        for target in row["physical_gpus"][:2]:
            args_c = app_args(row["application"], row["gpu_count"], row["size"], "C")
            c_dir = case_dir / "gpu{}-100khz".format(target)
            nsys_c = sudo_command("-E", "env", "CUDA_VISIBLE_DEVICES=" + physical, str(NSYS), "profile", "--force-overwrite=true", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none", "--gpu-metrics-devices=" + str(target), "--gpu-metrics-set=file:" + str(IFACE_SET), "--gpu-metrics-frequency=100000", "-o", str(c_dir / "report"), "--", str(binary), *args_c)
            print(command_text(nsys_c), flush=True)
            print("# diagnose; if the 100-kHz stream overflows, preserve it and rerun at 50 kHz:", flush=True)
            print(command_text([sys.executable, str(NSYS_CHECK), "--json", str(c_dir / "report.nsys-rep")]), flush=True)
            c50_dir = case_dir / "gpu{}-50khz".format(target)
            nsys_c50 = list(nsys_c)
            nsys_c50[nsys_c50.index("--gpu-metrics-frequency=100000")] = "--gpu-metrics-frequency=50000"
            nsys_c50[nsys_c50.index("-o") + 1] = str(c50_dir / "report")
            print("# conditional fallback: " + command_text(nsys_c50), flush=True)

        ncu_args = app_args(row["application"], row["gpu_count"], row["size"], "D")
        ncu_dir = case_dir / "ncu-gpu{}".format(row["physical_gpus"][0])
        ncu_command = [sys.executable, str(NCU_RUNNER), "--ncu", str(NCU), "--run-dir", str(ncu_dir), "--merged-csv", str(ncu_dir / "ncu.csv"), "--stats-file", str(ncu_dir / "stats.txt"), "--runner-log", str(ncu_dir / "ncu_runner.log"), "--run-args", " ".join(shlex.quote(arg) for arg in ncu_args), "--kernel-name", KERNEL_NAMES[row["application"]], "--groups", GROUPS, "--cache-control", "all", "--gpu-index", str(row["physical_gpus"][0]), "--ncu-devices", "0", "--visible-devices", physical, "--target-gpc-mhz", "1590", "--clock-lock-mode", "external", "--sudo-mode", "always", "--timeout", "900", "--binary-name", BINARY_NAMES[row["application"]]]
        print(visible + " " + command_text(ncu_command), flush=True)
        print("# This one runner launches the five independent NCU groups: {}".format(GROUPS), flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-matrix", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print all commands without querying GPUs or executing tools")
    args = parser.parse_args(argv)
    rows = matrix_rows()
    print_matrix(rows)
    if args.print_matrix:
        return 0
    if args.dry_run:
        print_dry_run_commands(Path(args.output_root))
        return 0
    install_signal_handlers()
    campaign = Campaign(Path(args.output_root), args.resume)
    try:
        campaign.run()
    except KeyboardInterrupt as exc:
        print("[campaign] interrupted: {}".format(exc), file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print("[campaign] ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
