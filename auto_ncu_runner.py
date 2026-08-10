#!/usr/bin/env python3
"""
Sequential NCU runner for htm-parallel-kernels/cal_kernels.

Design goals:
  * Do not call cal_kernels/Makefile; call each bench's own Makefile directly.
  * Read a YAML task file with multiple application cases.
  * Run cases strictly sequentially.
  * Put each case in its own RUN_ROOT/RUN_SUBDIR output directory.
  * Resume automatically: a completed case is skipped unless --force-refresh is used.
  * Use file locks to avoid two runner processes touching the same output root/case.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    print(
        "ERROR: PyYAML is required because the task file is YAML and may use anchors.\n"
        "Install it with: python3 -m pip install --user pyyaml",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

DONE_FILE = ".htm_ncu.done.json"
CASE_LOCK_FILE = ".htm_ncu.case.lock"
GLOBAL_LOCK_FILE = ".htm_ncu.runner.lock"
RUNNER_LOG = ".htm_ncu.runner.log"
SCHEMA_VERSION = 2


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def expand_pathish(value: Any) -> str:
    return os.path.expandvars(os.path.expanduser(str(value)))


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_obj(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def sanitize_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._+=,@:-]+", "_", text)
    return text.strip("._-") or "case"


def normalize_args(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(shlex.quote(str(x)) for x in value)
    return str(value)


def normalize_env_dict(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"env/make_vars must be a mapping, got {type(value).__name__}")
    return {str(k): expand_pathish(v) for k, v in value.items()}


class FileLock:
    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self.fd: Optional[Any] = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = self.path.open("a+")
        try:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another process already holds {self.label}: {self.path}")
        self.fd.seek(0)
        self.fd.truncate()
        self.fd.write(f"pid={os.getpid()} time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.fd.flush()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is not None:
            with contextlib.suppress(Exception):
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(Exception):
                self.fd.close()


@dataclass(frozen=True)
class Case:
    bench: str
    case_name: str
    run_args: str
    arch: str
    conf: str
    make_vars: Dict[str, str]
    env: Dict[str, str]
    report: bool
    raw: Dict[str, Any]


def choose_group(config: Mapping[str, Any], requested: Optional[str]) -> Tuple[str, Mapping[str, Any]]:
    if requested:
        if requested not in config:
            raise KeyError(f"group {requested!r} not found; available: {', '.join(config.keys())}")
        group = config[requested]
        if not isinstance(group, Mapping):
            raise TypeError(f"group {requested!r} is not a mapping")
        return requested, group

    candidates = [(k, v) for k, v in config.items() if isinstance(v, Mapping) and "execs" in v]
    if len(candidates) != 1:
        names = ", ".join(k for k, _ in candidates) or "<none>"
        raise ValueError(f"YAML has {len(candidates)} runnable groups ({names}); pass --group")
    return candidates[0][0], candidates[0][1]


def read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise TypeError("top-level YAML must be a mapping")
    return data


def default_cal_root() -> Path:
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent
    for base in (cwd, script_dir):
        if (base / "common.mk").is_file() and base.name == "cal_kernels":
            return base
        if (base / "cal_kernels" / "common.mk").is_file():
            return (base / "cal_kernels").resolve()
    raise RuntimeError("cannot infer cal_kernels root; pass --cal-root or set cal_root in YAML")


def group_path(group: Mapping[str, Any], cli_value: Optional[str], keys: Iterable[str], fallback: Optional[Path]) -> Path:
    if cli_value:
        return Path(expand_pathish(cli_value)).resolve()
    for key in keys:
        if key in group and group[key] is not None:
            return Path(expand_pathish(group[key])).resolve()
    if fallback is None:
        raise RuntimeError(f"missing required path; tried keys {list(keys)}")
    return fallback.resolve()


def parse_cases(group: Mapping[str, Any], default_arch: str, default_conf: str, cli_no_report: bool) -> List[Case]:
    execs = group.get("execs")
    if not isinstance(execs, list):
        raise TypeError("group.execs must be a list like: - BENCH: [cases]")

    group_make_vars = normalize_env_dict(group.get("make_vars"))
    group_env = normalize_env_dict(group.get("env"))
    group_report = bool(group.get("report", True)) and not cli_no_report
    group_arch = str(group.get("arch", default_arch))
    group_conf = str(group.get("conf", default_conf))

    out: List[Case] = []
    seen_names: Dict[str, str] = {}

    for entry in execs:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise TypeError(f"each exec entry must be a one-key mapping, got: {entry!r}")
        bench, cases_obj = next(iter(entry.items()))
        bench = str(bench)
        if isinstance(cases_obj, Mapping):
            cases_list = [cases_obj]
        elif isinstance(cases_obj, list):
            cases_list = cases_obj
        else:
            raise TypeError(f"cases for {bench} must be a list or mapping")

        for idx, raw_case in enumerate(cases_list):
            if raw_case is None:
                raw_case = {}
            if not isinstance(raw_case, Mapping):
                raise TypeError(f"case {bench}[{idx}] must be a mapping")
            raw: Dict[str, Any] = dict(raw_case)
            run_args = normalize_args(raw.get("args", raw.get("run_args", "")))
            case_name = str(
                raw.get("case_name")
                or raw.get("run_name")
                or raw.get("trace_name")
                or f"{bench}-{sanitize_name(run_args) if run_args else idx}"
            )
            case_name = sanitize_name(case_name)
            if case_name in seen_names:
                raise ValueError(f"duplicate case_name {case_name!r}: {seen_names[case_name]} and {bench}[{idx}]")
            seen_names[case_name] = f"{bench}[{idx}]"

            make_vars = dict(group_make_vars)
            make_vars.update(normalize_env_dict(raw.get("make_vars")))

            # Convenient aliases for common Makefile variables.
            if "ncu_args" in raw and "NCU_RUN_ARGS" not in make_vars:
                make_vars["NCU_RUN_ARGS"] = normalize_args(raw["ncu_args"])
            if "ncu_run_args" in raw and "NCU_RUN_ARGS" not in make_vars:
                make_vars["NCU_RUN_ARGS"] = normalize_args(raw["ncu_run_args"])
            if "metrics" in raw and "NATIVE_METRICS" not in make_vars:
                make_vars["NATIVE_METRICS"] = normalize_args(raw["metrics"])
            if "native_metrics" in raw and "NATIVE_METRICS" not in make_vars:
                make_vars["NATIVE_METRICS"] = normalize_args(raw["native_metrics"])
            if "timeout" in raw and "RUN_TIMEOUT" not in make_vars:
                make_vars["RUN_TIMEOUT"] = str(raw["timeout"])

            env = dict(group_env)
            env.update(normalize_env_dict(raw.get("env")))
            # Accept trace_env from the existing trace YAML shape, but do not require it.
            if bool(group.get("export_trace_env", False)):
                env.update(normalize_env_dict(raw.get("trace_env")))

            report = bool(raw.get("report", group_report)) and not cli_no_report
            arch = str(raw.get("arch", group_arch))
            conf = str(raw.get("conf", group_conf))
            out.append(Case(bench, case_name, run_args, arch, conf, make_vars, env, report, raw))
    return out


def make_command(case: Case, cal_root: Path, repo_root: Path, output_root: Path, target: str) -> List[str]:
    bench_dir = cal_root / case.bench
    cmd = [
        "make",
        "-C",
        str(bench_dir),
        f"REPO_ROOT={repo_root}",
        f"ARCH={case.arch}",
        f"CONF={case.conf}",
        f"RUN_ROOT={output_root}",
        f"RUN_SUBDIR={case.case_name}",
        f"RUN_ARGS={case.run_args}",
    ]
    for k in sorted(case.make_vars):
        cmd.append(f"{k}={case.make_vars[k]}")
    cmd.append(target)
    return cmd


def command_identity(case: Case, cal_root: Path, repo_root: Path, output_root: Path, targets: List[str]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "bench": case.bench,
        "case_name": case.case_name,
        "run_args": case.run_args,
        "arch": case.arch,
        "conf": case.conf,
        "make_vars": case.make_vars,
        "env": case.env,
        "cal_root": str(cal_root),
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "targets": targets,
    }


def has_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def completion_status(case_dir: Path, expected_digest: str, need_report: bool) -> Tuple[bool, str]:
    done_path = case_dir / DONE_FILE
    if not done_path.is_file():
        return False, "missing done marker"
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"cannot read done marker: {exc}"
    if done.get("command_digest") != expected_digest:
        return False, "done marker command digest mismatch"
    if not has_nonempty_file(case_dir / "stats.txt"):
        return False, "missing/nonempty stats.txt"
    if not has_nonempty_file(case_dir / "ncu.csv"):
        return False, "missing/nonempty ncu.csv"
    if not has_nonempty_file(case_dir / "ncu_groups" / "manifest.json"):
        return False, "missing/nonempty ncu_groups/manifest.json"
    if need_report and not has_nonempty_file(case_dir / "report.csv"):
        return False, "missing/nonempty report.csv"
    return True, "complete"


def append_runner_log(case_dir: Path, message: str) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / RUNNER_LOG).open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def run_one_command(cmd: List[str], env: Mapping[str, str], cwd: Path, dry_run: bool) -> int:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"+ {printable}", flush=True)
    if dry_run:
        return 0
    run_env = os.environ.copy()
    run_env.update(env)
    proc = subprocess.run(cmd, cwd=str(cwd), env=run_env)
    return int(proc.returncode)


def validate_case(case: Case, cal_root: Path) -> None:
    bench_dir = cal_root / case.bench
    if not bench_dir.is_dir():
        raise FileNotFoundError(f"missing bench directory: {bench_dir}")
    if not (bench_dir / "Makefile").is_file():
        raise FileNotFoundError(f"missing bench Makefile: {bench_dir / 'Makefile'}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential resumable NCU runner for cal_kernels")
    parser.add_argument("-c", "--config", required=True, help="YAML task file")
    parser.add_argument("-g", "--group", help="top-level YAML group to run")
    parser.add_argument("--cal-root", help="path to htm-parallel-kernels/cal_kernels")
    parser.add_argument("--repo-root", help="path to htm-parallel-kernels; default: parent of cal-root")
    parser.add_argument("--output-root", help="absolute or relative RUN_ROOT for all cases")
    parser.add_argument("--arch", default="perf", help="Makefile ARCH value; default: perf")
    parser.add_argument("--conf", default="GH100", help="Makefile CONF value; default: GH100")
    parser.add_argument(
        "--gpu",
        default=os.environ.get("NCU_GPU", "0"),
        help="physical nvidia-smi GPU index for the external GPC lock; default: NCU_GPU or 0",
    )
    parser.add_argument(
        "--visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        help="CUDA_VISIBLE_DEVICES value inherited by each profiled application",
    )
    parser.add_argument(
        "--gpc-clock-mhz",
        default=os.environ.get("NCU_GPC_CLOCK_MHZ", "1590"),
        help="locked GPC clock in MHz; default: NCU_GPC_CLOCK_MHZ or 1590",
    )
    parser.add_argument(
        "--ncu-groups",
        default=os.environ.get(
            "NCU_GROUPS", "timing_clock,l1_l2,dram,execution,scheduler_stall"
        ),
        help="comma-separated grouped NCU workflow names",
    )
    parser.add_argument(
        "--ncu-sudo-mode",
        choices=["auto", "never", "always"],
        default=os.environ.get("NCU_SUDO", "auto"),
        help="NCU privilege mode: auto retries as root on ERR_NVGPUCTRPERM",
    )
    parser.add_argument("--only", action="append", default=[], help="run only matching bench or case_name; repeatable")
    parser.add_argument("--skip", action="append", default=[], help="skip matching bench or case_name; repeatable")
    parser.add_argument("--force-refresh", action="store_true", help="rerun even if done marker exists")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed case")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing")
    parser.add_argument("--no-report", action="store_true", help="run NCU but skip make report")
    args = parser.parse_args(argv)

    config_path = Path(expand_pathish(args.config)).resolve()
    config = read_yaml(config_path)
    group_name, group = choose_group(config, args.group)

    cal_root = group_path(group, args.cal_root, ("cal_root", "kernel_root", "kernels_root"), None if args.cal_root else default_cal_root())
    repo_root = group_path(group, args.repo_root, ("repo_root",), cal_root.parent)
    output_root = group_path(group, args.output_root, ("output_root", "run_root"), cal_root / "run-perf-auto")
    output_root.mkdir(parents=True, exist_ok=True)

    cases = parse_cases(group, args.arch, args.conf, args.no_report)
    if args.only:
        pats = args.only
        cases = [c for c in cases if any(p in c.bench or p in c.case_name for p in pats)]
    if args.skip:
        pats = args.skip
        cases = [c for c in cases if not any(p in c.bench or p in c.case_name for p in pats)]
    if not cases:
        print("No cases selected.")
        return 0

    # Include hardware/profile identity in the resumable command digest.  A
    # completed result from GPU 0 at one locked clock must not be reused for a
    # different GPU, visibility mask, clock target, or metric-group plan.
    for case in cases:
        case.make_vars.setdefault("NCU_GPU", str(args.gpu))
        case.make_vars.setdefault("NCU_GPC_CLOCK_MHZ", str(args.gpc_clock_mhz))
        case.make_vars.setdefault("NCU_GROUPS", str(args.ncu_groups))
        case.make_vars.setdefault("NCU_SUDO", str(args.ncu_sudo_mode))
        case.make_vars.setdefault("NCU_WORKFLOW_VERSION", str(SCHEMA_VERSION))
        if args.visible_devices:
            case.make_vars.setdefault("NCU_VISIBLE_DEVICES", str(args.visible_devices))

    print(f"YAML group : {group_name}")
    print(f"cal_root   : {cal_root}")
    print(f"repo_root  : {repo_root}")
    print(f"output_root: {output_root}")
    print(f"cases      : {len(cases)}")

    for case in cases:
        validate_case(case, cal_root)

    failures: List[Tuple[Case, int]] = []
    global_lock_path = output_root / GLOBAL_LOCK_FILE
    try:
        lock_ctx = FileLock(global_lock_path, "global NCU runner lock")
        with lock_ctx:
            for idx, case in enumerate(cases, 1):
                case_dir = output_root / case.case_name
                targets = ["run"] + (["report"] if case.report else [])
                ident = command_identity(case, cal_root, repo_root, output_root, targets)
                ident_digest = digest_obj(ident)

                print(f"\n[{idx}/{len(cases)}] {case.bench} :: {case.case_name}", flush=True)

                if not args.force_refresh:
                    ok, reason = completion_status(case_dir, ident_digest, case.report)
                    if ok:
                        print(f"SKIP complete: {case_dir}", flush=True)
                        continue
                    print(f"RUN needed: {reason}", flush=True)
                else:
                    print("FORCE refresh requested; rerunning.", flush=True)
                    with contextlib.suppress(FileNotFoundError):
                        (case_dir / DONE_FILE).unlink()

                case_dir.mkdir(parents=True, exist_ok=True)
                try:
                    case_lock = FileLock(case_dir / CASE_LOCK_FILE, f"case lock {case.case_name}")
                    with case_lock:
                        append_runner_log(case_dir, f"START bench={case.bench} case={case.case_name} digest={ident_digest}")
                        rc = 0
                        for target in targets:
                            cmd = make_command(case, cal_root, repo_root, output_root, target)
                            append_runner_log(case_dir, "CMD " + " ".join(shlex.quote(x) for x in cmd))
                            rc = run_one_command(cmd, case.env, cwd=cal_root, dry_run=args.dry_run)
                            append_runner_log(case_dir, f"RC target={target} rc={rc}")
                            if rc != 0:
                                break

                        if rc == 0 and not args.dry_run:
                            done = {
                                "schema_version": SCHEMA_VERSION,
                                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "command_digest": ident_digest,
                                "command_identity": ident,
                            }
                            (case_dir / DONE_FILE).write_text(json.dumps(done, indent=2, sort_keys=True), encoding="utf-8")
                            append_runner_log(case_dir, "DONE")
                            print(f"DONE: {case_dir}", flush=True)
                        elif rc == 0 and args.dry_run:
                            print("DRY-RUN OK", flush=True)
                        else:
                            failures.append((case, rc))
                            print(f"FAILED rc={rc}: {case.bench} :: {case.case_name}", flush=True)
                            if not args.keep_going:
                                break
                except RuntimeError as exc:
                    failures.append((case, 125))
                    eprint(f"LOCK ERROR: {exc}")
                    if not args.keep_going:
                        break
    except RuntimeError as exc:
        eprint(f"LOCK ERROR: {exc}")
        return 125

    if failures:
        eprint("\nFailed cases:")
        for case, rc in failures:
            eprint(f"  rc={rc}: {case.bench} :: {case.case_name}")
        return failures[0][1] if failures[0][1] else 1

    print("\nAll selected cases are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
