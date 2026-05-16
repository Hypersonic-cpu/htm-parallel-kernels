#!/usr/bin/env python3
"""Pack cal_kernels run artifacts into a directory or .tar.gz archive."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


KERNEL_DIR_RE = re.compile(r"^(?:\d+[A-Za-z]*_|[a-z][A-Za-z0-9]*_|REAL_)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy every immediate numbered cal_kernels/*/run* directory into "
            "DEST, preserving paths such as "
            "DEST/a_float_op_latency/run-GV100/native."
        )
    )
    parser.add_argument(
        "dest",
        help=(
            "Destination folder name. With --compress, this may be either a "
            "package base name or a .tar.gz/.tgz path."
        ),
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="write DEST.tar.gz instead of leaving an unpacked destination folder",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing destination folder or archive",
    )
    return parser.parse_args()


def discover_run_dirs(cal_kernels_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for kernel_dir in sorted(cal_kernels_dir.iterdir()):
        if not kernel_dir.is_dir() or not KERNEL_DIR_RE.match(kernel_dir.name):
            continue
        run_dirs.extend(
            run_dir
            for run_dir in sorted(kernel_dir.glob("run-*"))
            if run_dir.is_dir()
        )
    return run_dirs


def safe_remove(path: Path) -> None:
    resolved = path.resolve()
    dangerous = {Path("/"), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in dangerous or len(resolved.parts) < 3:
        raise SystemExit(f"refusing to remove unsafe path: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_run_dirs(run_dirs: list[Path], cal_kernels_dir: Path, dest_root: Path) -> None:
    for run_dir in run_dirs:
        rel_path = run_dir.relative_to(cal_kernels_dir)
        target = dest_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_dir, target, symlinks=True)
        print(f"copied {rel_path}")


def compressed_paths(dest_arg: str) -> tuple[Path, str]:
    dest = Path(dest_arg).expanduser()
    if dest.name.endswith(".tar.gz"):
        return dest, dest.name[: -len(".tar.gz")]
    if dest.suffix == ".tgz":
        return dest, dest.stem
    return dest.with_name(dest.name + ".tar.gz"), dest.name


def main() -> int:
    args = parse_args()
    cal_kernels_dir = Path(__file__).resolve().parent
    run_dirs = discover_run_dirs(cal_kernels_dir)
    if not run_dirs:
        print("ERROR: found no numbered cal_kernels/*/run* directories", file=sys.stderr)
        return 1

    if args.compress:
        archive_path, root_name = compressed_paths(args.dest)
        archive_path = archive_path.resolve()
        if archive_path.exists():
            if not args.force:
                print(f"ERROR: destination exists: {archive_path}", file=sys.stderr)
                print("Use --force to replace it.", file=sys.stderr)
                return 1
            safe_remove(archive_path)

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".pack_runs.", dir=str(archive_path.parent)
        ) as temp_dir:
            staging_root = Path(temp_dir) / root_name
            copy_run_dirs(run_dirs, cal_kernels_dir, staging_root)
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(staging_root, arcname=root_name)
        print(f"wrote archive {archive_path}")
    else:
        dest_root = Path(args.dest).expanduser().resolve()
        if dest_root.exists():
            if not args.force:
                print(f"ERROR: destination exists: {dest_root}", file=sys.stderr)
                print("Use --force to replace it.", file=sys.stderr)
                return 1
            safe_remove(dest_root)

        copy_run_dirs(run_dirs, cal_kernels_dir, dest_root)
        print(f"wrote directory {dest_root}")

    print(f"packed {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
