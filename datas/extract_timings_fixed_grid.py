#!/usr/bin/env python3
"""Extract ADDA timings from a fixed-grid benchmark directory.

Expected layout, e.g. for grid 256:

    fixed_grid=256/
        SEQ/
            Nm=1/.../log
            Nm=2/.../log
            ...
        MPI/
            Nproc=4/Nm=1/.../log
            Nproc=16/Nm=1/.../log
            Nproc=128/Nm=1/.../log
        OCL/
            Nm=1/.../log
            ...
        OCL_BLAS/
            Nm=1/.../log
            ...

Usage:
    python3 extract_timings_fixed_grid.py fixed_grid=256
    python3 extract_timings_fixed_grid.py fixed_grid=256 -o figure1_timings_grid256.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

FLOAT_RE = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"


def search_int(text: str, label: str) -> Optional[int]:
    m = re.search(rf"^\s*{re.escape(label)}\s*(\d+)\s*$", text, flags=re.MULTILINE)
    return int(m.group(1)) if m else None


def search_float(text: str, label: str) -> Optional[float]:
    m = re.search(rf"^\s*{re.escape(label)}\s*{FLOAT_RE}\s*$", text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    return {
        "iterations": search_int(text, "Total number of iterations:"),
        "total_wall_time": search_float(text, "Total wall time:"),
        "initialization_time": search_float(text, "Initialization time:"),
        "internal_fields": search_float(text, "Internal fields:"),
        "matvec_products": search_float(text, "matvec products:"),
        "vector_readback": search_float(text, "vector readback:"),
        "scattered_fields": search_float(text, "Scattered fields:"),
        "other_sc_quantities": search_float(text, "Other sc.quantities:"),
    }


def nm_value(path: Path) -> int:
    m = re.fullmatch(r"Nm=(\d+)", path.name)
    if not m:
        raise ValueError(f"Invalid Nm directory name: {path}")
    return int(m.group(1))


def find_one_log(nm_dir: Path) -> Optional[Path]:
    logs = sorted(nm_dir.glob("run*/log"))
    if not logs:
        print(f"[WARNING] No log found in {nm_dir}", file=sys.stderr)
        return None
    if len(logs) > 1:
        print(f"[WARNING] Several logs found in {nm_dir}; skipping:\n  " + "\n  ".join(map(str, logs)), file=sys.stderr)
        return None
    return logs[0]


def collect_group(base_dir: Path, family: str, backend: str, nproc: Optional[int]) -> list[dict]:
    rows = []
    if not base_dir.is_dir():
        print(f"[SKIP] Directory not found: {base_dir}", file=sys.stderr)
        return rows

    nm_dirs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"Nm=\d+", p.name)],
        key=nm_value,
    )

    for nm_dir in nm_dirs:
        log_path = find_one_log(nm_dir)
        if log_path is None:
            continue

        values = parse_log(log_path)
        row = {
            "family": family,
            "backend": backend,
            "nproc": nproc,
            "Nm": nm_value(nm_dir),
            **values,
            "log_file": str(log_path),
        }

        required = [
            "iterations",
            "total_wall_time",
            "initialization_time",
            "internal_fields",
            "matvec_products",
            "scattered_fields",
            "other_sc_quantities",
        ]
        missing = [name for name in required if row[name] is None]
        if missing:
            print(f"[WARNING] Missing field(s) in {log_path}: " + ", ".join(missing), file=sys.stderr)

        rows.append(row)

    return rows


def collect_all(root: Path) -> list[dict]:
    rows = []

    rows += collect_group(root / "SEQ", family="SEQ", backend="SEQ", nproc=None)
    rows += collect_group(root / "OCL", family="GPU", backend="OCL", nproc=None)
    rows += collect_group(root / "OCL_BLAS", family="GPU", backend="OCL_BLAS", nproc=None)

    for nproc in (4, 16, 128):
        rows += collect_group(root / "MPI" / f"Nproc={nproc}", family="MPI", backend="MPI", nproc=nproc)

    backend_order = {"SEQ": 0, "MPI": 1, "OCL": 2, "OCL_BLAS": 3}
    rows.sort(key=lambda r: (backend_order.get(r["backend"], 99), r["nproc"] or 0, r["Nm"]))
    return rows


FIELDNAMES = [
    "family",
    "backend",
    "nproc",
    "Nm",
    "iterations",
    "total_wall_time",
    "initialization_time",
    "internal_fields",
    "matvec_products",
    "vector_readback",
    "scattered_fields",
    "other_sc_quantities",
    "log_file",
]


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ADDA timings from a fixed-grid benchmark directory.")
    parser.add_argument("root", type=Path, help="Benchmark root, e.g. fixed_grid=256")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output CSV")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"Root directory does not exist: {args.root}")

    output = args.output or (args.root / "figure1_timings.csv")
    rows = collect_all(args.root)
    write_csv(rows, output)

    counts = {}
    for row in rows:
        key = f"MPI-{row['nproc']}" if row["family"] == "MPI" else row["backend"]
        counts[key] = counts.get(key, 0) + 1

    print(f"Wrote {len(rows)} rows to: {output}")
    for key, count in counts.items():
        print(f"  {key:10s}: {count} rows")


if __name__ == "__main__":
    main()
