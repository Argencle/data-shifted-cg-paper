#!/usr/bin/env python3
"""
Extract ADDA timings from Figure2 benchmark directories into one tidy CSV.

Expected directory layout (root = data-shifted-m-paper/Figure2):

    fixed_Nm_GPU/
        OCL_BLAS/
            Nm=1/.../log
            Nm=2/.../log
            ...
        OCL/
            Nm=.../.../log

    fixed_Nm_SEQ/
        Nm=1/.../log
        ...

    fixed_Nm_MPI/
        Nproc=4/
            Nm=1/.../log
            ...
        Nproc=16/
        Nproc=128/

Usage:
    python3 extract_timings_Figure2.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

FLOAT_RE = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"


def search_int(text: str, label: str) -> Optional[int]:
    """Return the first integer found after an exact label."""
    m = re.search(
        rf"^\s*{re.escape(label)}\s*(\d+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return int(m.group(1)) if m else None


def search_float(text: str, label: str) -> Optional[float]:
    """
    Return the first floating-point value found after an exact label.

    This is intentional for 'matvec products:' because ADDA logs can contain
    both:
        matvec products: ...
    and, later:
        one iteration:
            matvec products: ...

    The first occurrence is the TOTAL matvec-products timing for the solve,
    which is the value wanted here.
    """
    m = re.search(
        rf"^\s*{re.escape(label)}\s*{FLOAT_RE}\s*$",
        text,
        flags=re.MULTILINE,
    )
    return float(m.group(1)) if m else None


def parse_log(log_path: Path) -> dict:
    """Parse one ADDA log file."""
    text = log_path.read_text(errors="replace")

    return {
        "iterations": search_int(text, "Total number of iterations:"),
        "total_wall_time": search_float(text, "Total wall time:"),
        "initialization_time": search_float(text, "Initialization time:"),
        "internal_fields": search_float(text, "Internal fields:"),
        "matvec_products": search_float(text, "matvec products:"),
        # Present for GPU OCL/OCL_BLAS logs; absent for SEQ/MPI -> blank in CSV
        "vector_readback": search_float(text, "vector readback:"),
        "scattered_fields": search_float(text, "Scattered fields:"),
        "other_sc_quantities": search_float(text, "Other sc.quantities:"),
    }


# ---------------------------------------------------------------------------
# Directory traversal
# ---------------------------------------------------------------------------


def nm_value(path: Path) -> int:
    """Extract integer N from a directory called Nm=N."""
    m = re.fullmatch(r"Nm=(\d+)", path.name)
    if not m:
        raise ValueError(f"Invalid Nm directory name: {path}")
    return int(m.group(1))


def find_one_log(nm_dir: Path) -> Optional[Path]:
    """
    Find the ADDA log inside one Nm directory.

    Expected pattern:
        Nm=X/run.../log

    If several logs are found, do not silently choose one.
    """
    logs = sorted(nm_dir.glob("run*/log"))

    if not logs:
        print(f"[WARNING] No log found in {nm_dir}", file=sys.stderr)
        return None

    if len(logs) > 1:
        print(
            f"[WARNING] Several logs found in {nm_dir}; skipping to avoid "
            f"an ambiguous result:\n  " + "\n  ".join(map(str, logs)),
            file=sys.stderr,
        )
        return None

    return logs[0]


def collect_group(
    base_dir: Path,
    family: str,
    backend: str,
    nproc: Optional[int],
) -> list[dict]:
    """Collect all Nm logs from one benchmark group."""
    rows = []

    if not base_dir.is_dir():
        # This is normal for the future OCL placeholder.
        print(f"[SKIP] Directory not found: {base_dir}", file=sys.stderr)
        return rows

    nm_dirs = []
    for p in base_dir.iterdir():
        if p.is_dir() and re.fullmatch(r"Nm=\d+", p.name):
            nm_dirs.append(p)

    nm_dirs.sort(key=nm_value)

    for nm_dir in nm_dirs:
        log_path = find_one_log(nm_dir)
        if log_path is None:
            continue

        values = parse_log(log_path)

        row = {
            "family": family,  # GPU / SEQ / MPI
            "backend": backend,  # OCL_BLAS / OCL / SEQ / MPI
            "nproc": nproc,  # blank for GPU/SEQ
            "Nm": nm_value(nm_dir),
            **values,
            "log_file": str(log_path),
        }

        # Warn if a requested field other than vector_readback is missing.
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
            print(
                f"[WARNING] Missing field(s) in {log_path}: "
                + ", ".join(missing),
                file=sys.stderr,
            )

        rows.append(row)

    return rows


def collect_all(root: Path) -> list[dict]:
    """Collect GPU, SEQ and MPI results."""
    rows = []

    # GPU: OCL_BLAS now, OCL reserved for later.
    gpu_root = Path(root) / "fixed_Nm_GPU"
    rows += collect_group(
        gpu_root / "OCL_BLAS",
        family="GPU",
        backend="OCL_BLAS",
        nproc=None,
    )
    rows += collect_group(
        gpu_root / "OCL",
        family="GPU",
        backend="OCL",
        nproc=None,
    )

    # Sequential CPU.
    rows += collect_group(
        Path(root) / "fixed_Nm_SEQ",
        family="SEQ",
        backend="SEQ",
        nproc=None,
    )

    # MPI CPU.
    mpi_root = Path(root) / "fixed_Nm_MPI"
    for nproc in (4, 16, 128):
        rows += collect_group(
            mpi_root / f"Nproc={nproc}",
            family="MPI",
            backend="MPI",
            nproc=nproc,
        )

    # Stable plot-friendly order.
    family_order = {"GPU": 0, "SEQ": 1, "MPI": 2}
    backend_order = {"OCL_BLAS": 0, "OCL": 1, "SEQ": 2, "MPI": 3}

    rows.sort(
        key=lambda r: (
            family_order.get(r["family"], 99),
            backend_order.get(r["backend"], 99),
            r["nproc"] if r["nproc"] is not None else 0,
            r["Nm"],
        )
    )

    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

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

        for row in rows:
            # csv writes None as an empty field, ideal for vector_readback
            # when that timing does not exist in SEQ/MPI logs.
            writer.writerow(row)


def main() -> None:

    root = Path(__file__).resolve().parents[1] / "datas" / "Figure2"
    filename = "figure2_timings.csv"
    output = root / filename

    if not Path(root).is_dir():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    rows = collect_all(root)
    write_csv(rows, output)

    counts = {}
    for row in rows:
        if row["family"] == "MPI":
            key = f"MPI-{row['nproc']}"
        else:
            key = row["backend"]
        counts[key] = counts.get(key, 0) + 1

    print(f"Wrote {len(rows)} rows to: {output}")
    for key, count in counts.items():
        print(f"  {key:10s}: {count} rows")


if __name__ == "__main__":
    main()
