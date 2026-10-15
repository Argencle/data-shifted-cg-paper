#!/usr/bin/env python3
"""
Extract Figure 1 (2D grid of refractive indices) ADDA results.

The script automatically discovers:
  - independent runs, with either naming convention:

        legacy / MPI style:
        .../independent/all/1.300_0.100/.../log
        .../independent/all/1.300_0.125/.../log

        indexed / SEQ style:
        .../independent/all/run000_sphere_g128_m1.3/.../log
        .../independent/all/run001_sphere_g128_m1.3/.../log
        ...

    For indexed runXXX directories, runXXX is matched to the corresponding
    shift_index in the sibling shifted_convergence.dat.  If that file is not
    available, the script falls back to the known Figure-1 imaginary-index
    grid 0.100:0.025:0.600.
  - shifted runs:
        .../shifted/run.../log
        .../shifted/run.../shifted_convergence.dat

It supports SEQ, GPU (OCL_BLAS / OCL) and MPI (e.g. 4, 16, 128 ranks)
as long as those names appear in the directory path.

For independent runs:
  - iterations are read from "Total number of iterations:" in each log
  - final_residual is the LAST "RE_<n> = ..." found before "Timing Results"
  - timings are read from that same individual log

For shifted runs:
  - per-index iterations and residuals are read from shifted_convergence.dat
  - global timings are read from the single shifted log
  - those global timing values are repeated on each per-index CSV row
  - timing_scope is set to "shared_shifted_run" to make this explicit
  - run_total_iterations stores the global iteration count from the shifted log

Output:
  figure1_grid2d.csv

Usage:
  python3 extract_grid2d_Figure1.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Optional

# ============================================================
# Generic parsing helpers
# ============================================================

FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"

M_DIR_RE = re.compile(rf"^(?P<real>{FLOAT_RE})_(?P<imag>{FLOAT_RE})$")

# New independent-run naming used e.g. by the SEQ dataset:
#   run000_sphere_g128_m1.3
#   run001_sphere_g128_m1.3
#   ...
# The directory name contains Re(m), while Im(m) is recovered from the
# matching shift_index in shifted_convergence.dat (preferred), or from the
# Figure-1 grid as a fallback.
RUN_DIR_RE = re.compile(rf"^run(?P<run_index>\d+).*?_m(?P<real>{FLOAT_RE})$")

# Fallback only, used when no sibling shifted_convergence.dat can be found.
FIG1_IMAG_MIN = 0.100
FIG1_IMAG_STEP = 0.025
FIG1_N_IMAG = 21

NPROC_RE = re.compile(r"^Nproc=(\d+)$")


def to_float(value: str) -> float:
    """Convert normal E/e or Fortran D/d scientific notation."""
    return float(value.replace("D", "E").replace("d", "e"))


def search_int(text: str, label: str) -> Optional[int]:
    m = re.search(
        rf"^\s*{re.escape(label)}\s*(\d+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return int(m.group(1)) if m else None


def search_float(text: str, label: str) -> Optional[float]:
    """
    Return the first occurrence of a timing field.

    For 'matvec products:' this deliberately gets the first/global solver
    occurrence, not the nested value under 'one iteration:'.
    """
    m = re.search(
        rf"^\s*{re.escape(label)}\s*({FLOAT_RE})\s*$",
        text,
        flags=re.MULTILINE,
    )
    return to_float(m.group(1)) if m else None


def parse_last_re_residual(text: str) -> tuple[Optional[int], Optional[float]]:
    """
    Return the last residual line of the form:

        RE_256 = 9.9585173178023347E-06  +   progress = ...

    Only the part before "Timing Results" is searched when that marker exists.
    """
    marker = text.find("Timing Results")
    search_text = text[:marker] if marker != -1 else text

    matches = list(
        re.finditer(
            rf"^\s*RE_(\d+)\s*=\s*({FLOAT_RE})\b",
            search_text,
            flags=re.MULTILINE,
        )
    )

    if not matches:
        return None, None

    last = matches[-1]
    return int(last.group(1)), to_float(last.group(2))


def parse_log(log_path: Path) -> dict:
    """Parse the timing block and final RE residual from one ADDA log."""
    text = log_path.read_text(errors="replace")

    residual_iteration, final_residual = parse_last_re_residual(text)

    return {
        "run_total_iterations": search_int(
            text, "Total number of iterations:"
        ),
        "residual_iteration": residual_iteration,
        "final_residual": final_residual,
        "total_wall_time": search_float(text, "Total wall time:"),
        "initialization_time": search_float(text, "Initialization time:"),
        "internal_fields": search_float(text, "Internal fields:"),
        "matvec_products": search_float(text, "matvec products:"),
        "vector_readback": search_float(text, "vector readback:"),
        "scattered_fields": search_float(text, "Scattered fields:"),
        "other_sc_quantities": search_float(text, "Other sc.quantities:"),
    }


# ============================================================
# Metadata inferred from directory path
# ============================================================


def infer_metadata(path: Path) -> dict:
    """
    Infer family/backend/nproc from path names such as:

        Figure1/2Dgrid_MPI/Nproc=128/...
        Figure1/2Dgrid_SEQ/...
        Figure1/2Dgrid_GPU/OCL_BLAS/...
        Figure1/2Dgrid_GPU/OCL/...
    """
    parts = list(path.parts)

    nproc = None
    for part in parts:
        m = NPROC_RE.fullmatch(part)
        if m:
            nproc = int(m.group(1))

    if "2Dgrid_MPI" in parts:
        family = "MPI"
        backend = "MPI"

    elif "2Dgrid_SEQ" in parts:
        family = "SEQ"
        backend = "SEQ"

    elif "2Dgrid_GPU" in parts:
        family = "GPU"

        if "OCL_BLAS" in parts:
            backend = "OCL_BLAS"
        elif "OCL" in parts:
            backend = "OCL"
        else:
            backend = "GPU"

    else:
        # Fallback for slightly different future directory names.
        joined = "/".join(parts).upper()

        if "MPI" in joined:
            family = "MPI"
            backend = "MPI"
        elif "OCL_BLAS" in joined:
            family = "GPU"
            backend = "OCL_BLAS"
        elif "OCL" in joined or "GPU" in joined:
            family = "GPU"
            backend = "OCL" if "OCL" in joined else "GPU"
        elif "SEQ" in joined:
            family = "SEQ"
            backend = "SEQ"
        elif "EPS16" in joined:
            family = "MPI"
            backend = "eps16"
        else:
            family = "UNKNOWN"
            backend = "UNKNOWN"

    return {
        "family": family,
        "backend": backend,
        "nproc": nproc,
    }


# ============================================================
# shifted_convergence.dat
# ============================================================


def parse_shifted_convergence(path: Path) -> list[dict]:
    """
    Parse shifted_convergence.dat.

    Expected columns:
      1  evaluation
      2  alpha
      3  beta
      4  gamma
      5  polarization
      6  shifted-system index
      7  Re(m)
      8  Im(m)
      9  convergence iteration (-1 if not converged)
      10 estimated final relative residual norm
    """
    rows = []

    with path.open(errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            fields = stripped.split()

            if len(fields) < 10:
                print(
                    f"[WARNING] {path}:{lineno}: expected >=10 columns, "
                    f"got {len(fields)}; skipping line",
                    file=sys.stderr,
                )
                continue

            try:
                row = {
                    "evaluation": int(fields[0]),
                    "alpha": to_float(fields[1]),
                    "beta": to_float(fields[2]),
                    "gamma": to_float(fields[3]),
                    "polarization": fields[4],
                    "shift_index": int(fields[5]),
                    "m_real": to_float(fields[6]),
                    "m_imag": to_float(fields[7]),
                    "iterations": int(fields[8]),
                    "final_residual": to_float(fields[9]),
                }
            except ValueError as exc:
                print(
                    f"[WARNING] {path}:{lineno}: cannot parse line: {exc}",
                    file=sys.stderr,
                )
                continue

            rows.append(row)

    return rows


# ============================================================
# File discovery
# ============================================================


def find_logs_below(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("log") if p.is_file())


def find_one_log(directory: Path, description: str) -> Optional[Path]:
    """
    Find one log recursively below directory.

    If several logs exist, do not silently pick one.
    """
    logs = find_logs_below(directory)

    if not logs:
        print(
            f"[WARNING] No log found for {description}: {directory}",
            file=sys.stderr,
        )
        return None

    if len(logs) > 1:
        print(
            f"[WARNING] Several logs found for {description}; skipping "
            f"ambiguous case:\n  " + "\n  ".join(map(str, logs)),
            file=sys.stderr,
        )
        return None

    return logs[0]


def build_shift_index_map(
    independent_dir: Path,
) -> dict[int, tuple[float, float]]:
    """
    Build {shift_index: (Re(m), Im(m))} from the sibling shifted dataset.

    Typical layouts are:

        2Dgrid_SEQ/independent/...
        2Dgrid_SEQ/shifted/.../shifted_convergence.dat

    and

        2Dgrid_MPI/Nproc=128/independent/...
        2Dgrid_MPI/Nproc=128/shifted/.../shifted_convergence.dat

    This is the preferred way to decode directories such as
    run000_sphere_g128_m1.3 because the run index is the same enumeration
    as shift_index in shifted_convergence.dat.
    """
    shifted_dir = independent_dir.parent / "shifted"

    if not shifted_dir.is_dir():
        return {}

    conv_files = sorted(
        p for p in shifted_dir.rglob("shifted_convergence.dat") if p.is_file()
    )

    if not conv_files:
        return {}

    # Usually there is exactly one shifted run.  If several are present,
    # accept them only when they give the same index -> m mapping.
    maps = []

    for conv_path in conv_files:
        conv_rows = parse_shifted_convergence(conv_path)
        current = {
            int(r["shift_index"]): (float(r["m_real"]), float(r["m_imag"]))
            for r in conv_rows
        }
        if current:
            maps.append((conv_path, current))

    if not maps:
        return {}

    reference_path, reference = maps[0]

    for other_path, other in maps[1:]:
        if other != reference:
            print(
                f"[WARNING] Several shifted_convergence.dat files with "
                f"different index mappings below {shifted_dir}; "
                f"using {reference_path}",
                file=sys.stderr,
            )
            break

    return reference


def decode_independent_directory(
    directory: Path,
    shift_index_map: dict[int, tuple[float, float]],
) -> Optional[tuple[float, float, Optional[int]]]:
    """
    Decode an independent-run directory name.

    Supported forms
    ---------------
    1. Explicit refractive index:
         1.300_0.100

    2. Indexed run:
         run000_sphere_g128_m1.3

       For form 2, Re(m) is also present in the directory name.  Im(m) is
       preferably obtained from shifted_convergence.dat using run_index as
       shift_index.  If no such mapping exists, use the known Figure-2 grid
       Im(m) = 0.100 + 0.025 * (run_index mod 21).
    """
    match = M_DIR_RE.fullmatch(directory.name)

    if match:
        return (
            to_float(match.group("real")),
            to_float(match.group("imag")),
            None,
        )

    match = RUN_DIR_RE.fullmatch(directory.name)

    if not match:
        return None

    run_index = int(match.group("run_index"))
    real_from_name = to_float(match.group("real"))

    if run_index in shift_index_map:
        m_real, m_imag = shift_index_map[run_index]

        # The directory already contains Re(m), so use it as an independent
        # consistency check on the run-index mapping.
        if not abs(real_from_name - m_real) <= 5e-10:
            print(
                f"[WARNING] Re(m) mismatch for {directory}: directory name "
                f"gives {real_from_name}, shifted_convergence.dat gives "
                f"{m_real} for shift_index={run_index}",
                file=sys.stderr,
            )

        return m_real, m_imag, run_index

    # Fallback for an independent dataset with no sibling shifted file.
    imag_index = run_index % FIG1_N_IMAG
    m_imag = FIG1_IMAG_MIN + FIG1_IMAG_STEP * imag_index

    print(
        f"[WARNING] No shifted index mapping for {directory}; "
        f"inferring Im(m)={m_imag:.3f} from run_index={run_index} "
        f"using the Figure-1 grid.",
        file=sys.stderr,
    )

    return real_from_name, m_imag, run_index


# ============================================================
# Independent runs
# ============================================================


def collect_independent(root: Path) -> list[dict]:
    rows = []

    independent_dirs = sorted(
        p for p in root.rglob("independent") if p.is_dir()
    )

    for independent_dir in independent_dirs:
        all_dir = independent_dir / "all"

        if not all_dir.is_dir():
            print(
                f"[WARNING] Independent directory has no 'all/' child: "
                f"{independent_dir}",
                file=sys.stderr,
            )
            continue

        meta = infer_metadata(independent_dir)

        # Needed for the new runXXX_sphere_g128_m... naming convention.
        # For the old m_real_m_imag naming convention this map is harmless.
        shift_index_map = build_shift_index_map(independent_dir)

        run_dirs = []

        for p in all_dir.iterdir():
            if not p.is_dir():
                continue

            decoded = decode_independent_directory(
                p,
                shift_index_map=shift_index_map,
            )

            if decoded is None:
                continue

            m_real, m_imag, run_index = decoded

            run_dirs.append((m_real, m_imag, run_index, p))

        run_dirs.sort(
            key=lambda x: (
                x[0],
                x[1],
                x[2] if x[2] is not None else -1,
            )
        )

        if not run_dirs:
            print(
                f"[WARNING] No recognized independent-run directories "
                f"found in {all_dir}. Supported names are e.g. "
                f"'1.300_0.100' and 'run000_sphere_g128_m1.3'.",
                file=sys.stderr,
            )
            continue

        for m_real, m_imag, run_index, run_dir in run_dirs:
            description = f"independent m={m_real:.3f}+{m_imag:.3f}i"
            if run_index is not None:
                description += f" (run{run_index:03d})"

            log_path = find_one_log(
                run_dir,
                description=description,
            )

            if log_path is None:
                continue

            log = parse_log(log_path)

            # Independent run: total iterations are the iterations for this m.
            iterations = log["run_total_iterations"]

            if (
                iterations is not None
                and log["residual_iteration"] is not None
                and iterations != log["residual_iteration"]
            ):
                print(
                    f"[WARNING] Iteration mismatch in {log_path}: "
                    f"Total number of iterations={iterations}, "
                    f"last RE index={log['residual_iteration']}",
                    file=sys.stderr,
                )

            row = {
                **meta,
                "method": "independent",
                "timing_scope": "per_index_run",
                "evaluation": None,
                "alpha": None,
                "beta": None,
                "gamma": None,
                "polarization": None,
                "shift_index": run_index,
                "m_real": m_real,
                "m_imag": m_imag,
                "m_label": f"{m_real:.3f}_{m_imag:.3f}",
                "iterations": iterations,
                "run_total_iterations": log["run_total_iterations"],
                "residual_iteration": log["residual_iteration"],
                "final_residual": log["final_residual"],
                "total_wall_time": log["total_wall_time"],
                "initialization_time": log["initialization_time"],
                "internal_fields": log["internal_fields"],
                "matvec_products": log["matvec_products"],
                "vector_readback": log["vector_readback"],
                "scattered_fields": log["scattered_fields"],
                "other_sc_quantities": log["other_sc_quantities"],
                "log_file": str(log_path),
                "convergence_file": None,
            }

            rows.append(row)

    return rows


# ============================================================
# Shifted runs
# ============================================================


def collect_shifted(root: Path) -> list[dict]:
    rows = []

    convergence_files = sorted(
        p for p in root.rglob("shifted_convergence.dat") if p.is_file()
    )

    for conv_path in convergence_files:
        run_dir = conv_path.parent
        meta = infer_metadata(conv_path)

        log_path = run_dir / "log"

        if not log_path.is_file():
            log_path = find_one_log(
                run_dir,
                description=f"shifted run {run_dir}",
            )

        if log_path is None or not log_path.is_file():
            continue

        log = parse_log(log_path)
        convergence_rows = parse_shifted_convergence(conv_path)

        if not convergence_rows:
            print(
                f"[WARNING] No convergence rows found in {conv_path}",
                file=sys.stderr,
            )
            continue

        for conv in convergence_rows:
            m_real = conv["m_real"]
            m_imag = conv["m_imag"]

            row = {
                **meta,
                "method": "shifted",
                # IMPORTANT:
                # these timings belong to the single 441-shift run,
                # so they are repeated for all 441 rows.
                "timing_scope": "shared_shifted_run",
                "evaluation": conv["evaluation"],
                "alpha": conv["alpha"],
                "beta": conv["beta"],
                "gamma": conv["gamma"],
                "polarization": conv["polarization"],
                "shift_index": conv["shift_index"],
                "m_real": m_real,
                "m_imag": m_imag,
                "m_label": f"{m_real:.3f}_{m_imag:.3f}",
                # Per-shift convergence information in shifted_convergence.dat
                "iterations": conv["iterations"],
                "run_total_iterations": log["run_total_iterations"],
                "residual_iteration": conv["iterations"],
                "final_residual": conv["final_residual"],
                # Global timing information from the shifted log
                "total_wall_time": log["total_wall_time"],
                "initialization_time": log["initialization_time"],
                "internal_fields": log["internal_fields"],
                "matvec_products": log["matvec_products"],
                "vector_readback": log["vector_readback"],
                "scattered_fields": log["scattered_fields"],
                "other_sc_quantities": log["other_sc_quantities"],
                "log_file": str(log_path),
                "convergence_file": str(conv_path),
            }

            rows.append(row)

    return rows


# ============================================================
# Validation / output
# ============================================================

FIELDNAMES = [
    "family",
    "backend",
    "nproc",
    "method",
    "timing_scope",
    "evaluation",
    "alpha",
    "beta",
    "gamma",
    "polarization",
    "shift_index",
    "m_real",
    "m_imag",
    "m_label",
    "iterations",
    "run_total_iterations",
    "residual_iteration",
    "final_residual",
    "total_wall_time",
    "initialization_time",
    "internal_fields",
    "matvec_products",
    "vector_readback",
    "scattered_fields",
    "other_sc_quantities",
    "log_file",
    "convergence_file",
]


def validate_rows(rows: list[dict]) -> None:
    """Print useful validation warnings."""
    groups: dict[tuple, list[dict]] = {}

    for row in rows:
        key = (
            row["family"],
            row["backend"],
            row["nproc"],
            row["method"],
            row["evaluation"],
            row["polarization"],
        )
        groups.setdefault(key, []).append(row)

    print("\nDetected datasets:")

    for key, group in sorted(
        groups.items(),
        key=lambda item: (
            str(item[0][0]),
            str(item[0][1]),
            item[0][2] if item[0][2] is not None else -1,
            str(item[0][3]),
            item[0][4] if item[0][4] is not None else -1,
            str(item[0][5]),
        ),
    ):
        family, backend, nproc, method, evaluation, polarization = key

        label = f"{family}/{backend}"
        if nproc is not None:
            label += f"/Nproc={nproc}"
        label += f"/{method}"

        if evaluation is not None:
            label += f"/eval={evaluation}"
        if polarization is not None:
            label += f"/pol={polarization}"

        unique_m = {
            (round(r["m_real"], 12), round(r["m_imag"], 12)) for r in group
        }

        missing_residuals = sum(r["final_residual"] is None for r in group)

        print(
            f"  {label:45s}: "
            f"{len(group):4d} rows, "
            f"{len(unique_m):4d} unique m, "
            f"{missing_residuals} missing residuals"
        )

        if len(unique_m) != 441:
            print(
                f"    [WARNING] Expected 441 unique refractive indices, "
                f"found {len(unique_m)}",
                file=sys.stderr,
            )


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:

    root = Path(__file__).resolve().parents[1] / "datas" / "Figure1"
    filename = "figure1_grid2d.csv"
    output = root / filename

    if not root.is_dir():
        print(f"Root directory does not exist: {root}", file=sys.stderr)

    independent_rows = collect_independent(root)
    shifted_rows = collect_shifted(root)

    rows = independent_rows + shifted_rows

    method_order = {
        "independent": 0,
        "shifted": 1,
    }

    family_order = {
        "SEQ": 0,
        "GPU": 1,
        "MPI": 2,
        "UNKNOWN": 99,
    }

    backend_order = {
        "SEQ": 0,
        "OCL_BLAS": 1,
        "OCL": 2,
        "GPU": 3,
        "MPI": 4,
        "UNKNOWN": 99,
    }

    rows.sort(
        key=lambda r: (
            family_order.get(r["family"], 99),
            backend_order.get(r["backend"], 99),
            r["nproc"] if r["nproc"] is not None else 0,
            method_order.get(r["method"], 99),
            r["evaluation"] if r["evaluation"] is not None else 0,
            r["polarization"] or "",
            r["m_real"],
            r["m_imag"],
        )
    )

    validate_rows(rows)
    write_csv(rows, output)

    print()
    print(f"Independent rows : {len(independent_rows)}")
    print(f"Shifted rows     : {len(shifted_rows)}")
    print(f"Total rows       : {len(rows)}")
    print(f"Output           : {output}")


if __name__ == "__main__":
    main()
