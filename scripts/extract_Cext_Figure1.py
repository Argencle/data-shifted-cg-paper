from pathlib import Path
import re
import pandas as pd

# ============================================================
# User options
# ============================================================

ROOT = Path(__file__).resolve().parents[1] / "datas" / "Figure1"

OUTPUT_FILE = ROOT / "figure1_Cext.csv"


# ============================================================
# Regular expressions
# ============================================================

# Generic floating-point number, including scientific notation
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


# Example:
# Cext = 1.234567890E+02
CEXT_RE = re.compile(
    rf"^\s*Cext\s*=\s*({FLOAT})",
    re.MULTILINE,
)


# Example shifted directory:
# m1.3_0.1
# m1.325_0.125
SHIFTED_M_RE = re.compile(
    r"^m" r"(?P<n>\d+(?:\.\d+)?)" r"_" r"(?P<k>\d+(?:\.\d+)?)" r"$"
)


# Example ADDA log:
# refractive index: 1.3+0.1i
REFRACTIVE_INDEX_RE = re.compile(
    rf"refractive\s+index\s*:\s*"
    rf"(?P<n>{FLOAT})"
    rf"\s*(?P<sign>[+-])\s*"
    rf"(?P<k>{FLOAT})"
    rf"\s*i",
    re.IGNORECASE,
)


# ============================================================
# Helpers
# ============================================================


def to_float(value):
    """Convert Fortran D exponent or standard E exponent to float."""
    return float(value.replace("D", "E").replace("d", "e"))


def read_cext(path):
    """Extract Cext from a CrossSec-Y file."""

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    match = CEXT_RE.search(text)

    if match is None:
        raise ValueError(f"Cext not found in {path}")

    return to_float(match.group(1))


def get_method(path):
    """Determine whether the result is shifted or independent."""

    parts = [part.lower() for part in path.parts]

    if "shifted" in parts:
        return "shifted"

    return "independent"


def get_shifted_index(path):
    """
    Recover n and k from a shifted directory such as:
        m1.3_0.1/CrossSec-Y
    """

    for parent in path.parents:

        match = SHIFTED_M_RE.match(parent.name)

        if match is not None:

            n = float(match.group("n"))
            k = float(match.group("k"))

            return n, k

    return None


def get_independent_index(path):
    """
    Recover n and k from the ADDA log located in the same
    run directory as CrossSec-Y.
    """

    log_file = path.parent / "log"

    if not log_file.exists():

        raise FileNotFoundError(f"No log file found for {path}")

    text = log_file.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    match = REFRACTIVE_INDEX_RE.search(text)

    if match is None:

        raise ValueError(f"Refractive index not found in {log_file}")

    n = to_float(match.group("n"))
    k = to_float(match.group("k"))

    if match.group("sign") == "-":
        k = -k

    return n, k


# ============================================================
# Find all CrossSec-Y files
# ============================================================

crosssec_files = sorted(ROOT.rglob("CrossSec-Y"))

print(f"Found {len(crosssec_files)} " f"CrossSec-Y files")


if len(crosssec_files) == 0:

    raise RuntimeError(f"No CrossSec-Y files found under {ROOT}")


# ============================================================
# Extract data
# ============================================================

rows = []

for crosssec in crosssec_files:

    method = get_method(crosssec)

    try:

        cext = read_cext(crosssec)

        if method == "shifted":

            result = get_shifted_index(crosssec)

            if result is None:

                raise ValueError(
                    "Cannot determine shifted "
                    f"refractive index from {crosssec}"
                )

            n, k = result

        else:

            n, k = get_independent_index(crosssec)

        rows.append(
            {
                "method": method,
                "m_real": n,
                "m_imag": k,
                "Cext": cext,
                "path": str(crosssec.relative_to(ROOT)),
            }
        )

    except Exception as exc:

        print()
        print(f"WARNING: {crosssec}")
        print(f"    {exc}")


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame(rows)

df["m_real"] = df["m_real"].round(6)
df["m_imag"] = df["m_imag"].round(6)

df = df.sort_values(
    [
        "method",
        "m_real",
        "m_imag",
    ]
).reset_index(drop=True)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("Extracted results")
print("=" * 70)

print(df.groupby("method").size())

print()

print(df.head(20).to_string(index=False))


# ============================================================
# Check expected 21 x 21 grid
# ============================================================

for method in [
    "independent",
    "shifted",
]:

    subset = df[df["method"] == method]

    if len(subset) > 0:

        print()
        print(f"{method}: {len(subset)} results")

        print(f"  unique Re(m): " f"{subset['m_real'].nunique()}")

        print(f"  unique Im(m): " f"{subset['m_imag'].nunique()}")


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.15e",
)

print()
print(f"Saved to: {OUTPUT_FILE}")
