from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11


# ============================================================
# User options
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "datas" / "Figure1"

# File containing Cext for independent and shifted calculations.
# If it is not found directly, DATA_DIR and the script directory are tried.
CEXT_FILE = DATA_DIR / "figure1_Cext.csv"

# Files containing per-index iteration counts. All matching standard files
# are concatenated; files containing "eps16" are ignored.
TIMING_PATTERN = "figure1_grid2d.csv"

# Available configurations: "SEQ", "MPI", "OCL", "OCL_BLAS", "eps16"
BACKEND = "SEQ"
NPROC = 16  # only used for BACKEND == "MPI"

# Solver residual tolerance
ETA = 1e-5

# ----------------------------
# Panel (a): refractive-index map
# ----------------------------
VMIN = 5
VMAX = 15
CMAP_MAP = "viridis"

SHOW_CONTOURS = True
CONTOUR_LEVELS = [6, 8, 10]
CONTOUR_LABELS = True

# ----------------------------
# Panel (b): agreement vs iterations
# ----------------------------
# Which iteration count to use on the x axis:
#   "independent" : BiCG_CS iterations
#   "shifted"     : per-shift SBiCG_CS iterations
#   "max"         : max(independent, shifted)
ITERATION_SOURCE = "independent"

YMIN = 4.8
YMAX = 15.0

# Point coloring in panel (b):
#   "abs_delta_k"    : |K_shift - K_ind|  (recommended)
#   "signed_delta_k" :  K_shift - K_ind
#   "m_real"         : Re(m)
#   "m_imag"         : Im(m)
#   None              : no color encoding
COLOR_BY = None

# For |Delta K| coloring, values >= DELTA_K_CLIP get the same top-end color.
DELTA_K_CLIP = 4

SHOW_BINNED_MEDIAN = False
N_BINS = 10

# Figure output
OUTPUT_FILE = None


# ============================================================
# Helpers
# ============================================================


def get_backend_from_path(path):
    # CSV paths may have been generated on Windows or POSIX.
    path = str(path).replace("\\", "/")

    if path.startswith("2Dgrid_SEQ/"):
        return "SEQ", None

    if path.startswith("2Dgrid_GPU/OCL_BLAS/"):
        return "OCL_BLAS", None

    if path.startswith("2Dgrid_GPU/OCL/"):
        return "OCL", None

    if path.startswith("eps16/"):
        return "eps16", None

    match = re.search(r"2Dgrid_MPI/Nproc=(\d+)/", path)
    if match:
        return "MPI", int(match.group(1))

    return "UNKNOWN", None


def select_configuration(df, backend, nproc=None):
    if backend == "MPI":
        out = df[(df["backend"] == "MPI") & (df["nproc"] == nproc)].copy()
        label = f"MPI {nproc}"
        tag = f"MPI{nproc}"
    else:
        out = df[df["backend"] == backend].copy()
        label = backend
        tag = backend

    if len(out) == 0:
        raise RuntimeError(f"No data found for configuration {label}")

    return out, label, tag


def resolve_file(path):
    """Try the supplied path, DATA_DIR, then the script directory."""
    candidates = [
        path,
        DATA_DIR / path.name,
        Path(__file__).resolve().parent / path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find {path.name}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


# ============================================================
# Load Cext data and compute agreement
# ============================================================

CEXT_FILE = resolve_file(CEXT_FILE)
cext = pd.read_csv(CEXT_FILE)

backend_info = cext["path"].apply(get_backend_from_path)
cext["backend"] = [x[0] for x in backend_info]
cext["nproc"] = [x[1] for x in backend_info]

cext, config_label, config_tag = select_configuration(cext, BACKEND, NPROC)

ind = cext[cext["method"] == "independent"][
    ["m_real", "m_imag", "Cext"]
].copy()
shf = cext[cext["method"] == "shifted"][["m_real", "m_imag", "Cext"]].copy()

if len(ind) == 0:
    raise RuntimeError(f"No independent data found for {config_label}")
if len(shf) == 0:
    raise RuntimeError(f"No shifted data found for {config_label}")

ind = ind.rename(columns={"Cext": "Cext_independent"})
shf = shf.rename(columns={"Cext": "Cext_shifted"})

for d in (ind, shf):
    d["m_real"] = d["m_real"].round(3)
    d["m_imag"] = d["m_imag"].round(3)

comparison = pd.merge(ind, shf, on=["m_real", "m_imag"], how="inner")

if len(comparison) == 0:
    raise RuntimeError(
        "No matching refractive indices between independent and shifted data."
    )

comparison["absolute_difference"] = np.abs(
    comparison["Cext_shifted"] - comparison["Cext_independent"]
)
comparison["relative_difference"] = comparison["absolute_difference"] / np.abs(
    comparison["Cext_independent"]
)

positive = comparison.loc[
    comparison["relative_difference"] > 0, "relative_difference"
]
if len(positive) == 0:
    raise RuntimeError("All Cext differences are exactly zero.")

smallest_positive = positive.min()
n_exact_zero = (comparison["relative_difference"] == 0).sum()

# Exact zero cannot be converted to -log10. For visualization only,
# replace it by the smallest positive measured difference.
comparison["relative_difference_plot"] = comparison[
    "relative_difference"
].replace(0, smallest_positive)

comparison["agreement_orders"] = -np.log10(
    comparison["relative_difference_plot"]
)


# ============================================================
# Load iteration counts
# ============================================================

timing_files = [
    p for p in DATA_DIR.glob(TIMING_PATTERN) if "eps16" not in p.name.lower()
]

# Fallback: useful when the script and CSV files are kept together.
if not timing_files:
    local_dir = Path(__file__).resolve().parent
    timing_files = [
        p
        for p in local_dir.glob(TIMING_PATTERN)
        if "eps16" not in p.name.lower()
    ]

if not timing_files:
    raise FileNotFoundError(
        f"Cannot find any standard timing file matching {TIMING_PATTERN}."
    )

timing = pd.concat(
    [pd.read_csv(path) for path in timing_files],
    ignore_index=True,
)

# Multiple extraction files can overlap. Keep one row per configuration,
# method, and refractive index.
timing = timing.drop_duplicates(
    subset=["backend", "nproc", "method", "m_real", "m_imag"],
    keep="last",
)

timing, _, _ = select_configuration(timing, BACKEND, NPROC)

iter_ind = timing[timing["method"] == "independent"][
    ["m_real", "m_imag", "iterations"]
].copy()
iter_shf = timing[timing["method"] == "shifted"][
    ["m_real", "m_imag", "iterations"]
].copy()

iter_ind = iter_ind.rename(columns={"iterations": "K_ind"})
iter_shf = iter_shf.rename(columns={"iterations": "K_shift"})

for d in (iter_ind, iter_shf):
    d["m_real"] = d["m_real"].round(3)
    d["m_imag"] = d["m_imag"].round(3)

iterations = pd.merge(iter_ind, iter_shf, on=["m_real", "m_imag"], how="inner")
iterations["K_max"] = iterations[["K_ind", "K_shift"]].max(axis=1)
iterations["delta_K"] = iterations["K_shift"] - iterations["K_ind"]
iterations["abs_delta_K"] = np.abs(iterations["delta_K"])

plot_data = pd.merge(
    comparison,
    iterations,
    on=["m_real", "m_imag"],
    how="inner",
)

if ITERATION_SOURCE == "independent":
    xcol = "K_ind"
    xlabel = r"BiCG$_{\mathrm{CS}}$ iterations"
elif ITERATION_SOURCE == "shifted":
    xcol = "K_shift"
    xlabel = r"SBiCG$_{\mathrm{CS}}$ iterations, $K_{\mathrm{shift}}$"
elif ITERATION_SOURCE == "max":
    xcol = "K_max"
    xlabel = r"$\max(K_{\mathrm{ind}},K_{\mathrm{shift}})$"
else:
    raise ValueError(
        "ITERATION_SOURCE must be 'independent', 'shifted', or 'max'."
    )


# ============================================================
# Diagnostics
# ============================================================

mean_error = comparison["relative_difference"].mean()
median_error = comparison["relative_difference"].median()
max_error = comparison["relative_difference"].max()
eta_orders = -np.log10(ETA)

pearson_K = plot_data["agreement_orders"].corr(
    plot_data[xcol], method="pearson"
)
spearman_K = plot_data["agreement_orders"].corr(
    plot_data[xcol], method="spearman"
)
pearson_abs_dK = plot_data["agreement_orders"].corr(
    plot_data["abs_delta_K"], method="pearson"
)
spearman_abs_dK = plot_data["agreement_orders"].corr(
    plot_data["abs_delta_K"], method="spearman"
)

print("=" * 72)
print("Cext numerical equivalence")
print("=" * 72)
print(f"Configuration                    : {config_label}")
print(f"Matched Cext points              : {len(comparison)}")
print(f"Matched points with iterations   : {len(plot_data)}")
print(f"Exact-zero Cext differences      : {n_exact_zero}")
print(f"Mean relative difference         : {mean_error:.16e}")
print(f"Median relative difference       : {median_error:.16e}")
print(f"Maximum relative difference      : {max_error:.16e}")
print(f"Smallest positive difference     : {smallest_positive:.16e}")
print(
    f"Iteration range                  : {plot_data[xcol].min():.0f} -- {plot_data[xcol].max():.0f}"
)
print(
    f"Delta K range                    : {plot_data['delta_K'].min():.0f} -- {plot_data['delta_K'].max():.0f}"
)
print(f"Pearson agreement vs iterations  : {pearson_K:.4f}")
print(f"Spearman agreement vs iterations : {spearman_K:.4f}")
print(f"Pearson agreement vs |Delta K|   : {pearson_abs_dK:.4f}")
print(f"Spearman agreement vs |Delta K|  : {spearman_abs_dK:.4f}")


# ============================================================
# Panel (a) data
# ============================================================

grid = comparison.pivot(
    index="m_imag",
    columns="m_real",
    values="agreement_orders",
)
grid = grid.sort_index().sort_index(axis=1)

m_real = grid.columns.to_numpy()
m_imag = grid.index.to_numpy()
Z = grid.to_numpy()

vmin = np.nanmin(Z) if VMIN is None else VMIN
vmax = np.nanmax(Z) if VMAX is None else VMAX

if vmax <= vmin:
    raise ValueError("VMAX must be larger than VMIN.")


# ============================================================
# Combined figure
# ============================================================

fig, (ax_a, ax_b) = plt.subplots(
    1,
    2,
    figsize=(12.2, 5.0),
    constrained_layout=True,
)

# ----------------------------
# (a) Refractive-index map
# ----------------------------

im = ax_a.pcolormesh(
    m_real,
    m_imag,
    Z,
    shading="auto",
    cmap=CMAP_MAP,
    vmin=vmin,
    vmax=vmax,
    rasterized=True,
)

ax_a.set_aspect("equal", adjustable="box")

if SHOW_CONTOURS:
    levels = [level for level in CONTOUR_LEVELS if vmin < level < vmax]
    if levels:
        contours = ax_a.contour(
            m_real,
            m_imag,
            Z,
            levels=levels,
            colors="white",
            linewidths=0.55,
            alpha=0.65,
        )
        if CONTOUR_LABELS:
            ax_a.clabel(
                contours,
                inline=True,
                fontsize=8,
                fmt="%.0f",
            )

cbar_a = fig.colorbar(im, ax=ax_a, pad=0.02)
# cbar_a.set_label(r"$-\log_{10}(\eta_{C_{\mathrm{ext}}})$")
cbar_a.set_label("Matching digits", labelpad=10)
cbar_a.set_ticks(np.arange(np.ceil(vmin), np.floor(vmax) + 0.1, 2.0))

ax_a.set_xlabel(r"$\mathrm{Re}(m)$")
ax_a.set_ylabel(r"$\mathrm{Im}(m)$")
ax_a.text(
    0.02,
    0.98,
    "(a)",
    transform=ax_a.transAxes,
    ha="left",
    va="top",
    fontweight="bold",
)

# ----------------------------
# (b) Agreement vs iterations
# ----------------------------

if COLOR_BY is None:
    sc = ax_b.scatter(
        plot_data[xcol],
        plot_data["agreement_orders"],
        s=26,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )

elif COLOR_BY == "abs_delta_k":
    color_values = plot_data["abs_delta_K"].clip(upper=DELTA_K_CLIP)
    norm = Normalize(vmin=0, vmax=DELTA_K_CLIP)

    sc = ax_b.scatter(
        plot_data[xcol],
        plot_data["agreement_orders"],
        c=color_values,
        s=28,
        alpha=0.82,
        linewidths=0,
        cmap="viridis",
        norm=norm,
        rasterized=True,
    )

    cbar_b = fig.colorbar(sc, ax=ax_b, pad=0.02)
    cbar_b.set_label(r"$|\Delta K|=|K_{\mathrm{shift}}-K_{\mathrm{ind}}|$")
    ticks = np.arange(0, DELTA_K_CLIP + 1)
    cbar_b.set_ticks(ticks)
    ticklabels = [str(x) for x in ticks]
    ticklabels[-1] = rf"$\geq {DELTA_K_CLIP}$"
    cbar_b.set_ticklabels(ticklabels)

elif COLOR_BY == "signed_delta_k":
    color_values = plot_data["delta_K"]
    limit = max(
        abs(plot_data["delta_K"].min()),
        abs(plot_data["delta_K"].max()),
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    sc = ax_b.scatter(
        plot_data[xcol],
        plot_data["agreement_orders"],
        c=color_values,
        s=28,
        alpha=0.82,
        linewidths=0,
        cmap="coolwarm",
        norm=norm,
        rasterized=True,
    )

    cbar_b = fig.colorbar(sc, ax=ax_b, pad=0.02)
    cbar_b.set_label(r"$\Delta K=K_{\mathrm{shift}}-K_{\mathrm{ind}}$")

elif COLOR_BY in ("m_real", "m_imag"):
    sc = ax_b.scatter(
        plot_data[xcol],
        plot_data["agreement_orders"],
        c=plot_data[COLOR_BY],
        s=28,
        alpha=0.82,
        linewidths=0,
        cmap="viridis",
        rasterized=True,
    )

    cbar_b = fig.colorbar(sc, ax=ax_b, pad=0.02)
    if COLOR_BY == "m_real":
        cbar_b.set_label(r"$\mathrm{Re}(m)$")
    else:
        cbar_b.set_label(r"$\mathrm{Im}(m)$")

else:
    raise ValueError(
        "COLOR_BY must be None, 'abs_delta_k', 'signed_delta_k', "
        "'m_real', or 'm_imag'."
    )

if SHOW_BINNED_MEDIAN and len(plot_data) >= N_BINS:
    trend = plot_data[[xcol, "agreement_orders"]].copy()
    trend["bin"] = pd.qcut(
        trend[xcol],
        q=N_BINS,
        duplicates="drop",
    )
    trend = (
        trend.groupby("bin", observed=True)
        .agg(
            iterations=(xcol, "median"),
            agreement=("agreement_orders", "median"),
        )
        .reset_index(drop=True)
    )
    ax_b.plot(
        trend["iterations"],
        trend["agreement"],
        marker="o",
        markersize=4,
        linewidth=1.2,
        color="black",
        label="binned median",
    )

if YMIN <= eta_orders <= YMAX:
    ax_b.axhline(
        eta_orders,
        linestyle="--",
        linewidth=0.9,
        color="black",
        alpha=0.7,
        label=rf"$\eta=10^{{-5}}$",
    )

ax_b.set_xlabel(xlabel)
# ax_b.set_ylabel(r"$-\log_{10}(\eta_{C_{\mathrm{ext}}})$")
ax_b.set_ylim(YMIN, YMAX)

xrange = plot_data[xcol].max() - plot_data[xcol].min()
if xrange > 0:
    ax_b.set_xlim(
        plot_data[xcol].min() - 0.03 * xrange,
        plot_data[xcol].max() + 0.03 * xrange,
    )

ax_b.text(
    0.02,
    0.98,
    "(b)",
    transform=ax_b.transAxes,
    ha="left",
    va="top",
    fontweight="bold",
)

if SHOW_BINNED_MEDIAN or (YMIN <= eta_orders <= YMAX):
    ax_b.legend(frameon=False, loc="upper right")


# ============================================================
# Save data and figure
# ============================================================

comparison_file = (
    Path(__file__).resolve().parents[1]
    / "datas"
    / f"Figure1/Cext_data_{config_tag}.csv"
)
plot_data.to_csv(comparison_file, index=False, float_format="%.15e")

if OUTPUT_FILE is None:
    OUTPUT_FILE = (
        Path(__file__).resolve().parents[1] / "figures" / "Figure1.pdf"
    )

fig.savefig(
    OUTPUT_FILE,
    bbox_inches="tight",
    dpi=300,
)

print(f"Combined data saved to: {comparison_file}")
print(f"Figure saved to       : {OUTPUT_FILE}")

plt.show()
