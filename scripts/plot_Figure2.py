from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11

# ============================================================
# User option
# ============================================================

FIT_EXAMPLE = "OCL"  # "SEQ", "OCL", "OCL_BLAS", "MPI4", "MPI16", "MPI128"
# Multi-node options: "MPI128_2nodes", "MPI128_4nodes"

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(
    Path(__file__).resolve().parents[1]
    / "datas"
    / "Figure2"
    / "figure2_timings.csv"
)

# Independent BiCG references are stored in the same CSV as the shifted runs.
bicg_references = df[(df["solver"] == "BiCG") & (df["Nm"] == 1)]
df = df[df["solver"] == "SBiCG"]

seq = df[df["backend"] == "SEQ"].sort_values("Nm")
ocl = df[df["backend"] == "OCL"].sort_values("Nm")
ocl_blas = df[df["backend"] == "OCL_BLAS"].sort_values("Nm")
mpi4 = df[(df["family"] == "MPI") & (df["nproc"] == 4)].sort_values("Nm")
mpi16 = df[(df["family"] == "MPI") & (df["nproc"] == 16)].sort_values("Nm")
mpi128_all = df[(df["family"] == "MPI") & (df["nproc"] == 128)]
mpi128 = mpi128_all[mpi128_all["nnodes"].isna()].sort_values("Nm")
mpi128_2nodes = mpi128_all[mpi128_all["nnodes"] == 2].sort_values("Nm")
mpi128_4nodes = mpi128_all[mpi128_all["nnodes"] == 4].sort_values("Nm")

# ============================================================
# Fit model
# T(Nm) = T1 + tau * (Nm - 1)
#
# T1 is fixed to the measured SBiCG timing at Nm = 1.
# BiCG supplies only the numerator of the speedup.
# Only tau is fitted.
# ============================================================


def fit_time(data):
    Nm = data["Nm"].to_numpy()
    T = data["internal_fields"].to_numpy()

    # Keep the measured SBiCG timing, including the Nm=1 point.
    T1 = data.loc[data["Nm"] == 1, "internal_fields"].iloc[0]

    x = Nm - 1

    # Least-squares fit of tau with T1 fixed
    tau = np.sum(x * (T - T1)) / np.sum(x**2)

    T_fit = T1 + tau * x

    # R²
    ss_res = np.sum((T - T_fit) ** 2)
    ss_tot = np.sum((T - np.mean(T)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return T1, tau, r2


def speedup_data(data, T1_bicg):
    return data["Nm"] * T1_bicg / data["internal_fields"]


def speedup_model(Nm, T1_bicg, T1_sbicg, tau):
    return Nm * T1_bicg / (T1_sbicg + tau * (Nm - 1))


# ============================================================
# Datasets and fits
# ============================================================

datasets = {
    "SEQ": seq,
    "OCL": ocl,
    "OCL_BLAS": ocl_blas,
    "MPI4": mpi4,
    "MPI16": mpi16,
    "MPI128": mpi128,
    "MPI128_2nodes": mpi128_2nodes,
    "MPI128_4nodes": mpi128_4nodes,
}

labels = {
    "SEQ": "1 CPU",
    "OCL": "OCL",
    "OCL_BLAS": "OCL_BLAS",
    "MPI4": "4 CPUs",
    "MPI16": "16 CPUs",
    "MPI128": "128 CPUs",
    "MPI128_2nodes": "128 CPUs (2 nodes)",
    "MPI128_4nodes": "128 CPUs (4 nodes)",
}

reference_comparisons = {}

print("\nNm=1 reference: SBiCG -> independent BiCG (internal_fields)")
print("Differences = SBiCG - BiCG; percentages = 100 * (SBiCG - BiCG) / BiCG.")
print(
    f"{'Backend':20s} {'SBiCG (s)':>11s} {'BiCG (s)':>11s} "
    f"{'Delta (s)':>11s} {'Delta (%)':>11s}"
)

for key, data in datasets.items():
    old_T1 = float(data.loc[data["Nm"] == 1, "internal_fields"].iloc[0])

    config = data.iloc[0]
    references = bicg_references[
        (bicg_references["family"] == config["family"])
        & (bicg_references["backend"] == config["backend"])
    ]
    if pd.isna(config["nproc"]):
        references = references[references["nproc"].isna()]
    else:
        references = references[references["nproc"] == config["nproc"]]

    if pd.isna(config["nnodes"]):
        references = references[references["nnodes"].isna()]
    else:
        node_references = references[references["nnodes"] == config["nnodes"]]
        if node_references.empty:
            references = references[references["nnodes"].isna()]
            print(
                f"  {labels[key]}: using MPI-{int(config['nproc'])} BiCG "
                "reference with 1 node."
            )
        else:
            references = node_references

    if len(references) != 1:
        raise ValueError(
            f"Expected one BiCG Nm=1 reference for {key}, found {len(references)}"
        )

    bicg_T1 = float(references.iloc[0]["internal_fields"])
    delta = old_T1 - bicg_T1
    delta_percent = 100 * delta / bicg_T1
    reference_comparisons[key] = {
        "SBiCG": old_T1,
        "BiCG": bicg_T1,
        "delta": delta,
        "delta_percent": delta_percent,
    }
    print(
        f"{labels[key]:20s} {old_T1:11.4f} {bicg_T1:11.4f} "
        f"{delta:+11.4f} {delta_percent:+11.2f}"
    )

print("For all Nm, the percentages also give 100 * "
      "(speedup_SBiCG - speedup_BiCG) / speedup_BiCG.")

fits = {}

print("\nFit model: T_SBiCG(Nm) = T1_SBiCG + tau * (Nm - 1)")
print("Speedup: Nm * T1_BiCG / T_SBiCG(Nm), including Nm=1.")
print("The timing fit (tau and R²) uses the original SBiCG measurements.")
print("=" * 75)

for key, data in datasets.items():
    T1, tau, r2 = fit_time(data)
    T1_bicg = reference_comparisons[key]["BiCG"]
    S_sat = T1_bicg / tau

    fits[key] = {
        "T1_SBiCG": T1,
        "T1_BiCG": T1_bicg,
        "tau": tau,
        "r2": r2,
        "S_sat": S_sat,
    }

    print(
        f"{labels[key]:8s}: "
        f"T1_SBiCG = {T1:10.4f} s   "
        f"T1_BiCG = {T1_bicg:10.4f} s   "
        f"tau = {tau:10.6f} s/index   "
        f"R² = {r2:.6f}   "
        f"S_sat = {S_sat:10.4f}"
    )
    old_S_sat = T1 / tau
    print(
        f"  S_sat (SBiCG -> BiCG normalization): {old_S_sat:.4f} -> {S_sat:.4f}; "
        f"100 * (S_sat_SBiCG - S_sat_BiCG) / S_sat_BiCG = "
        f"{100 * (old_S_sat / S_sat - 1):+.2f}%"
    )

# ============================================================
# Figure: internal-field speedup
# ============================================================

fig, ax = plt.subplots(figsize=(6.8, 5.0))

# Measured curves
line_ocl = ax.plot(
    ocl["Nm"],
    speedup_data(ocl, fits["OCL"]["T1_BiCG"]),
    "^",
    markersize=4,
    label="OCL",
)[0]

line_ocl_blas = ax.plot(
    ocl_blas["Nm"],
    speedup_data(ocl_blas, fits["OCL_BLAS"]["T1_BiCG"]),
    ":s",
    markersize=0,
    label="OCL_BLAS",
)[0]

line_seq = ax.plot(
    seq["Nm"],
    speedup_data(seq, fits["SEQ"]["T1_BiCG"]),
    "-o",
    markersize=0,
    label="1 CPU",
)[0]

line_mpi4 = ax.plot(
    mpi4["Nm"],
    speedup_data(mpi4, fits["MPI4"]["T1_BiCG"]),
    "--o",
    markersize=0,
    label="4 CPUs",
)[0]

line_mpi16 = ax.plot(
    mpi16["Nm"],
    speedup_data(mpi16, fits["MPI16"]["T1_BiCG"]),
    "-.o",
    markersize=0,
    label="16 CPUs",
)[0]

line_mpi128 = ax.plot(
    mpi128["Nm"],
    speedup_data(mpi128, fits["MPI128"]["T1_BiCG"]),
    linestyle=(0, (5, 10)),
    marker="o",
    markersize=0,
    label="128 CPUs",
)[0]

line_mpi128_4nodes = ax.plot(
    mpi128_4nodes["Nm"],
    speedup_data(mpi128_4nodes, fits["MPI128_4nodes"]["T1_BiCG"]),
    linestyle=(0, (1, 5)),
    markersize=0,
    label=labels["MPI128_4nodes"],
)[0]

# Ideal speedup
Nm_ideal = np.logspace(np.log10(df["Nm"].min()), np.log10(df["Nm"].max()), 300)

ax.plot(
    Nm_ideal,
    Nm_ideal,
    "k-",
    linewidth=1.4,
    label=r"Ideal",
)

# ============================================================
# Add one fit example
# ============================================================

if FIT_EXAMPLE not in datasets:
    raise ValueError(
        "FIT_EXAMPLE must be one of: "
        "'SEQ', 'OCL', 'OCL_BLAS', 'MPI4', 'MPI16', 'MPI128', "
        "'MPI128_2nodes', 'MPI128_4nodes'"
    )

data_fit = datasets[FIT_EXAMPLE]
T1_bicg_fit = fits[FIT_EXAMPLE]["T1_BiCG"]
T1_sbicg_fit = fits[FIT_EXAMPLE]["T1_SBiCG"]
tau_fit = fits[FIT_EXAMPLE]["tau"]
r2_fit = fits[FIT_EXAMPLE]["r2"]

Nm_fit = np.logspace(
    np.log10(data_fit["Nm"].min()),
    np.log10(data_fit["Nm"].max()),
    300,
)

# Use the same color as the selected dataset
line_map = {
    "SEQ": line_seq,
    "OCL": line_ocl,
    "OCL_BLAS": line_ocl_blas,
    "MPI4": line_mpi4,
    "MPI16": line_mpi16,
    "MPI128": line_mpi128,
    "MPI128_4nodes": line_mpi128_4nodes,
}

# fit_color = line_map[FIT_EXAMPLE].get_color()

ax.plot(
    Nm_fit,
    speedup_model(Nm_fit, T1_bicg_fit, T1_sbicg_fit, tau_fit),
    color="black",  # fit_color,
    linestyle="--",
    linewidth=1.0,
    alpha=0.8,
    label=(
        r"Fit: $S=\frac{N_m T_{1,\mathrm{BiCG}}}"
        r"{T_{1,\mathrm{SBiCG}}+\tau(N_m-1)}$"
    ),
)

# ============================================================
# Axes and layout
# ============================================================
ax.tick_params(axis="both", which="both", top=True, right=True)

ax.set_xlabel(r"Number of refractive indices")
ax.set_ylabel("Solver speedup")

ax.set_xscale("log")
ax.set_yscale("log")

ax.legend(frameon=False, fontsize=9)

plt.tight_layout()

fig.savefig(
    Path(__file__).resolve().parents[1] / "figures" / "Figure2.pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.show()
