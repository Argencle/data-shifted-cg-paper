import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11

# ============================================================
# User options
# ============================================================

GRID = 256
FIT_EXAMPLE = "OCL"  # "SEQ", "OCL", "OCL_BLAS", "MPI4", "MPI16", "MPI128"

INPUT_FILE = f"fixed_grid=256/figure1_timings.csv"
OUTPUT_FILE = f"Figure2_grid{GRID}.pdf"

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

seq = df[df["backend"] == "SEQ"].sort_values("Nm")
ocl = df[df["backend"] == "OCL"].sort_values("Nm")
ocl_blas = df[df["backend"] == "OCL_BLAS"].sort_values("Nm")
mpi4 = df[(df["family"] == "MPI") & (df["nproc"] == 4)].sort_values("Nm")
mpi16 = df[(df["family"] == "MPI") & (df["nproc"] == 16)].sort_values("Nm")
mpi128 = df[(df["family"] == "MPI") & (df["nproc"] == 128)].sort_values("Nm")

# ============================================================
# Fit model
# T(Nm) = T1 + tau * (Nm - 1)
#
# T1 is fixed to the measured value at Nm = 1.
# Only tau is fitted.
# ============================================================

def fit_time(data):
    Nm = data["Nm"].to_numpy()
    T = data["internal_fields"].to_numpy()

    T1 = data.loc[data["Nm"] == 1, "internal_fields"].iloc[0]

    x = Nm - 1

    tau = np.sum(x * (T - T1)) / np.sum(x**2)

    T_fit = T1 + tau * x

    ss_res = np.sum((T - T_fit) ** 2)
    ss_tot = np.sum((T - np.mean(T)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return T1, tau, r2


def speedup_data(data):
    T1 = data.loc[data["Nm"] == 1, "internal_fields"].iloc[0]
    return data["Nm"] * T1 / data["internal_fields"]


def speedup_model(Nm, T1, tau):
    return Nm * T1 / (T1 + tau * (Nm - 1))


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
}

labels = {
    "SEQ": "1 CPU",
    "OCL": "OCL",
    "OCL_BLAS": "OCL_BLAS",
    "MPI4": "4 CPUs",
    "MPI16": "16 CPUs",
    "MPI128": "128 CPUs",
}

fits = {}

print(f"\nGrid = {GRID}")
print("Fit model: T(Nm) = T1 + tau * (Nm - 1)")
print("T1 fixed to measured T(Nm=1)")
print("=" * 75)

for key, data in datasets.items():
    if data.empty:
        continue

    T1, tau, r2 = fit_time(data)
    S_sat = T1 / tau

    fits[key] = {
        "T1": T1,
        "tau": tau,
        "r2": r2,
        "S_sat": S_sat,
    }

    print(
        f"{labels[key]:8s}: "
        f"T1 = {T1:10.4f} s   "
        f"tau = {tau:10.6f} s/index   "
        f"R² = {r2:.6f}   "
        f"S_sat = {S_sat:10.4f}"
    )

# ============================================================
# Figure: solver speedup
# ============================================================

fig, ax = plt.subplots(figsize=(6.8, 5.0))

line_ocl = ax.plot(
    ocl["Nm"],
    speedup_data(ocl),
    "^",
    markersize=4,
    label="OCL",
)[0]

line_ocl_blas = ax.plot(
    ocl_blas["Nm"],
    speedup_data(ocl_blas),
    ":s",
    markersize=0,
    label="OCL_BLAS",
)[0]

line_seq = ax.plot(
    seq["Nm"],
    speedup_data(seq),
    "-o",
    markersize=0,
    label="1 CPU",
)[0]

line_mpi4 = ax.plot(
    mpi4["Nm"],
    speedup_data(mpi4),
    "--o",
    markersize=0,
    label="4 CPUs",
)[0]

line_mpi16 = ax.plot(
    mpi16["Nm"],
    speedup_data(mpi16),
    "-.o",
    markersize=0,
    label="16 CPUs",
)[0]

line_mpi128 = ax.plot(
    mpi128["Nm"],
    speedup_data(mpi128),
    linestyle=(0, (5, 10)),
    marker="o",
    markersize=0,
    label="128 CPUs",
)[0]

# Ideal speedup
Nm_ideal = np.logspace(
    np.log10(df["Nm"].min()),
    np.log10(df["Nm"].max()),
    300,
)

ax.plot(
    Nm_ideal,
    Nm_ideal,
    "k-",
    linewidth=1.4,
    label="Ideal",
)

# ============================================================
# Add one fit example
# ============================================================

if FIT_EXAMPLE not in fits:
    raise ValueError(
        "FIT_EXAMPLE must be one of: "
        "'SEQ', 'OCL', 'OCL_BLAS', 'MPI4', 'MPI16', 'MPI128'"
    )

data_fit = datasets[FIT_EXAMPLE]
T1_fit = fits[FIT_EXAMPLE]["T1"]
tau_fit = fits[FIT_EXAMPLE]["tau"]

Nm_fit = np.logspace(
    np.log10(data_fit["Nm"].min()),
    np.log10(data_fit["Nm"].max()),
    300,
)

ax.plot(
    Nm_fit,
    speedup_model(Nm_fit, T1_fit, tau_fit),
    color="black",
    linestyle="--",
    linewidth=1.0,
    alpha=0.8,
    label=rf"Fit: $S=\frac{{N_m T_1}}{{T_1+\tau(N_m-1)}}$",
)

# ============================================================
# Axes and layout
# ============================================================

ax.tick_params(axis="both", which="both", top=True, right=True)

ax.set_xlabel("Number of refractive indices")
ax.set_ylabel("Solver speedup")

ax.set_xscale("log")
ax.set_yscale("log")

ax.legend(frameon=False, fontsize=9)

plt.tight_layout()

fig.savefig(
    OUTPUT_FILE,
    bbox_inches="tight",
    dpi=300,
)

plt.show()
