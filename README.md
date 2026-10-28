# Data Availability - Shifted CG paper

This repository contains the raw ADDA results, aggregated CSV datasets, and Python scripts used to generate the figures for the shifted CG paper.

The CSV datasets are provided, so you can reproduce the figures without extracting the raw data or compiling ADDA.

## Reproduce the figures from the provided CSV files

Run the commands below from the repository root.

### 1. Clone the repository

```bash
git clone https://github.com/Argencle/data-shifted-cg-paper.git
cd data-shifted-cg-paper
```

### 2. Set up Python

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Generate the figures

```bash
python scripts/plot_Figure1.py
python scripts/plot_Figure2.py
```

For the additional fixed-grid benchmark with grid size 256:

```bash
python scripts/plot_Figure2_grid256.py
```

The output files are `figures/Figure1.pdf`, `figures/Figure2.pdf`, and `figures/Figure2_grid256.pdf`, respectively. Plot settings, such as the backend shown in Figure 1, can be changed in the user options near the top of each script.

The plotting scripts print numerical diagnostics to standard output. With its default SEQ configuration, the Figure 1 script also generates `datas/Figure1/Cext_data_SEQ.csv`, containing the matched independent and shifted extinction cross sections, absolute and relative differences, agreement in matching digits, and iteration counts used in the figure.

## Rebuild the CSV datasets from the raw data (optional)

Use this workflow to regenerate the CSV files from the archived simulation results. It requires the Python environment described above, but does not require compiling or running ADDA.

### 1. Extract the raw data

```bash
bash scripts/unzip_data.sh
```

The script finds ZIP archives recursively under `datas/` and extracts each one beside its ZIP file. Each ZIP file is deleted only after a successful extraction.

### 2. Rebuild the CSV files

```bash
python scripts/extract_Cext_Figure1.py
python scripts/extract_grid2d_Figure1.py
python scripts/extract_timings_Figure2.py
python scripts/extract_timings_Figure2_grid256.py
```

| Extraction script | Output CSV |
| --- | --- |
| `scripts/extract_Cext_Figure1.py` | `datas/Figure1/figure1_Cext.csv` |
| `scripts/extract_grid2d_Figure1.py` | `datas/Figure1/figure1_grid2d.csv` |
| `scripts/extract_timings_Figure2.py` | `datas/Figure2/figure2_timings.csv` |
| `scripts/extract_timings_Figure2_grid256.py` | `datas/Figure2/figure2_timings_grid256.csv` |

The grid-256 extraction script reads raw results from `datas/Figure2/fixed_grid=256/`. After extraction, rerun the plotting commands above to regenerate the figures from the rebuilt CSV files.

## ADDA version used for the simulations

The simulations used [ADDA commit 35deda91d07de7572163220d7c539070409931a1](https://github.com/Argencle/adda/tree/35deda91d07de7572163220d7c539070409931a1). Building ADDA is only needed if you intend to run simulations yourself.

To obtain and compile that version, run the following commands from this repository's root in a shell with the required ADDA build tools installed:

```bash
git clone https://github.com/Argencle/adda.git
cd adda
git checkout 35deda91d07de7572163220d7c539070409931a1
cd src
make all
cd ../..
```

Alternatively, download the [ZIP snapshot of the same commit](https://github.com/Argencle/adda/archive/35deda91d07de7572163220d7c539070409931a1.zip), extract it, and compile it.

## Repository structure

| Directory | Contents |
| --- | --- |
| `datas/Figure1/` | Raw-data archives and CSV datasets for Figure 1 |
| `datas/Figure2/` | Raw-data archives and CSV datasets for Figure 2 and the grid-256 benchmark |
| `scripts/` | Data extraction, plotting, and archive extraction scripts |
| `figures/` | Generated PDF figures |

## License

- Source code: [MIT License](LICENSE).
- Data and figures: [Creative Commons Attribution 4.0 (CC-BY-4.0)](LICENSE_DATA).

## Citation

If you use this dataset, please cite both the archived dataset version on Zenodo and the associated paper.
