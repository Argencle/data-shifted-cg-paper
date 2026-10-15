# Data Availability - Shifted CG paper

The exact version of ADDA used in this work is available at:
- https://github.com/Argencle/adda/tree/35deda91

```bash
git clone https://github.com/Argencle/adda.git
cd adda
git checkout 35deda91
cd src
make all
```

Optional (without `git clone` and `git checkout`): download the ZIP snapshot of that commit directly from
https://github.com/Argencle/adda/archive/35deda91d07de7572163220d7c539070409931a1.zip
extract it, and compile it.

## License

- Source code is licensed under the MIT License.
- Data and figures are licensed under CC-BY-4.0.

## Repository Purpose

- Provide the raw results (logs + CSV) used in the paper.
- Provide post-processing scripts to rebuild figure datasets.
- Provide plotting scripts to regenerate the final PDF figures.

## Repository Structure

- `datas/`
- In `datas`, raw data are stored as `.zip` archives; only aggregated files are kept uncompressed.
- `*.csv`: aggregated figure datasets.
- `scripts/`
- `extract_grid2d_Figure1.py`, `extract_grid2d_Figure1.py`, `extract_timings_Figure2.py`: build figure CSV datasets.
- `plot_Figure1.py`, `plot_Figure2.py`: generate PDF figures in `figures/`.
- `figures/`: output folder storing generated PDF figures.

## Python Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Build final figure datasets

Final figure datasets are already provided in `datas/` as CSV files.

Run:

```bash
python scripts/extract_Cext_Figure1.py
python scripts/extract_grid2d_Figure1.py
python scripts/extract_timings_Figure2.py
```

**Note**: 
- the `datas/*` folders must be unzipped in order to use these scripts.

## Figure Reproduction

Run:

```bash
python scripts/plot_figure1.py
python scripts/plot_figure2.py
```

Outputs are written to `figures/`.

## Citation

If you use this dataset, please cite both the archived dataset version on Zenodo and the associated paper.