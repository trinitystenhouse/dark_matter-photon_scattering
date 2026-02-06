# Scalar DM EFT operator

This folder contains an EFT-based scalar DM–photon scattering attenuation model and a script/notebook used for comparisons and cross-section sanity checks.

## Setup (recommended)

From the **repo root**:

```bash
source setup.sh
```

This:

- exports `REPO_PATH` (repo root)
- installs the top-level `helpers` package in editable mode

The analysis script defaults to reading:

- `${REPO_PATH}/fermi_data/york/processed/spectrum_data.txt`

## Run the analysis script

From this folder:

```bash
python Fermi-LAT_analysis_eff_coupling_scalar.py
```

Or from anywhere:

```bash
python "${REPO_PATH}/Scalar_DM_Eff_Operator/Fermi-LAT_analysis_eff_coupling_scalar.py"
```

## Key CLI flags

- `--filename`
  - Input spectrum file (default: `${REPO_PATH}/fermi_data/york/processed/spectrum_data.txt`)
- `--outdir`
  - Output directory for plots (default: `plots` relative to current working directory)
- `--is-cosmic`
  - Use cosmic mean DM density + long baseline (default uses GC-like values)
- EFT params:
  - `--Lambda`
  - `--c_phi`
  - `--mchi`

## Notebooks

- `scalar_DM_cross_section.ipynb`
  - Cross-section sanity checks and plots.
  - Saves plots into a `plots/` directory (relative to where you run the notebook).
