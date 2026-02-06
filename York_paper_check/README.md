# York Paper Check

This folder contains:

- A weak cross-section notebook for theoretical checks/plots.
- Fermi-LAT data download + processing utilities to reproduce the spectrum used in York paper checks.
- The Figure 6 analysis scripts (fits the spectrum with/without DM scattering).

## Prerequisites

- Conda (Miniconda or Anaconda)
- Internet connection for data download
- ~5 GB free disk space

## Repo path setup (recommended)

From the **repo root**:

```bash
source setup.sh
```

This exports `REPO_PATH` (repo root) and installs the top-level `helpers` package in editable mode.

This also sets up the `scattering` conda environment, which is required for the weak cross-section notebook.

## Weak cross-section notebook

The weak cross-section notebook contains theoretical calculations and visualizations of weak cross-sections for DM–photon scattering, focusing on:

- χγ → χγ

It generates plots in a `plots/` directory, e.g.:

- `Weakly_dep_on_angle.png`
- `Weakly_dep_on_egamma.png`
- `Weakly_egamma_vs_angle.png`
- `Weakly_full_no_log.png`

To run it:

```bash
jupyter notebook weak_cross_section.ipynb
```

## Theoretical background (notebook)

The notebook includes:

- W boson loop contributions
- Fermion loop contributions
- Combined (full) cross-section calculations

Calculations are performed in both center-of-mass and lab frames, with support for different dark matter masses (default: 1 TeV).

## Fermi tools environment

Activate your Fermitools environment:

```bash
conda activate fermi
```

If you don’t have it yet:

```bash
conda create -n fermi -c conda-forge -c fermi python=3.9 fermitools
```

## Recreate Figure 6 (end-to-end)

### 1) Download data

From `York_paper_check/data_processing`:

```bash
bash download_fermi.sh
```

This downloads:

- 1 spacecraft file (~2.4 GB)
- 7 photon files (~2.5 MB total)

### 2) Process data

```bash
python batch_process_fermi.py
```

This applies the required cuts:

- Energy: 50–500 GeV
- ROI: 10° around Galactic Center
- Zenith: < 90°
- Event class: SOURCE (128)
- Good time intervals (DATA_QUAL>0, LAT_CONFIG==1)

Output:

- `${REPO_PATH}/fermi_data/york/processed/GC_filtered_merged.fits`

### 3) Extract the photon spectrum

```bash
python extract_spectrum.py
```

Output:

- `${REPO_PATH}/fermi_data/york/processed/spectrum_data.txt`

### 4) Run the analysis (Figure 6)

You can run the scripts from anywhere (as long as you sourced `setup.sh` so `REPO_PATH` is set).

Example:

```bash
python "${REPO_PATH}/York_paper_check/fig6/Fermi-LAT_analysis_coupling.py" \
  --is-cosmic \
  --y_eff 1e4
```

Notes:

- `--is-cosmic` uses cosmic distances instead of the GC distance.
- `--mchi` changes the dark matter mass.
- `--filename` overrides the input spectrum file. By default it uses:
  - `${REPO_PATH}/fermi_data/york/processed/spectrum_data.txt`

Estimated time:

- Download: 10–30 min (depends on connection)
- Processing: 5–15 min (depends on CPU)
- Total: ~30–45 min

