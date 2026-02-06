# Fermionic DM EFT operator

This folder contains an EFT-based DM–photon scattering attenuation model and a script to compare against the Fermi-LAT spectrum.

## Setup (recommended)

From the **repo root**:

```bash
source setup.sh
```

This:

- exports `REPO_PATH` (repo root)
- installs the top-level `helpers` package in editable mode

The script defaults to reading the York spectrum file from:

- `${REPO_PATH}/fermi_data/york/processed/spectrum_data.txt`

## Run the analysis

From this folder:

```bash
python Fermi-LAT_analysis_eff_coupling_fermionic.py
```

Or from anywhere:

```bash
python "${REPO_PATH}/Fermionic_DM_Eff_Operator/Fermi-LAT_analysis_eff_coupling_fermionic.py"
```

## Key CLI flags

- `--filename`
  - Input spectrum file (default: `${REPO_PATH}/fermi_data/york/processed/spectrum_data.txt`)
- `--outdir`
  - Output directory for plots (default: `plots` relative to current working directory)
- `--is-cosmic`
  - Use cosmic mean DM density + long baseline (default uses GC-like values)
- EFT params (single point):
  - `--Lambda`
  - `--c_s`
  - `--c_p`
  - `--mchi`
- Visible-dip search mode:
  - `--find-visible`
  - `--dip-energy`, `--dip-depth`
  - `--eft-kinematic-factor`
  - `--find-visible-samples`, `--find-visible-seed`
  - `--log10-mchi-min/max`, `--log10-cs-min/max`, `--log10-cp-min/max`

## Outputs

The script writes a PNG to `--outdir` with filename containing the EFT parameters, e.g.

- `spectrum_with_attenuation_Lambda_..._cs_..._cp_..._mchi_....png`
