# Scalar DM EFT operator

This folder contains a script that:

- computes a scalar EFT DM–photon scattering cross section
- fits a smooth power-law to the Fermi-LAT spectrum
- applies an attenuation factor `exp(-tau(E))` with `tau(E) = (rho_chi * L / mchi) * sigma(E)`
- writes plots of the spectrum and residuals

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

## What the script can do

- **Single-point attenuation plot (default)**
  - Uses `--Lambda`, `--mchi`, `--c_phi` to compute `sigma(E)` and produces a spectrum+residuals plot.
- **Find parameters that produce a visible dip**
  - With `--find-visible`, the script searches over `(mchi, c_phi)` to hit a target attenuation depth at `--dip-energy`.
  - If no point meets the target, it raises `RuntimeError` but prints the best attempt found.
- **Max-tau reach plots**
  - With `--tau-grid`, computes `tau_max(Lambda, mchi)` assuming `c_phi=1`.
  - By default, `tau_max` is defined as the maximum optical depth over an energy band (taken from the input data energy range unless overridden).
  - You can switch back to the legacy single-energy definition using `--tau-energy-mode dip` (evaluated at `--dip-energy`).
  - Saves:
    - `tau_vs_lambda_rayleigh_scalar_<baseline>.png`
    - `tau_grid_rayleigh_scalar_<baseline>.png`

## Key CLI flags

- `--filename`
  - Input spectrum file (default: `${REPO_PATH}/fermi_data/york/processed/spectrum_data.txt`)
- `--outdir`
  - Output directory for plots (default: `plots` relative to current working directory)
- Baseline / astrophysics:
  - `--baseline {gc,cosmic,custom}`
  - `--is-cosmic` (legacy switch; equivalent to `--baseline cosmic` when `--baseline` is not provided)
  - `--rho-chi`, `--L-cm` (used with `--baseline custom`)
- EFT params:
  - `--Lambda`
  - `--c_phi`
  - `--mchi`
 - Tau-grid energy definition:
  - `--tau-energy-mode {band,dip}`
  - `--tau-energy-min`, `--tau-energy-max`, `--tau-energy-n` (used when `--tau-energy-mode band`)

## Outputs

The script writes PNGs to `--outdir` (default `plots/`).

- The main spectrum plot (always produced in non-`--tau-grid` mode):
  - `spectrum_with_attenuation_Lambda_<...>_cphi_<...>_mchi_<...>.png`
- If `--tau-grid` is used:
  - `tau_vs_lambda_rayleigh_scalar_<baseline>.png`
  - `tau_grid_rayleigh_scalar_<baseline>.png`

## Notebooks

- `scalar_DM_cross_section.ipynb`
  - Cross-section sanity checks and plots.
  - Saves plots into a `plots/` directory (relative to where you run the notebook).

## How to interpret the `--tau-grid` plots

The `--tau-grid` workflow reframes the analysis as a *reach/constraint* study.

- **What is being plotted**
  - The script computes `tau_max(Lambda, mchi)` at a fixed dip energy (set by `--dip-energy`) and fixed maximal coupling (`c_phi=1`).
  - The 2D map shows `log10(tau_max)` over the `(log10 Lambda, log10 mchi)` grid.

- **Sensitivity boundary (solid white contour)**
  - This is the contour where `tau_max = tau_needed`, where `tau_needed` is set by your target dip depth (e.g. `--dip-depth`).
  - The side with **larger** `tau_max` (typically at **smaller** `Lambda`) is the region where the interaction is strong enough to produce an observable attenuation feature *under the assumptions of this dip model*.

- **EFT validity limit (red dashed contour)**
  - This contour marks where the EFT kinematic validity criterion transitions from valid to invalid.
  - On the EFT-invalid side, the code still reports `tau_max` (so you can see where a UV completion would need to live), but you should not interpret that region as a controlled EFT prediction.

- **CDM bound (white dotted horizontal line)**
  - This is an optional external prior (e.g. `mchi >= 1e-3 GeV`) drawn as a reference for a “cold dark matter” lower-mass bound.

- **The region that is both EFT-valid and testable**
  - The most defensible target region is the **overlap** of:
    - points on the `tau_max > tau_needed` side of the **sensitivity boundary**, and
    - points on the **EFT-valid** side of the EFT validity limit,
    - (optionally) points **above** the CDM bound line.