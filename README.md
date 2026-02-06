# Paper_Check

Reproducibility/consistency checks and exploratory reimplementations of results from multiple papers (York, Totani, etc.), plus EFT-based DM–photon scattering toy models.

A core design goal of this repo is **portability**: most scripts can be run from any working directory by relying on a repo-root environment variable (`REPO_PATH`) that is exported when you `source setup.sh`.

## Repository structure (high-level)

- `helpers/`
  - Small helper package used across the repo (e.g. plot styling).
  - Installed as `paper-check-helpers` via `pip install -e .`.
- `York_paper_check/`
  - Fermi-LAT download + processing pipeline and scripts to reproduce/check York-related figures.
  - See `York_paper_check/README.md`.
- `Totani_paper_check/`
  - WIP Totani-style pipeline: templates + figure scripts.
  - See `Totani_paper_check/README.md`.
- `Scalar_DM_Eff_Operator/`
  - Scalar EFT scattering toy model script + notebook.
  - See `Scalar_DM_Eff_Operator/README.md`.
- `Fermionic_DM_Eff_Operator/`
  - Fermionic EFT scattering toy model script.
  - See `Fermionic_DM_Eff_Operator/README.md`.
- `alternative_mediator_scan/`
  - Scan framework over alternative mediator parameterizations.
  - See `alternative_mediator_scan/README.md`.
- `fermi_data/`
  - Local data products (gitignored).

## Environments

There are two main conda environments used in this repo.

### 1) `scattering` (general analysis + notebooks)

Create:

```bash
conda create -n scattering python=3.13
conda activate scattering
python -m pip install --upgrade pip
python -m pip install -r requirements_analysis.txt
```

This environment is used for:

- most analysis notebooks
- the scalar/fermionic EFT scripts
- the alternative mediator scan

### 2) `fermi` (Fermi Science Tools)

Create (recommended channels):

```bash
conda create -n fermi -c conda-forge -c fermi python=3.11 fermitools
conda activate fermi
python -m pip install --upgrade pip
python -m pip install -r requirements_analysis.txt
python -m pip install -r requirements_fermi.txt
```

Notes:

- In practice, `fermitools` is usually best installed via conda (as above). The `requirements_fermi.txt` is kept as a convenience for pip-side dependencies.
- Use this environment for the York/Totani Fermi-LAT processing and any scripts that call Fermitools executables.

## Setup (recommended)

From the repo root:

```bash
source setup.sh
```

This does two important things:

- exports `REPO_PATH` (repo root)
- installs the top-level `helpers` package in editable mode

To set up the Totani package and also export `REPO_PATH`:

```bash
source Totani_paper_check/setup.sh
```

## Path conventions (`REPO_PATH`)

Many scripts build paths like:

- `${REPO_PATH}/fermi_data/york/...`
- `${REPO_PATH}/fermi_data/totani/...`

so they don’t depend on your current working directory.

## Current functionality (what works today)

- **York pipeline**
  - Download/process/extract spectrum and run Figures 1, 2, 3, 4 and 6 analysis scripts.
  - See `York_paper_check/README.md`.
- **Alternative mediator scans**
  - Batch scans over (naively defined) mediators and masses with output organization.
  - See `alternative_mediator_scan/README.md`.
- **Scalar/Fermionic EFT scripts**
  - Constructs a model for scalar/fermionic dark matter using a 4-point model with an effective operator.
  - Fits a smooth spectrum + attenuation model with optional “find visible dip” scanning.
  - See `Scalar_DM_Eff_Operator/README.md` and `Fermionic_DM_Eff_Operator/README.md`.
- **Totani pipeline (WIP)**
  - Template builders + figure scripts for Fig 1, Fig 2/3 (in progress) and a naive first attempt at Fig 9.
  - Known differences vs paper are documented.
  - See `Totani_paper_check/README.md`.

## Packaging notes

When you run editable installs (`pip install -e .`), you may see `*.egg-info/` metadata directories created. These are normal and should be gitignored.
