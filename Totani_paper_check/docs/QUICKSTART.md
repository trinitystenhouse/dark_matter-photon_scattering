# Quick Start Guide

Get started with the Totani (2025) validation analysis in minutes.

## Prerequisites

- Conda or Miniconda installed
- ~50 GB free disk space for Fermi data
- Python ≥3.10

## Installation (5 minutes)

```bash
# Navigate to repository root
cd /path/to/DM_Photon_Scattering

# Run setup script
source Totani_paper_check/setup.sh
```

This creates the `fermi` conda environment and installs all dependencies.

## Complete Analysis (2-3 hours)

### 1. Download Data (30 minutes)

```bash
cd Totani_paper_check/data_download
bash download_fermi_totani.sh
```

Downloads ~10 GB of Fermi-LAT photon data.

### 2. Process Data (20 minutes)

```bash
bash process_totani.sh
```

Creates counts cubes, exposure maps, and livetime cubes.

### 3. Validate Data (2 minutes)

```bash
cd ../sanity_checks
python sanity_checks.py
```

Check `sanity_plots/` for diagnostic plots.

### 4. Build Templates (15 minutes)

```bash
cd ../make_templates

# Required first step
python mask_extended_sources.py

# Build all templates
python build_ps_template.py
python build_gas_template.py
python build_ics_template.py
python build_iso_template.py
python build_nfw_template.py --rho-power 2.5 --norm pole
python build_loopI_template.py
python build_bubbles_templates.py
```

### 5. Run MCMC Fit (1-2 hours)

```bash
cd ../mcmc

# Single energy bin test (5 minutes)
python run_mcmc.py --energy-bin 5 --nwalkers 32 --nsteps 5000 --burn 1000

# Full analysis (1-2 hours)
python run_mcmc_from_config.py configs/base.json
```

### 6. Generate Figures (5 minutes)

```bash
cd ../figures

# Sky maps
python make_totani_fig1_maps.py

# Energy spectra
python make_figures_from_config.py configs/fig2_3.json
```

## Quick Test (10 minutes)

Test the pipeline on a single energy bin:

```bash
cd Totani_paper_check

# Assume data is already downloaded and processed

# Build one template
cd make_templates
python mask_extended_sources.py
python build_nfw_template.py

# Quick MCMC test
cd ../mcmc
python run_mcmc.py --energy-bin 5 --nwalkers 16 --nsteps 1000 --burn 200

# Check results
python get_norms.py --mcmc-dir . --output test_coeffs.txt
```

## Common Commands

### Check MCMC Results

```bash
cd mcmc
python get_norms.py --mcmc-dir mcmc_results --output coefficients.csv
```

### Validate Templates

```bash
cd check_templates
python check_nfw.py --energy-bin 5
python check_loopI.py --energy-bin 5
```

### Rerun Single Energy Bin

```bash
cd mcmc
python run_mcmc.py --energy-bin 3 --nwalkers 64 --nsteps 10000 --burn 2000
```

## Troubleshooting

### "Command not found: conda"
Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

### "File not found: counts_ccube_*.fits"
Run data processing: `bash data_download/process_totani.sh`

### "MCMC not converging"
- Increase walkers: `--nwalkers 128`
- Increase burn-in: `--burn 5000`
- Check templates with validation scripts

### "Out of memory"
- Reduce walkers: `--nwalkers 32`
- Don't save chains: remove `--save-chain` flag
- Increase thinning: `--thin-save 20`

## Next Steps

- Read `README.md` for comprehensive documentation
- See `docs/USAGE_GUIDE.md` for detailed examples
- Check `CONTRIBUTING.md` for development guidelines
- Review `docs/TEMPLATE_NORMALIZATION.md` for methodology

## Getting Help

- Check documentation in `docs/`
- Review example configurations in `configs/`
- Run validation scripts in `check_templates/`
- See `CHANGELOG.md` for recent changes
