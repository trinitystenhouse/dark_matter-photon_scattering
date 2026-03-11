# Usage Guide

This guide provides detailed examples for common analysis tasks.

## Quick Start

### Complete Analysis Pipeline

Run the full analysis from data download to figure generation:

```bash
# 1. Setup environment
cd /path/to/DM_Photon_Scattering
source Totani_paper_check/setup.sh

# 2. Download and process data
cd Totani_paper_check/data_download
bash download_fermi_totani.sh
bash process_totani.sh

# 3. Validate data products
cd ../sanity_checks
python sanity_checks.py

# 4. Generate templates
cd ../make_templates
bash run_all_templates.sh  # Or run individually (see below)

# 5. Run MCMC fits
cd ../mcmc
python run_mcmc_from_config.py configs/base.json

# 6. Generate figures
cd ../figures
python make_figures_from_config.py configs/fig2_3.json
```

## Template Generation

### Individual Template Building

Build templates one at a time with custom parameters:

```bash
cd make_templates

# Extended source mask (REQUIRED FIRST)
python mask_extended_sources.py \
  --radius-deg 3.0 \
  --output ${REPO_PATH}/fermi_data/totani/processed/srcmask.fits

# Point sources
python build_ps_template.py \
  --catalog 4FGL-DR3 \
  --min-significance 5.0

# Diffuse backgrounds
python build_gas_template.py --model gll_iem_v07
python build_ics_template.py --model gll_iem_v07
python build_iso_template.py --spectrum isotropic_iem_v07.txt

# Dark matter candidate (NFW)
python build_nfw_template.py \
  --rho-power 2.5 \
  --gamma 1.2 \
  --rs-kpc 20.0 \
  --norm pole \
  --expo-sampling center

# Loop I superbubble
python build_loopI_template.py \
  --center-l 330 \
  --center-b 15 \
  --radius-deg 60 \
  --spectral-index 2.7

# Fermi bubbles
python build_bubbles_templates.py \
  --vertices-north bubble_vertices_north.txt \
  --vertices-south bubble_vertices_south.txt
```

### Batch Template Generation

Use the provided script to build all templates:

```bash
cd make_templates
bash run_all_templates.sh
```

## MCMC Fitting

### Single Energy Bin

Fit a single energy bin for testing:

```bash
cd mcmc

python run_mcmc.py \
  --energy-bin 5 \
  --nwalkers 64 \
  --nsteps 10000 \
  --burn 2000 \
  --outdir mcmc_results \
  --run-name test_k5
```

### Multiple Energy Bins

Use configuration files for batch processing:

```bash
# Edit config file
cat > configs/my_analysis.json << EOF
{
  "energy_bins": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "nwalkers": 64,
  "nsteps": 10000,
  "burn": 2000,
  "outdir": "mcmc_results/my_analysis",
  "components": ["gas", "ics", "ps", "iso", "nfw", "loopI", "fb_pos", "fb_neg"]
}
EOF

# Run batch analysis
python run_mcmc_from_config.py configs/my_analysis.json
```

### Advanced MCMC Options

Control convergence, output, and initialization:

```bash
python run_mcmc.py \
  --energy-bin 5 \
  --nwalkers 128 \
  --nsteps 20000 \
  --burn 5000 \
  --thin-save 10 \
  --save-chain \
  --convergence-check \
  --gelman-rubin-threshold 1.1 \
  --autocorr-threshold 50 \
  --iso-target-e2 None \
  --outdir mcmc_results/high_precision
```

## Template Validation

### Visual Inspection

Check individual templates:

```bash
cd check_templates

# NFW profile
python check_nfw.py --energy-bin 5 --plot-spatial --plot-spectrum

# Loop I
python check_loopI.py --show-geometry --energy-bin 5

# Fermi bubbles
python check_bubbles_posneg.py --overlay-vertices
```

### Systematics Suite

Run comprehensive validation:

```bash
cd check_templates

python run_systematics_suite.py \
  --vary-nfw-slope \
  --vary-loopI-radius \
  --vary-bubble-vertices \
  --output systematics_outputs/
```

## Figure Reproduction

### Individual Figures

Generate specific figures from Totani (2025):

```bash
cd figures

# Figure 1: Sky maps at multiple energies
python make_totani_fig1_maps.py \
  --energy-bins 2 5 8 \
  --projection mollweide \
  --output plots_fig1/

# Figure 8: Component spectra
python fig8/make_totani_fig8.py \
  --mcmc-dir ../mcmc/mcmc_results \
  --show-uncertainties \
  --output plots_fig8/

# Figure 12: Posterior distributions
python fig12/make_fig12.py \
  --mcmc-file ../mcmc/mcmc_results/mcmc_k5.npz \
  --corner-plot \
  --output plots_fig12/
```

### Batch Figure Generation

Use configuration files:

```bash
cd figures

python make_figures_from_config.py configs/all_figures.json
```

## Data Analysis

### Extract MCMC Coefficients

Get fitted coefficients for downstream analysis:

```bash
cd mcmc

python get_norms.py \
  --mcmc-dir mcmc_results \
  --output coefficients.csv \
  --format csv
```

### Compute Flux Scaling

Convert coefficients to physical fluxes:

```bash
python get_norms.py \
  --mcmc-dir mcmc_results \
  --scale-flux \
  --expo ${REPO_PATH}/fermi_data/totani/processed/expcube_*.fits \
  --binsz 0.125 \
  --output flux_table.csv
```

## Troubleshooting

### MCMC Not Converging

If MCMC chains don't converge:

1. Increase walkers: `--nwalkers 128`
2. Increase burn-in: `--burn 5000`
3. Check initialization: Add `--verbose` flag
4. Inspect chains: `--save-chain` and plot with `corner.py`

### Template Validation Failures

If templates don't match expectations:

1. Check input data: Run `sanity_checks.py`
2. Verify normalization: See `docs/TEMPLATE_NORMALIZATION.md`
3. Compare with checks: Run `check_templates/check_*.py`
4. Review parameters: Check command-line arguments

### Memory Issues

For large MCMC runs:

1. Reduce walkers: `--nwalkers 32`
2. Increase thinning: `--thin-save 20`
3. Don't save chains: Remove `--save-chain`
4. Process bins sequentially instead of parallel

## Advanced Topics

### Custom ROI Definition

Define a custom region of interest:

```python
from totani_helpers.totani_io import get_totani_roi

# Custom ROI: |b| < 20°, |l| < 30°
roi_mask = get_totani_roi(
    wcs=my_wcs,
    shape=(ny, nx),
    b_range=(-20, 20),
    l_range=(-30, 30)
)
```

### Custom Template Components

Add a new template component:

1. Create template FITS file with proper WCS
2. Add to `load_all_templates()` in `totani_io.py`
3. Update MCMC configuration to include new component
4. Add prior bounds in `mcmc_helper.py`

### Parallel Processing

Run multiple energy bins in parallel:

```bash
# Using GNU parallel
parallel -j 4 python run_mcmc.py --energy-bin {} --outdir mcmc_results ::: {0..9}

# Or use the batch script
bash run_mcmc_batch.sh configs/parallel.json
```

## Best Practices

1. **Always run sanity checks** before template generation
2. **Validate templates** with check scripts before MCMC
3. **Start with short MCMC runs** to test convergence
4. **Save configurations** for reproducibility
5. **Document parameter choices** in analysis notebooks
6. **Version control** your configuration files
7. **Archive raw data** separately from processed products

## Getting Help

- Check `docs/` for detailed documentation
- Review example configurations in `configs/`
- Examine validation scripts in `check_templates/`
- See `DEVELOPMENT_NOTES.md` for known issues
