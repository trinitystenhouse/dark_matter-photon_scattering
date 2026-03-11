# Contributing to Totani Paper Check

Thank you for your interest in this Fermi-LAT analysis validation project. This document provides guidelines for understanding, using, and potentially extending the codebase.

## Project Structure

```
Totani_paper_check/
├── totani_helpers/          # Core analysis library
│   ├── totani_io.py         # Data I/O utilities
│   ├── fit_utils.py         # Fitting and likelihood functions
│   ├── mcmc_io.py           # MCMC result handling
│   ├── mcmc_plotting.py     # Spectral analysis utilities
│   ├── cellwise_fit.py      # Spatial cell fitting
│   ├── bubbles_templates.py # Fermi bubble geometry
│   └── plotting.py          # Plotting utilities
├── data_download/           # Data acquisition scripts
├── make_templates/          # Template construction scripts
├── mcmc/                    # MCMC fitting scripts
├── figures/                 # Figure generation scripts
├── check_templates/         # Template validation tools
├── sanity_checks/           # Data validation tools
└── docs/                    # Documentation

```

## Code Style

### Python Conventions

- **Python version**: ≥3.10
- **Line length**: 100 characters (configured in pyproject.toml)
- **Type hints**: Encouraged for public functions
- **Docstrings**: NumPy style for all public functions

### Example Function Documentation

```python
def load_template(path: str, energy_bin: int) -> np.ndarray:
    """
    Load emission template from FITS file.
    
    Parameters
    ----------
    path : str
        Path to template FITS file
    energy_bin : int
        Energy bin index (0-indexed)
    
    Returns
    -------
    template : np.ndarray
        Template array with shape (ny, nx)
    
    Raises
    ------
    FileNotFoundError
        If template file does not exist
    """
    # Implementation here
```

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
cd /path/to/DM_Photon_Scattering

# Setup environment
source Totani_paper_check/setup.sh

# Install development dependencies
pip install -e "Totani_paper_check[dev]"
```

### Making Changes

1. **Create a feature branch** (if using version control)
2. **Make your changes** following the code style guidelines
3. **Test your changes** with validation scripts
4. **Document your changes** in code and docs
5. **Update relevant examples** if adding new features

### Testing Changes

#### Template Changes

If you modify template construction:

```bash
cd make_templates
python build_<component>_template.py  # Rebuild template

cd ../check_templates
python check_<component>.py  # Validate template
```

#### Fitting Changes

If you modify fitting routines:

```bash
cd mcmc
python run_mcmc.py --energy-bin 2 --nsteps 1000  # Quick test

cd ../check_templates
python run_systematics_suite.py  # Full validation
```

#### Plotting Changes

If you modify plotting functions:

```bash
cd figures
python make_figures_from_config.py configs/test.json
```

## Adding New Features

### Adding a New Template Component

1. **Create template builder** in `make_templates/`:
   ```python
   # make_templates/build_mycomponent_template.py
   import numpy as np
   from astropy.io import fits
   
   def build_mycomponent_template(...):
       # Template construction logic
       pass
   ```

2. **Add to template loader** in `totani_helpers/totani_io.py`:
   ```python
   def load_all_templates(...):
       templates = {}
       # ... existing templates ...
       templates['mycomponent'] = load_template('mycomponent_template.fits')
       return templates
   ```

3. **Add MCMC prior** in `mcmc/mcmc_helper.py`:
   ```python
   def get_prior_bounds(...):
       bounds = {
           # ... existing bounds ...
           'mycomponent': (0.0, 10.0),  # (min, max)
       }
       return bounds
   ```

4. **Add plotting style** in `totani_helpers/plotting.py`:
   ```python
   def totani_component_style(label):
       styles = {
           # ... existing styles ...
           'mycomponent': {'color': 'purple', 'ls': '--', 'marker': 's'},
       }
       return styles.get(label, default_style)
   ```

5. **Create validation script** in `check_templates/`:
   ```bash
   cp check_templates/check_nfw.py check_templates/check_mycomponent.py
   # Edit to validate your component
   ```

### Adding a New Figure

1. **Create figure directory**:
   ```bash
   mkdir -p figures/figN
   ```

2. **Create plotting script**:
   ```python
   # figures/figN/make_figN.py
   from totani_helpers import plotting, mcmc_io
   
   def make_figN(...):
       # Figure generation logic
       pass
   ```

3. **Add configuration** in `figures/configs/`:
   ```json
   {
     "figure": "N",
     "mcmc_dir": "../mcmc/mcmc_results",
     "output_dir": "plots_figN/",
     "components": ["gas", "ics", "ps", "iso"]
   }
   ```

## Understanding the Analysis Pipeline

### Data Flow

```
Raw Fermi Data (FITS)
    ↓
[data_download/process_totani.sh]
    ↓
Counts/Exposure Cubes
    ↓
[make_templates/build_*.py]
    ↓
Emission Templates
    ↓
[mcmc/run_mcmc.py]
    ↓
MCMC Coefficients (.npz)
    ↓
[figures/make_*.py]
    ↓
Publication Figures
```

### Key Concepts

#### Template Normalization

- **Physical templates** (gas, ICS, PS, iso): Normalized to baseline predictions
- **Spatial templates** (NFW, Loop I, bubbles): Shape-only, fitted coefficients = flux

#### MCMC Fitting

- Uses `emcee` affine-invariant ensemble sampler
- Poisson likelihood for photon counting statistics
- Non-negative priors for most components
- Free-sign priors for NFW and fb_neg (can be negative)

#### Energy Binning

Energy bins are defined in log-space from 1 GeV to 1 TeV. Check `data_download/make_energybins.py` for bin definitions.

## Common Tasks

### Changing Energy Binning

1. Edit `data_download/make_energybins.py`
2. Reprocess data: `bash data_download/process_totani.sh`
3. Rebuild all templates
4. Rerun MCMC fits

### Adjusting ROI

1. Edit `totani_helpers/totani_io.py::get_totani_roi()`
2. Rebuild templates (they cache ROI information)
3. Rerun MCMC fits

### Modifying NFW Profile

1. Edit `make_templates/build_nfw_template.py`
2. Adjust parameters: `--rho-power`, `--gamma`, `--rs-kpc`
3. Rebuild: `python build_nfw_template.py --rho-power 2.0`
4. Validate: `python ../check_templates/check_nfw.py`

## Performance Optimization

### MCMC Performance

- **Parallel walkers**: Already parallelized via `emcee`
- **Energy bins**: Run in parallel using GNU parallel or job arrays
- **Thinning**: Use `--thin-save` to reduce output size
- **Early stopping**: Enable `--convergence-check` for automatic termination

### Template Construction

- Templates are cached as FITS files
- Only rebuild when parameters change
- Use `--expo-sampling center` for faster NFW builds

## Troubleshooting

### MCMC Convergence Issues

**Symptom**: Chains don't converge, Gelman-Rubin > 1.2

**Solutions**:
- Increase walkers: `--nwalkers 128`
- Increase burn-in: `--burn 5000`
- Check template normalization

### Template Validation Failures

**Symptom**: Check scripts show unexpected morphology

**Solutions**:
- Verify input data paths
- Check FITS header WCS information
- Compare with reference templates

### Memory Issues

**Symptom**: Out of memory during MCMC

**Solutions**:
- Reduce walkers: `--nwalkers 32`
- Don't save chains: remove `--save-chain`
- Increase thinning: `--thin-save 50`
- Process energy bins sequentially

## Questions and Support

For questions about:
- **Analysis methodology**: See `docs/USAGE_GUIDE.md`
- **Code issues**: Check validation scripts in `check_templates/`

## Acknowledgments

This analysis builds on:
- Totani (2025) methodology
- Fermi Science Tools
- `emcee` MCMC sampler
- Astropy ecosystem
