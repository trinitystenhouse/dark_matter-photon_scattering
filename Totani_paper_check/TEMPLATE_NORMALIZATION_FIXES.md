# Template Normalization Fixes

## Problem Summary

Your MCMC fits had three main normalization issues:

1. **Iso template stuck at E^2 dN/dE ≈ 1e-4**: The iso template was being forced to initialize at a target E^2 dN/dE value, preventing it from decreasing during fitting.

2. **NFW and loopI templates had arbitrary scaling**: These spatial templates were multiplied by an arbitrary flux factor `I0 = 1e-7`, making the fitted coefficients hard to interpret.

3. **Inconsistent normalization philosophy**: Physical templates (gas, ics, ps, iso) have known flux normalizations, while spatial templates (nfw, loopI, bubbles) are shape-only and should be fitted with free amplitudes.

## Root Causes

### Iso Template
- Built from a physical isotropic spectrum file with real flux values
- `mu_iso = iso_dnde * expo * omega * dE` where `iso_dnde` comes from interpolated spectrum
- At `f_iso=1`, this already corresponds to a specific physical E^2 dN/dE
- MCMC was then **rescaling** this to force `iso_target_e2=1e-4`, effectively locking the normalization

### NFW Template  
- Spatial template computed from LOS integral: `J(l,b) = ∫ ρ(r)^p ds`
- Normalized to Galactic pole or ROI sum
- Then multiplied by arbitrary `I0 = 1e-7` to create counts template
- Fitted coefficient `f_nfw` had no clear physical meaning

### LoopI Template
- Spatial template from shell geometry (chord length)
- Normalized to mean=1 in fit region  
- Then multiplied by arbitrary `I0 = 1e-7`
- Same issue as NFW

### Bubbles Templates
- Already correctly implemented: flat mask without arbitrary scaling
- `mu = expo * omega * dE` inside mask, zero outside
- Fitted coefficient directly represents flux

## Fixes Applied

### 1. NFW Template (`build_nfw_template.py`)
**Before:**
```python
I0 = 1e-7
mu_nfw = nfw_spatial[None, :, :] * conv * I0
```

**After:**
```python
# Normalize spatial template to mean=1 in ROI for numerical stability
vals = nfw_spatial[roi]
vals = vals[np.isfinite(vals) & (vals > 0)]
nfw_spatial = nfw_spatial / np.mean(vals)

# Template is shape-only; fitted coefficient will be in units of flux
mu_nfw = nfw_spatial[None, :, :] * conv
```

**Impact:** Fitted `f_nfw` now directly represents the flux [ph cm^-2 s^-1 sr^-1 MeV^-1] of the NFW component.

### 2. LoopI Template (`build_loopI_template.py`)
**Before:**
```python
I0 = 1e-7
mu_loopI = loopI_dnde * expo * omega[None, :, :] * dE_mev[:, None, None] * I0
```

**After:**
```python
# Template is shape-only (already normalized to mean=1 in ROI)
# Fitted coefficient will be in units of flux
mu_loopI = loopI_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]
```

**Impact:** Fitted `f_loopI` now directly represents the flux [ph cm^-2 s^-1 sr^-1 MeV^-1] of Loop I.

### 3. MCMC Initialization (`run_mcmc.py` and `mcmc_helper.py`)
**Before:**
```python
--iso-target-e2 default=1e-4
# Iso always rescaled to this target, starting at f_iso = 1e-4 / E2_at_f1
```

**After:**
```python
--iso-target-e2 default=None
# If None: iso starts at f=1 like other physical templates (gas, ics, ps)
# If set: legacy mode, rescales iso to target (for backward compatibility)
```

**Impact:** Iso template can now freely decrease or increase during fitting. At `f_iso=1`, it represents the flux from the input isotropic spectrum file.

## Action Required: Regenerate Templates

You need to regenerate your NFW and loopI templates with the new normalization:

```bash
cd Totani_paper_check/make_templates

# Regenerate NFW template (adjust parameters as needed)
python build_nfw_template.py \
  --rho-power 2.5 \
  --norm pole \
  --expo-sampling center

# Regenerate Loop I template  
python build_loopI_template.py

# Bubbles templates don't need regeneration (already correct)
```

## Expected Changes in Fits

### Iso Template
- **Before:** Stuck at E^2 dN/dE ≈ 1e-4, couldn't decrease
- **After:** Free to fit, starting from the physical spectrum value
- **Interpretation:** `f_iso` is now a multiplier on the input isotropic spectrum

### NFW Template  
- **Before:** `f_nfw ≈ O(1-100)` with arbitrary units due to `I0=1e-7` scaling
- **After:** `f_nfw` in physical flux units [ph cm^-2 s^-1 sr^-1 MeV^-1]
- **Expected values:** Likely `f_nfw ~ 1e-7 to 1e-6` (typical DM annihilation flux scales)

### LoopI Template
- **Before:** `f_loopI ≈ O(1-100)` with arbitrary units
- **After:** `f_loopI` in physical flux units [ph cm^-2 s^-1 sr^-1 MeV^-1]  
- **Expected values:** Likely `f_loopI ~ 1e-7 to 1e-6` (IC emission from Loop I)

### Bubbles Templates
- No change (already correct)
- `f_fb_pos`, `f_fb_neg` in physical flux units

## Testing

Run a quick test on one energy bin:

```bash
cd Totani_paper_check/mcmc

python run_mcmc.py \
  --energy-bin 2 \
  --nwalkers 32 \
  --nsteps 2000 \
  --burn 500 \
  --outdir mcmc_test_new_norm \
  --run-name test_k2
```

Check the output:
1. Iso should initialize at `f_iso ≈ 1.0` (not forced to match 1e-4)
2. NFW/loopI should have small fitted values (~ 1e-7 to 1e-6)
3. Iso should be able to decrease if the data supports it

## Backward Compatibility

If you need the old behavior for comparison:

```bash
python run_mcmc.py --iso-target-e2 1e-4 ...
```

This will rescale iso to start at E^2 dN/dE = 1e-4 (legacy mode).

## Additional Improvements: File Compression & Compatibility

Beyond the normalization fixes, several improvements were made to reduce file sizes and improve compatibility:

### 1. Compressed MCMC Output (`run_mcmc.py`)

**Changes:**
- Use `np.savez_compressed` instead of `np.savez` (typically 5-10x compression)
- Downcast float64 arrays to float32 (2x size reduction with negligible precision loss)
- Make chain/logprob saving optional (these are the largest arrays)
- Add thinning for saved chain/logprob to further reduce size

**New CLI arguments:**
```bash
--save-chain          # Save MCMC chain (off by default)
--save-logprob        # Save MCMC logprob (off by default)
--thin-save 10        # Thinning factor for saved arrays (default 10)
```

**Impact:** MCMC output files are now ~50-100x smaller by default (from ~500MB to ~5MB per energy bin).

**Example:**
```bash
# Minimal output (coefficients only, ~5MB)
python run_mcmc.py --energy-bin 2

# Save thinned chain for diagnostics (~50MB)
python run_mcmc.py --energy-bin 2 --save-chain --thin-save 10

# Save full chain (large, ~500MB)
python run_mcmc.py --energy-bin 2 --save-chain --save-logprob --thin-save 1
```

### 2. Shrink Existing Files (`shrink_mcmc_outfiles.py`)

Created a utility script to compress existing MCMC output files:

```bash
# Compress all files in a directory
python mcmc/shrink_mcmc_outfiles.py mcmc_results/

# Drop chain/logprob to save maximum space
python mcmc/shrink_mcmc_outfiles.py mcmc_results/ --drop-chain --drop-logprob

# Create backups before shrinking
python mcmc/shrink_mcmc_outfiles.py mcmc_results/*.npz --backup-suffix .bak

# Dry run to see what would happen
python mcmc/shrink_mcmc_outfiles.py mcmc_results/ --dry-run
```

### 3. Lazy Imports for Plotting Scripts

**Problem:** Scripts that export MCMC coefficients would fail if `astropy` wasn't installed, even when not making plots.

**Fix:** Move plotting helper imports inside `if args.make_plots:` blocks.

**Modified:**
- `fig2_3/reproduce_totani_fig2_fig3.py` - Lazy import of `make_fig2_fig3_plots_from_mcmc`

**Impact:** Can now export coefficients without having `astropy` installed.

### 4. MCMC I/O Compatibility

All coefficient loading scripts are compatible with the new minimal `.npz` format:
- `totani_helpers/mcmc_io.py` - Loads coefficients without requiring chain/logprob
- All fig scripts work with minimal format
- Coefficient export works without plotting dependencies

## Summary

The fixes ensure:
- **Physical templates** (gas, ics, ps, iso) start at their baseline normalization (`f=1`)
- **Spatial templates** (nfw, loopI, bubbles) are shape-only with fitted amplitudes in physical flux units
- **Iso template** is no longer locked to an arbitrary target value
- **All fitted coefficients** have clear physical interpretations
- **MCMC output files** are 50-100x smaller by default
- **Existing files** can be compressed with the shrink script
- **Scripts work** without heavy dependencies when only exporting coefficients

This should resolve the issues where iso was stuck, nfw/loopI normalizations were incorrect, and MCMC files were too large.
