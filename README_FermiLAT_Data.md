# Quick Start: Recreate Figure 6

## Prerequisites

- Conda (Miniconda or Anaconda)
- Internet connection for data download
- ~5 GB free disk space

## Step 1: Download Data

```bash
bash download_fermi.sh
```

This downloads:
- 1 spacecraft file (2.4 GB)
- 7 photon files (~2.5 MB total)

**Total download: ~2.4 GB**

## Step 2: Activate Fermitools

```bash
conda activate fermi
```

If not installed:
```bash
conda create -n fermi -c conda-forge -c fermi python=3.9 fermitools
```

## Step 3: Process Data

```bash
python batch_process_fermi.py
```

This applies the required cuts:
- ✓ Energy: 50-500 GeV
- ✓ ROI: 10° around Galactic Center
- ✓ Zenith: < 90°
- ✓ Event class: SOURCE (128)
- ✓ Good time intervals (DATA_QUAL>0, LAT_CONFIG==1)

**Output:** `fermi_data/processed/GC_filtered_merged.fits`

## Step 4: Analyse

Use the processed file to extract the photon spectrum.
```bash
python extract_spectrum.py
```

Then run the analysis script: 
Can set `--is-cosmic` to use cosmic distances instead of distances to galactic centre
Can set `--mchi` to change the dark matter mass
Can set `--filename` to change the input filename (automatically generated name from `extract_spectrum.py` is `spectrum_data.txt`)

```bash
python Fermi-LAT_analysis_coupling.py --is-cosmic --y_eff 1e4 --filename spectrum_data.txt
```

---

**Estimated time:**
- Download: 10-30 min (depends on connection)
- Processing: 5-15 min (depends on CPU)
- Total: ~30-45 min
