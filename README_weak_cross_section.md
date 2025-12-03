# Weak Cross-Section Analysis

This notebook contains theoretical calculations and visualizations of weak cross-sections for dark matter (DM) interactions with photons, specifically focusing on the process χγ → χγ (dark matter scattering with photons).

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Integration with Fermi Data](#integration-with-fermi-data)
- [Theoretical Background](#theoretical-background)
- [License](#license)

## Overview
This notebook implements theoretical calculations of differential cross-sections for dark matter-photon interactions, with a focus on weak-scale physics. It generates various plots to analyze the relationship between cross-sections, photon energies, and scattering angles.

## Prerequisites
- Python 3.6+
- Required Python packages:
  - NumPy
  - Matplotlib
  - SciPy
  - `trinity_plotting` (custom module for plot styling)

## Installation
1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements_fermi.txt
   ```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook weak_cross_section.ipynb
   ```
2. Run the cells sequentially to perform calculations and generate plots
3. The notebook will save output plots to the `plots/` directory

## Output Files
The notebook generates the following plots in the `plots/` directory:
- `Weakly_dep_on_angle.png`: Shows the angular dependence of the differential cross-section
- `Weakly_dep_on_egamma.png`: Shows the photon energy dependence of the differential cross-section
- `Weakly_egamma_vs_angle.png`: Shows the cross section dependence on photon energy and angle as a heatmap
- `Weakly_full_no_log.png`: Displays the total cross-section as a function of photon energy

## Integration with Fermi Data
To compare theoretical predictions with actual Fermi-LAT data:
Read `README_FermiLAT_Data.md` for instructions on how to process Fermi-LAT data.

## Theoretical Background
The notebook implements calculations for the following processes:
- W boson loop contributions
- Fermion loop contributions
- Combined (full) cross-section calculations

Calculations are performed in both center-of-mass and lab frames, with support for different dark matter masses (default: 1 TeV).

