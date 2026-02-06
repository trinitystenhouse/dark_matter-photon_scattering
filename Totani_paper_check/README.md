# Totani paper check (WIP)

This folder is a work-in-progress reimplementation/check of several results in Totani (2025) using Fermi-LAT data products prepared under `${REPO_PATH}/fermi_data/totani/`.

## Setup

From the **repo root**:

```bash
source Totani_paper_check/setup.sh
```

This:

- activates the `fermi` conda environment (as currently written)
- installs the top-level `helpers` package (repo root) in editable mode
- installs `Totani_paper_check` (including `totani_helpers`) in editable mode
- exports `REPO_PATH` so scripts can locate `${REPO_PATH}/fermi_data/totani/...` from any working directory

## Data layout

This code expects the Totani-style data products under:

- `${REPO_PATH}/fermi_data/totani/`
  - `processed/`
    - `counts_ccube_1000to1000000.fits`
    - `expcube_1000to1000000.fits`
    - `templates/`

## Workflow (recommended)

1. Download Totani-style data

   ```bash
   bash data_download/download_fermi_totani.sh
   ```

2. Process data products

   ```bash
   bash data_download/process_totani.sh
   ```

3. Sanity checks

   ```bash
   python sanity_checks/sanity_checks.py
   ```

4. Generate templates

   Template builders live in `make_templates/`. Typical order:

   ```bash
   python make_templates/mask_extended_sources.py
   python make_templates/build_ps_template.py
   python make_templates/build_iem_iso_templates.py
   python make_templates/build_nfw_template.py
   python make_templates/build_loopI_template.py
   python make_templates/build_fermi_bubbles_templates.py
   ```

   Outputs are written to:

   - `${REPO_PATH}/fermi_data/totani/processed/templates/`

  Note that `mask_extended_sources.py` must be run first.

5. Replicate figures

   Currently implemented (WIP):

   - Figure 1:

     ```bash
     python fig1/make_totani_fig1_maps.py
     ```

   - Figures 2–3:

     ```bash
     python fig2_3/make_totani_fig2_fig3_all.py
     ```

   - Figure 9:

     ```bash
     python fig9/make_totani_fig9_likelihood.py
     ```

     This is currently a naive first attempt and is not validated.

## Known differences vs Totani (2025)

This repository currently differs from Totani (2025) in important ways:

- The background model uses **IEM as a combined template**, rather than splitting **gas** and **ICS** components.
  - This is a known discrepancy and is one of the main items to work on.

## Known issues / WIP status

- **Figure 9**: the likelihood scan is currently not working / not validated yet.
- **Figures 2–3**: the script generates and uses templates, but the templates are **not perfect yet** (needs iteration/validation).

To improve agreement with Totani, the next major work items are:

- splitting IEM into gas + ICS templates (and updating the fitting scripts accordingly)
- validating/fixing the Fig 9 likelihood workflow
- iterating on template construction/normalization choices used by Fig 2/3
