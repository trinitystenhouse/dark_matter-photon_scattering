#!/bin/bash

TOTANI_DEBUG_ENERGY=1 python mask_extended_sources.py
TOTANI_DEBUG_ENERGY=1 python build_ps_template.py
TOTANI_DEBUG_ENERGY=1 python build_gas_template.py
TOTANI_DEBUG_ENERGY=1 python build_ics_template.py
TOTANI_DEBUG_ENERGY=1 python build_iso_template.py
TOTANI_DEBUG_ENERGY=1 python build_loopI_template.py
TOTANI_DEBUG_ENERGY=1 python build_nfw_template.py
TOTANI_DEBUG_ENERGY=1 python build_bubbles_templates.py \
  --force-bubbles-close-residual \
  --k-construct 2 \
  --restrict-to-vertices \
  --disk-cut-deg 0 \
  --no-keep-by-hemisphere \
  --components gas ics iso ps loopA loopB nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno
TOTANI_DEBUG_ENERGY=1 python build_nfw_template.py --rho-power 2
TOTANI_DEBUG_ENERGY=1 python build_nfw_template.py --rho-power 1