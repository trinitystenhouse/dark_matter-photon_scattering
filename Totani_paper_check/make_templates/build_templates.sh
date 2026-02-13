#!/bin/bash

python mask_extended_sources.py
python build_ps_template.py
python build_gas_template.py
python build_ics_template.py
python build_iem_iso_templates.py
python build_loopI_template.py
python build_nfw_template.py
python build_bubbles_flat_binary.py
python build_bubbles_posneg_totani.py