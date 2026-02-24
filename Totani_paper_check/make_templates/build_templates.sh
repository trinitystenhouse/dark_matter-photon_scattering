#!/bin/bash

python mask_extended_sources.py
python build_ps_template.py
python build_gas_template.py
python build_ics_template.py
python build_iso_template.py
python build_loopI_template.py
python build_nfw_template.py
python build_bubbles_templates.py --restrict-to-vertices
