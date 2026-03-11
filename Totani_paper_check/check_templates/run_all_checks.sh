python check_bubbles_flat.py
python check_iso.py --scale-flux
python check_ps.py --scale-flux
python check_ics.py --scale-flux
python check_gas.py --scale-flux
python check_loopI.py
python check_nfw.py
python make_template_grid_21gev.py
python run_systematics_suite.py --cases-root ../mcmc/mcmc_results_fig2_3 --make-spectra --make-template-grids
