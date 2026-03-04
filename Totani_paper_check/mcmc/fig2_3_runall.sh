for k in {0..12}; do
  python run_mcmc.py --outdir mcmc_results_fig2_3 \
    --labels gas iso ps loopA loopB ics fb_flat nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno \
    --energy-bin "$k" --nwalkers 64 --nsteps 30000 --burn 2500 --thin 10 --tighten-halo-neg-bound --no-plots \
    --verbosity 1 \
    --iso-target-e2 1e-4 \
    --iso-anchor --iso-anchor-e2 1e-4 \
    --iso-prior-sigma-dex 0 \
    --cancellation-check \
    --iso-prior-mode f --nonstable-prior-sigma 0.5 \
    --require-autocorr 
done

    #--require-autocorr \