# for k in {0..5}; do
#   python run_mcmc.py --outdir mcmc_results_test \
#       --labels gas iso ps loopA loopB ics fb_flat nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno \
#       --energy-bin "$k" --nwalkers 64 --nsteps 3000 --burn 100 --thin 10 --tighten-halo-neg-bound --no-plots \
#       --verbosity 2 \
#       --iso-target-e2 1e-4 \
#       --iso-anchor --iso-anchor-e2 1e-3 \
#       --iso-prior-sigma-dex 0.2 \
#       --cancellation-check \
#       --iso-prior-mode f_upper --nonstable-prior-sigma 0.5 \
#       --iso-floor-e2 1e-4
# done
    #--require-autocorr \

for k in {0..12}; do
  python run_mcmc.py --outdir mcmc_results_test \
      --labels gas iso ps loopA loopB ics fb_flat nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno \
      --energy-bin "$k" --nwalkers 64 --nsteps 30000 --burn 1000 --thin 10 --tighten-halo-neg-bound --no-plots \
      --verbosity 1 \
      --iso-target-e2 1e-4 \
      --iso-anchor --iso-anchor-e2 1e-4 \
      --iso-prior-sigma-dex 0.1 \
      --cancellation-check \
      --iso-prior-mode f \
      --iso-floor-e2 5e-6  --nonstable-prior-sigma 0.5 
done
    