# for k in {0..12}; do
#   python run_mcmc.py --outdir mcmc_results_fig2_3 \
#     --labels gas iso ps loopA loopB ics fb_flat nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno \
#     --energy-bin "$k" --nwalkers 64 --nsteps 300000 --burn 25000 --thin 10 --tighten-halo-neg-bound --no-plots \
#     --require-autocorr \
#     --early-stop \
#     --autocorr-target 50 \
#     --autocorr-check-every 5000 \
#     --autocorr-min-steps 50000 \
#     --verbosity 1 \
#     --iso-target-e2 1e-4 \
#     --iso-anchor --iso-anchor-e2 1e-4 \
#     --iso-prior-sigma-dex 0.5 \
#     --cancellation-check \
#     --iso-prior-mode f --nonstable-prior-sigma 0.5
# done

common=(
  --outdir mcmc_results_fig2_3
  --labels gas iso ps loopA loopB ics fb_flat nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno
  --nwalkers 48 --nsteps 60000 --burn 500 --thin 5
  --tighten-halo-neg-bound --no-plots
  --require-autocorr 
# --early-stop
#   --autocorr-target 30 --autocorr-check-every 2000 --autocorr-min-steps 15000
  --verbosity 1
  --iso-target-e2 1e-4
  --iso-anchor --iso-anchor-e2 1e-4
  --iso-prior-sigma-dex 0.5
  --cancellation-check
  --iso-prior-mode f --nonstable-prior-sigma 0.5
)

for k in $(seq 0 12); do
  python run_mcmc.py "${common[@]}" --energy-bin "$k"
done