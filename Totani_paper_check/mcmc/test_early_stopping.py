import sys
import types
import unittest

import numpy as np


def _install_fake_emcee(*, converge_after: int | None, tau: float = 50.0, raise_autocorr: bool = False):
    """Install a fake 'emcee' module into sys.modules.

    converge_after:
      - If int: once total steps >= converge_after, get_autocorr_time returns a small tau
        so that N/tau >= autocorr_target is achievable.
      - If None: never converges (tau always large).

    raise_autocorr:
      - If True: get_autocorr_time always raises (to test require_autocorr behavior).
    """

    class _State:
        pass

    class EnsembleSampler:
        def __init__(self, nwalkers, ndim, logprob_fn):
            self.nwalkers = int(nwalkers)
            self.ndim = int(ndim)
            self._logprob_fn = logprob_fn
            self._n_done = 0
            self._chain = np.zeros((0, self.nwalkers, self.ndim), dtype=float)
            self._logprob = np.zeros((0, self.nwalkers), dtype=float)
            self.acceptance_fraction = np.full(self.nwalkers, 0.25, dtype=float)

        def run_mcmc(self, state, nsteps, progress=False):
            nsteps = int(nsteps)
            self._n_done += nsteps
            # Ensure shapes look like emcee outputs: (nsteps_total, nwalkers, ndim)
            self._chain = np.zeros((self._n_done, self.nwalkers, self.ndim), dtype=float)
            self._logprob = np.zeros((self._n_done, self.nwalkers), dtype=float)
            return _State()

        def get_chain(self):
            return self._chain

        def get_log_prob(self):
            return self._logprob

        def get_autocorr_time(self, tol=0):
            if raise_autocorr:
                raise RuntimeError("autocorr unavailable")

            if converge_after is not None and self._n_done >= int(converge_after):
                # Return tau small enough that N/tau >= 50 when N>=converge_after
                # If N==converge_after and target=50, tau must be <= N/50.
                eff_tau = min(float(tau), float(self._n_done) / 50.0)
                return np.full(self.ndim, eff_tau, dtype=float)

            # Not converged yet => tau very large => N/tau small.
            return np.full(self.ndim, 1e9, dtype=float)

    fake = types.ModuleType("emcee")
    fake.EnsembleSampler = EnsembleSampler

    old = sys.modules.get("emcee")
    sys.modules["emcee"] = fake
    return old


def _restore_emcee(old):
    if old is None:
        sys.modules.pop("emcee", None)
    else:
        sys.modules["emcee"] = old


class TestEarlyStopping(unittest.TestCase):
    def _toy_problem(self):
        # Simple positive templates and counts.
        rng = np.random.default_rng(0)
        npix = 20
        labels = ["gas", "iso"]
        mu = np.vstack([
            np.full(npix, 1.0, dtype=float),
            np.full(npix, 0.2, dtype=float),
        ])
        f_init = np.array([1.0, 1.0], dtype=float)
        Cexp = f_init @ mu
        Cobs = rng.poisson(Cexp).astype(float)
        bounds = [(0.0, np.inf), (0.0, np.inf)]
        return Cobs, mu, labels, f_init, bounds

    def test_early_stop_stops_before_nsteps_when_converged(self):
        old = _install_fake_emcee(converge_after=3000)
        try:
            from mcmc.mcmc_helper import totani_mcmc_fit

            Cobs, mu, labels, f_init, bounds = self._toy_problem()
            res = totani_mcmc_fit(
                Cobs=Cobs,
                mu=mu,
                labels=labels,
                f_init=f_init,
                bounds=bounds,
                nwalkers=8,
                nsteps=10000,
                burn=1000,
                thin=10,
                early_stop=True,
                require_autocorr=False,
                autocorr_target=50.0,
                autocorr_check_every=1000,
                autocorr_min_steps=2000,
                iso_prior_sigma_dex=None,
                verbosity=0,
            )

            self.assertLess(int(res.chain.shape[0]), 10000)
            self.assertGreaterEqual(int(res.chain.shape[0]), 3000)
        finally:
            _restore_emcee(old)

    def test_early_stop_runs_full_length_if_never_converges(self):
        old = _install_fake_emcee(converge_after=None)
        try:
            from mcmc.mcmc_helper import totani_mcmc_fit

            Cobs, mu, labels, f_init, bounds = self._toy_problem()
            res = totani_mcmc_fit(
                Cobs=Cobs,
                mu=mu,
                labels=labels,
                f_init=f_init,
                bounds=bounds,
                nwalkers=8,
                nsteps=5000,
                burn=1000,
                thin=10,
                early_stop=True,
                require_autocorr=False,
                autocorr_target=50.0,
                autocorr_check_every=1000,
                autocorr_min_steps=2000,
                iso_prior_sigma_dex=None,
                verbosity=0,
            )

            self.assertEqual(int(res.chain.shape[0]), 5000)
        finally:
            _restore_emcee(old)


if __name__ == "__main__":
    unittest.main()
