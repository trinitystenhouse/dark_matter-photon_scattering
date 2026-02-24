import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence


@dataclass
class MCMCResult:
    labels: List[str]
    f_ml: np.ndarray
    f_p16: np.ndarray
    f_p50: np.ndarray
    f_p84: np.ndarray
    loglike_ml: float
    chain: np.ndarray
    logprob: np.ndarray
    acceptance_fraction: np.ndarray
    used_emcee: bool


def loglike_poisson_counts(Cobs: np.ndarray, Cexp: np.ndarray) -> float:
    """
    Poisson log-likelihood up to an additive constant:
      lnL = sum_i [ Cobs_i * ln(Cexp_i) - Cexp_i ]
    Robust handling of Cexp==0:
      - If Cobs>0 and Cexp==0 -> -inf
      - If Cobs==0 and Cexp==0 -> contributes 0
    """
    Cobs = np.asarray(Cobs, float)
    Cexp = np.asarray(Cexp, float)

    if np.any(~np.isfinite(Cexp)):
        return -np.inf
    if np.any(Cexp < 0.0):
        return -np.inf

    pos = Cexp > 0.0
    if np.any((~pos) & (Cobs > 0.0)):
        return -np.inf

    # Only pos entries contribute Cobs*ln(Cexp); zeros contribute -0 = 0
    return float(np.sum(Cobs[pos] * np.log(Cexp[pos]) - Cexp[pos]))


def logprior_bounds_inclusive(f: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> float:
    """
    Flat prior within bounds, inclusive:
      lo <= f <= hi
    """
    for x, (lo, hi) in zip(f, bounds):
        if not np.isfinite(x):
            return -np.inf
        if (x < lo) or (x > hi):
            return -np.inf
    return 0.0


def totani_bounds(
    labels: Sequence[str],
    *,
    halo_keys: Sequence[str] = ("nfw", "halo"),
    lo_known: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Totani text: known components constrained f>=0 (or >0), halos allowed negative.
    We use inclusive bounds (>=0) to allow "start at 0" for some components.
    """
    hk = tuple(k.lower() for k in halo_keys)
    bnds: List[Tuple[float, float]] = []
    for lab in labels:
        key = lab.lower()
        if any(h in key for h in hk):
            bnds.append((-np.inf, np.inf))
        else:
            bnds.append((lo_known, np.inf))
    return bnds


def infer_iso_E2dNdE_at_f1(mu_iso: np.ndarray, denom: np.ndarray, Ectr_mev: float) -> float:
    """
    Given iso counts template mu_iso (for f_iso=1) and denom=expo*omega*dE,
    infer the median isotropic E^2 dN/dE that f_iso=1 corresponds to.
    """
    mu_iso = np.asarray(mu_iso, float)
    denom = np.asarray(denom, float)
    good = np.isfinite(mu_iso) & np.isfinite(denom) & (denom > 0)
    if not np.any(good):
        raise RuntimeError("Cannot infer isotropic normalization: denom has no valid entries.")
    I_med = float(np.median(mu_iso[good] / denom[good]))  # dN/dE
    return float((Ectr_mev**2) * I_med)


def totani_init_from_mu_counts(
    labels: Sequence[str],
    *,
    mu: np.ndarray,              # (ncomp, npix) counts-space templates for THIS energy bin
    denom: np.ndarray,           # (npix,) expo*omega*dE on the same masked pixels
    Ectr_mev: float,
    iso_target_E2: float = 1e-4, # Totani typical initial E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
    ps_keys: Sequence[str] = ("ps", "point_sources", "pointsources"),
    galprop_keys: Sequence[str] = ("gas", "ics", "galprop_gas", "galprop_ics"),
    iso_keys: Sequence[str] = ("iso", "isotropic"),
) -> np.ndarray:
    """
    Totani-style initialisation, but consistent with your parameterisation:
      - PS + GALPROP start at baseline normalization -> f=1
      - isotropic starts at target E^2 dN/dE, converted into your dimensionless f_iso
      - others start at 0
    """
    labels_l = [s.lower() for s in labels]
    ncomp = len(labels_l)
    f0 = np.zeros(ncomp, float)

    for j, lab in enumerate(labels_l):
        if lab in tuple(k.lower() for k in ps_keys):
            f0[j] = 1.0
        if lab in tuple(k.lower() for k in galprop_keys):
            f0[j] = 1.0

    # isotropic conversion
    jiso = None
    for k in iso_keys:
        k = k.lower()
        if k in labels_l:
            jiso = labels_l.index(k)
            break

    if jiso is not None:
        E2_at_f1 = infer_iso_E2dNdE_at_f1(mu[jiso], denom, Ectr_mev)
        if E2_at_f1 <= 0 or not np.isfinite(E2_at_f1):
            raise RuntimeError(f"Inferred iso E^2 dN/dE at f=1 is invalid: {E2_at_f1}")
        f0[jiso] = iso_target_E2 / E2_at_f1

    return f0

def totani_mcmc_fit(
    *,
    Cobs: np.ndarray,
    mu: np.ndarray,
    labels: Sequence[str],
    f_init: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    nwalkers: int = 64,
    nsteps: int = 5000,
    burn: int = 1000,
    thin: int = 5,
    init_jitter_frac: float = 1e-2,
    rng: Optional[np.random.Generator] = None,
    progress: bool = False,
) -> MCMCResult:
    rng = rng or np.random.default_rng()

    Cobs = np.asarray(Cobs, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 2:
        raise ValueError("mu must be 2D (ncomp, npix).")
    ncomp, npix = mu.shape
    if Cobs.shape[0] != npix:
        raise ValueError(f"Cobs has {Cobs.shape[0]} entries but mu has {npix}.")
    labels = list(labels)
    if len(labels) != ncomp:
        raise ValueError("labels length must match mu.shape[0].")
    f_init = np.asarray(f_init, dtype=float)
    if f_init.shape != (ncomp,):
        raise ValueError("f_init must be shape (ncomp,).")
    if len(bounds) != ncomp:
        raise ValueError("bounds length must match ncomp.")

    def logprob_fn(f: np.ndarray) -> float:
        lp = logprior_bounds_inclusive(f, bounds)
        if not np.isfinite(lp):
            return -np.inf
        Cexp = f @ mu
        ll = loglike_poisson_counts(Cobs, Cexp)
        return lp + ll

    # -------------------------------------------------------------------------
    # Initialize walkers near f_init, making sure each has finite logprob
    # -------------------------------------------------------------------------
    p0 = np.zeros((nwalkers, ncomp), dtype=float)
    scale = init_jitter_frac * (np.abs(f_init) + 1.0)

    for w in range(nwalkers):
        trial = f_init + rng.normal(0.0, scale, size=ncomp)

        # Try a bunch of times to get a finite logprob start
        for _ in range(200):
            # Enforce bounds by repairing any offending dimensions
            ok = True
            for j, (lo, hi) in enumerate(bounds):
                x = trial[j]
                if (not np.isfinite(x)) or (x < lo) or (x > hi):
                    ok = False
                    x = f_init[j] + rng.normal(0.0, scale[j])
                    if np.isfinite(lo) and lo >= 0.0:
                        x = max(lo, abs(x))
                    if np.isfinite(hi):
                        x = min(hi, x)
                    trial[j] = x

            if ok and np.isfinite(logprob_fn(trial)):
                break

            # otherwise, resample entire vector (helps if Cexp<=0 issues)
            trial = f_init + rng.normal(0.0, scale, size=ncomp)

        p0[w, :] = trial

    # -------------------------------------------------------------------------
    # Diagnose + respawn any remaining invalid walkers
    # -------------------------------------------------------------------------
    lp0 = np.array([logprob_fn(p0[w]) for w in range(nwalkers)], dtype=float)
    good = np.isfinite(lp0)
    if good.sum() < max(2, ncomp + 1):
        raise RuntimeError(f"Too few valid initial walkers: {good.sum()}/{nwalkers}")

    bad_idx = np.where(~good)[0]
    good_idx = np.where(good)[0]
    for w in bad_idx:
        src = rng.choice(good_idx)
        # tiny jitter around a valid walker, then repair bounds if needed
        trial = p0[src] + rng.normal(0.0, 1e-4 * (np.abs(p0[src]) + 1.0), size=ncomp)

        for _ in range(200):
            ok = True
            for j, (lo, hi) in enumerate(bounds):
                x = trial[j]
                if (not np.isfinite(x)) or (x < lo) or (x > hi):
                    ok = False
                    x = p0[src, j] + rng.normal(0.0, 1e-4 * (abs(p0[src, j]) + 1.0))
                    if np.isfinite(lo) and lo >= 0.0:
                        x = max(lo, abs(x))
                    if np.isfinite(hi):
                        x = min(hi, x)
                    trial[j] = x
            if ok and np.isfinite(logprob_fn(trial)):
                break
            trial = p0[src] + rng.normal(0.0, 1e-4 * (np.abs(p0[src]) + 1.0), size=ncomp)

        p0[w] = trial

    # Final check
    lp0 = np.array([logprob_fn(p0[w]) for w in range(nwalkers)], dtype=float)
    n_bad = int(np.sum(~np.isfinite(lp0)))
    print(f"Init walkers finite: {nwalkers - n_bad}/{nwalkers}  (respawned {len(bad_idx)})")
    if n_bad:
        # It's safer to fail than to run with dead walkers.
        raise RuntimeError(f"{n_bad} walkers still have -inf logprob after respawn; check bounds/Cexp positivity.")

    # -------------------------------------------------------------------------
    # Run sampler
    # -------------------------------------------------------------------------
    used_emcee = False
    try:
        import emcee  # type: ignore
        used_emcee = True
        sampler = emcee.EnsembleSampler(nwalkers, ncomp, logprob_fn)
        sampler.run_mcmc(p0, nsteps, progress=progress)
        try:
            tau = sampler.get_autocorr_time(tol=0)
            print("tau:", tau)
            n_over_tau = nsteps / tau
            print("N/tau:", n_over_tau)
            # Totani-style rule of thumb: require sufficiently many autocorrelation times.
            # Enforce N/tau >= 50 for all parameters.
            if np.any(~np.isfinite(n_over_tau)):
                raise RuntimeError(f"Non-finite N/tau encountered: {n_over_tau}")
            if np.any(n_over_tau < 50.0):
                bad = np.where(n_over_tau < 50.0)[0].tolist()
                raise RuntimeError(
                    f"Autocorr convergence check failed: N/tau < 50 for parameter indices {bad}. "
                    f"Min N/tau={float(np.min(n_over_tau)):.3f}. Increase nsteps."
                )
        except Exception as e:
            print("autocorr time not available yet:", repr(e))
        chain = sampler.get_chain()
        logprob = sampler.get_log_prob()
        acc = sampler.acceptance_fraction
    except Exception:
        # Simple RW Metropolis fallback
        chain = np.zeros((nsteps, nwalkers, ncomp), dtype=float)
        logprob = np.full((nsteps, nwalkers), -np.inf, dtype=float)
        acc = np.zeros(nwalkers, dtype=float)

        current = p0.copy()
        current_lp = np.array([logprob_fn(current[w]) for w in range(nwalkers)], dtype=float)
        prop_scale = 0.05 * (np.abs(f_init) + 1.0)

        for t in range(nsteps):
            for w in range(nwalkers):
                proposal = current[w] + rng.normal(0.0, prop_scale, size=ncomp)
                if not np.isfinite(logprior_bounds_inclusive(proposal, bounds)):
                    lp_new = -np.inf
                else:
                    lp_new = logprob_fn(proposal)

                if np.isfinite(lp_new) and (np.log(rng.random()) < (lp_new - current_lp[w])):
                    current[w] = proposal
                    current_lp[w] = lp_new
                    acc[w] += 1.0

            chain[t] = current
            logprob[t] = current_lp

        acc /= float(nsteps)

    # -------------------------------------------------------------------------
    # Summarize posterior
    # -------------------------------------------------------------------------
    if burn >= nsteps:
        raise ValueError("burn must be < nsteps.")

    post = chain[burn::thin, :, :].reshape(-1, ncomp)
    post_lp = logprob[burn::thin, :].reshape(-1)

    imax = int(np.nanargmax(post_lp))
    f_ml = post[imax].copy()
    loglike_ml = float(post_lp[imax])

    q16, q50, q84 = np.quantile(post, [0.16, 0.50, 0.84], axis=0)

    return MCMCResult(
        labels=labels,
        f_ml=f_ml,
        f_p16=q16,
        f_p50=q50,
        f_p84=q84,
        loglike_ml=loglike_ml,
        chain=chain,
        logprob=logprob,
        acceptance_fraction=acc,
        used_emcee=used_emcee,
    )