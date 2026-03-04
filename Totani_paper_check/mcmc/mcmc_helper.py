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
    negative_keys: Sequence[str] = ("nfw", "fb_neg"),
    halo_keys: Optional[Sequence[str]] = None,
    lo_known: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Totani text: known components constrained f>=0 (or >0), some components allowed negative.
    We use inclusive bounds (>=0) to allow "start at 0" for some components.
    """
    keys = list(negative_keys)
    if halo_keys is not None:
        keys.extend(list(halo_keys))
    nk = tuple(str(k).lower() for k in keys)
    bnds: List[Tuple[float, float]] = []
    for lab in labels:
        key = lab.lower()
        if any(n in key for n in nk):
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
    iso_target_E2: Optional[float] = 1e-4, # If None, iso starts at f=1 like other physical templates
    ps_keys: Sequence[str] = ("ps", "point_sources", "pointsources"),
    galprop_keys: Sequence[str] = ("gas", "ics", "galprop_gas", "galprop_ics"),
    iso_keys: Sequence[str] = ("iso", "isotropic"),
) -> np.ndarray:
    """
    Initialize MCMC parameters for counts-space templates.
    
    Physical templates (ps, gas, ics, iso) start at f=1 (their baseline normalization).
    Spatial shape templates (nfw, loopI, bubbles) start at 0.
    
    If iso_target_E2 is provided (legacy mode), iso will be rescaled to that target.
    """
    labels_l = [s.lower() for s in labels]
    ncomp = len(labels_l)
    f0 = np.zeros(ncomp, float)

    # Physical templates with known normalization: start at f=1
    for j, lab in enumerate(labels_l):
        if lab in tuple(k.lower() for k in ps_keys):
            f0[j] = 1.0
        if lab in tuple(k.lower() for k in galprop_keys):
            f0[j] = 1.0
        if lab in tuple(k.lower() for k in iso_keys):
            f0[j] = 1.0

    # Optional: rescale iso to target E^2 dN/dE (legacy mode)
    if iso_target_E2 is not None:
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
        
    iso_counts_sum_f1 = mu[jiso].sum()
    iso_counts_sum_init = f0[jiso] * iso_counts_sum_f1
    print(f"iso sum counts (f=1):   {iso_counts_sum_f1:.3e}")
    print(f"iso sum counts (init):  {iso_counts_sum_init:.3e}")
    print(f"iso implied E2 dN/dE (f=1): {E2I:.3e}")
    print(f"iso f_init: {f0[jiso]:.3e}")

    return f0
import numpy as np
from typing import Sequence, Tuple, Optional

def logprior_bounds_inclusive(x: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> float:
    x = np.asarray(x, float)
    for j, (lo, hi) in enumerate(bounds):
        v = x[j]
        if not np.isfinite(v):
            return -np.inf
        if np.isfinite(lo) and (v < lo):
            return -np.inf
        if np.isfinite(hi) and (v > hi):
            return -np.inf
    return 0.0


def loglike_poisson_counts(Cobs: np.ndarray, Cexp: np.ndarray) -> float:
    # Drop constants (log(y!)); requires Cexp > 0
    if np.any(~np.isfinite(Cexp)) or np.any(Cexp <= 0):
        return -np.inf
    return float(np.sum(Cobs * np.log(Cexp) - Cexp))


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
    require_autocorr: bool = False,
    early_stop: bool = False,
    autocorr_target: float = 50.0,
    autocorr_check_every: int = 1000,
    autocorr_min_steps: int = 2000,
    iso_prior_sigma_dex: Optional[float] = None,
    iso_prior_mode: str = "f",
    iso_prior_center: Optional[float] = None,
    nonstable_prior_sigma_dex: Optional[float] = None,
    nonstable_prior_centers: Optional[dict] = None,
    init_jitter_frac: float = 1e-2,
    rng: Optional[np.random.Generator] = None,
    progress: bool = False,
    verbosity: int = 2,
):
    def vprint(level: int, *args, **kwargs):
        if int(verbosity) >= int(level):
            print(*args, **kwargs)

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

    labels_l = [str(x).lower() for x in labels]

    def _log10_ratio_gauss(x: float, x0: float, sigma_dex: float) -> float:
        if (sigma_dex is None) or (sigma_dex <= 0) or (not np.isfinite(sigma_dex)):
            return 0.0
        if (not np.isfinite(x)) or (not np.isfinite(x0)) or (x <= 0) or (x0 <= 0):
            return -np.inf
        z = np.log10(x / x0)
        return -0.5 * (z / float(sigma_dex)) ** 2

    def _log10_ratio_gauss_upper_only(x: float, x0: float, sigma_dex: float) -> float:
        if (sigma_dex is None) or (sigma_dex <= 0) or (not np.isfinite(sigma_dex)):
            return 0.0
        if (not np.isfinite(x)) or (not np.isfinite(x0)) or (x <= 0) or (x0 <= 0):
            return -np.inf
        if x <= x0:
            return 0.0
        z = np.log10(x / x0)
        return -0.5 * (z / float(sigma_dex)) ** 2

    def logprior_gaussian(f: np.ndarray) -> float:
        lp = 0.0

        # ISO prior (log-space around iso_prior_center)
        if iso_prior_sigma_dex is not None:
            jiso = None
            for key in ("iso", "isotropic"):
                if key in labels_l:
                    jiso = labels_l.index(key)
                    break

            if jiso is not None:
                center = float(iso_prior_center) if (iso_prior_center is not None) else float(f_init[jiso])
                mode = str(iso_prior_mode).strip().lower()
                if mode in ("f_upper", "upper", "upper_only"):
                    lp_i = _log10_ratio_gauss_upper_only(float(f[jiso]), center, float(iso_prior_sigma_dex))
                else:
                    lp_i = _log10_ratio_gauss(float(f[jiso]), center, float(iso_prior_sigma_dex))
                if not np.isfinite(lp_i):
                    return -np.inf
                lp += lp_i

        # Nonstable priors (optional): dict(label->center)
        if nonstable_prior_sigma_dex is not None:
            centers = nonstable_prior_centers or {}
            for lab_key, ctr in centers.items():
                lk = str(lab_key).lower()
                if lk not in labels_l:
                    continue
                j = labels_l.index(lk)
                lp_j = _log10_ratio_gauss(float(f[j]), float(ctr), float(nonstable_prior_sigma_dex))
                if not np.isfinite(lp_j):
                    return -np.inf
                lp += lp_j

        return float(lp)

    def logprob_fn(f: np.ndarray) -> float:
        lp = logprior_bounds_inclusive(f, bounds)
        if not np.isfinite(lp):
            return -np.inf

        lp_g = logprior_gaussian(f)
        if not np.isfinite(lp_g):
            return -np.inf

        # Expected counts (npix,)
        Cexp = f @ mu

        # Poisson requires strictly positive expectation everywhere in the fit mask
        ll = loglike_poisson_counts(Cobs, Cexp)
        if not np.isfinite(ll):
            return -np.inf

        return float(lp + lp_g + ll)

    # --------------------------
    # Initialize walkers
    # --------------------------
    p0 = np.zeros((nwalkers, ncomp), dtype=float)
    scale = init_jitter_frac * (np.abs(f_init) + 1.0)

    for w in range(nwalkers):
        trial = f_init + rng.normal(0.0, scale, size=ncomp)

        for _ in range(200):
            # repair bounds component-wise
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
            trial = f_init + rng.normal(0.0, scale, size=ncomp)

        p0[w, :] = trial

    lp0 = np.array([logprob_fn(p0[w]) for w in range(nwalkers)], dtype=float)
    good = np.isfinite(lp0)
    if good.sum() < max(2, ncomp + 1):
        raise RuntimeError(f"Too few valid initial walkers: {good.sum()}/{nwalkers}")

    bad_idx = np.where(~good)[0]
    good_idx = np.where(good)[0]
    for w in bad_idx:
        src = rng.choice(good_idx)
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

    lp0 = np.array([logprob_fn(p0[w]) for w in range(nwalkers)], dtype=float)
    n_bad = int(np.sum(~np.isfinite(lp0)))
    vprint(2, f"Init walkers finite: {nwalkers - n_bad}/{nwalkers}  (respawned {len(bad_idx)})")
    if n_bad:
        raise RuntimeError(f"{n_bad} walkers still have -inf logprob after respawn; check bounds/Cexp positivity.")

    # --------------------------
    # Run emcee
    # --------------------------
    used_emcee = False
    try:
        import emcee  # type: ignore
        used_emcee = True

        sampler = emcee.EnsembleSampler(nwalkers, ncomp, logprob_fn)

        def _report_autocorr_status(*, n_done: int):
            try:
                tau = sampler.get_autocorr_time(tol=0)
                n_over_tau = float(n_done) / tau
                if np.any(~np.isfinite(n_over_tau)):
                    raise RuntimeError(f"Non-finite N/tau encountered: {n_over_tau}")
                ok = bool(np.all(n_over_tau >= float(autocorr_target)))
                if int(verbosity) >= 2:
                    vprint(2, "tau:", tau)
                    vprint(2, "N/tau:", n_over_tau)
                if int(verbosity) >= 1:
                    min_not = float(np.min(n_over_tau))
                    vprint(1, f"autocorr status: {'achieved' if ok else 'NOT achieved'} (min N/tau={min_not:.3f})")
            except Exception as e:
                if int(verbosity) >= 2:
                    vprint(2, "autocorr time not available yet:", repr(e))
                if int(verbosity) >= 1:
                    vprint(1, "autocorr status: unavailable")

        if early_stop:
            total = int(nsteps)
            step = max(1, int(autocorr_check_every))
            min_steps = int(autocorr_min_steps)
            state = None
            n_done = 0

            while n_done < total:
                n_chunk = min(step, total - n_done)
                state = sampler.run_mcmc(p0 if state is None else state, int(n_chunk), progress=progress)
                n_done += int(n_chunk)

                if n_done < min_steps or n_done <= burn:
                    continue

                try:
                    tau = sampler.get_autocorr_time(tol=0)
                    n_over_tau = n_done / tau
                    if np.any(~np.isfinite(n_over_tau)):
                        raise RuntimeError(f"Non-finite N/tau encountered: {n_over_tau}")
                    if np.all(n_over_tau >= float(autocorr_target)):
                        vprint(2, "tau:", tau)
                        vprint(2, "N/tau:", n_over_tau)
                        vprint(2, f"Early stop: reached N/tau >= {float(autocorr_target):g} at N={n_done}")
                        break
                except Exception as e:
                    if bool(require_autocorr):
                        raise RuntimeError(f"Autocorr convergence check requested but failed: {repr(e)}") from e

            _report_autocorr_status(n_done=int(n_done))
        else:
            sampler.run_mcmc(p0, int(nsteps), progress=progress)
            _report_autocorr_status(n_done=int(nsteps))

        chain = sampler.get_chain()
        logprob = sampler.get_log_prob()
        acc = sampler.acceptance_fraction

    except Exception:
        raise  # keep your fallback if you want; but this is usually better to fail loudly here

    # --------------------------
    # Summarize posterior
    # --------------------------
    nsteps_eff = int(chain.shape[0])
    if burn >= nsteps_eff:
        raise ValueError(f"burn must be < nsteps_done. burn={burn}, nsteps_done={nsteps_eff}")

    post = chain[burn::thin, :, :].reshape(-1, ncomp)
    post_lp = logprob[burn::thin, :].reshape(-1)
    if post.shape[0] == 0:
        raise RuntimeError(f"No posterior samples after burn/thin. nsteps_done={nsteps_eff}, burn={burn}, thin={thin}.")

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