#!/usr/bin/env python3
import argparse
import json
import os
import runpy
import sys
from typing import Any, Dict, List, Optional, Sequence, Union


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.totani_io import read_counts_and_ebounds


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-merged dict, where b overrides a."""
    out: Dict[str, Any] = dict(a)
    for k, vb in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(vb, dict):
            out[k] = _deep_merge(out[k], vb)
        else:
            out[k] = vb
    return out


def _abspath_from(base_dir: str, p: Optional[Union[str, os.PathLike]]) -> Optional[str]:
    if p is None:
        return None
    p = str(p)
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def _energy_bins_from_cfg(cfg: Dict[str, Any], *, counts_path: str) -> List[int]:
    bins = cfg.get("energy_bins", "all")
    if isinstance(bins, str):
        if bins.lower() != "all":
            raise SystemExit("energy_bins must be 'all' or a list of integers")
        counts, _hdr, _Emin, _Emax, _Ectr, _dE = read_counts_and_ebounds(counts_path)
        nE = int(counts.shape[0])
        return list(range(nE))

    if isinstance(bins, Sequence):
        out: List[int] = []
        for x in bins:
            out.append(int(x))
        return out

    raise SystemExit("energy_bins must be 'all' or a list of integers")


def _run_one(argv_tail: List[str]) -> None:
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "run_mcmc.py"))
    old_argv = sys.argv
    try:
        sys.argv = [script_path] + list(argv_tail)
        runpy.run_path(script_path, run_name="__main__")
    finally:
        sys.argv = old_argv


def _labels_from_argv(argv: Sequence[str]) -> Optional[List[str]]:
    """Best-effort extraction of --labels ... from a run_mcmc.py argv list."""
    try:
        idx = list(argv).index("--labels")
    except ValueError:
        return None
    out: List[str] = []
    for x in list(argv)[idx + 1 :]:
        if str(x).startswith("--"):
            break
        out.append(str(x))
    return out or None


def _print_prior_summary(*, cfg: Dict[str, Any], base_argv: Optional[Sequence[str]] = None) -> None:
    """Print config-level (pre-run) prior specs.

    Note: this prints what the config *requests*. Some priors (e.g. iso anchor center)
    are energy-bin dependent and are computed inside run_mcmc.py.
    """
    verbosity = int(cfg.get("verbosity", 1) or 0)
    if verbosity < 1:
        return

    labels = cfg.get("labels", None)
    if labels is not None:
        if isinstance(labels, list):
            labels_l = [str(x) for x in labels]
        else:
            labels_l = None
    else:
        labels_l = _labels_from_argv(base_argv or [])

    nonstable_centers = cfg.get("nonstable_prior_centers", {})
    component_priors = cfg.get("component_priors", {})

    negative_keys = cfg.get("negative_keys", ["nfw", "fb_neg"])
    if not isinstance(negative_keys, list):
        negative_keys = [str(negative_keys)]
    negative_keys_l = [str(x).lower() for x in negative_keys]

    component_priors_l = None
    if isinstance(component_priors, dict):
        component_priors_l = {str(k).lower(): v for k, v in component_priors.items()}

    nonstable_centers_l = None
    if isinstance(nonstable_centers, dict):
        nonstable_centers_l = {str(k).lower(): v for k, v in nonstable_centers.items()}

    extra_keys: List[str] = []
    if isinstance(nonstable_centers, dict):
        extra_keys.extend([str(k) for k in nonstable_centers.keys()])
    if isinstance(component_priors, dict):
        extra_keys.extend([str(k) for k in component_priors.keys()])

    all_labels: List[str] = []
    if labels_l is not None:
        all_labels.extend(labels_l)
    for k in extra_keys:
        if k not in all_labels:
            all_labels.append(k)

    print("\nConfig prior settings:")
    print(f"  iso_prior_sigma_dex={cfg.get('iso_prior_sigma_dex', None)} iso_prior_mode={cfg.get('iso_prior_mode', None)}")
    print(f"  iso_anchor={cfg.get('iso_anchor', None)} iso_anchor_e2={cfg.get('iso_anchor_e2', None)} iso_floor_e2={cfg.get('iso_floor_e2', None)}")
    print(f"  nonstable_prior_sigma={cfg.get('nonstable_prior_sigma', None)}")
    if isinstance(nonstable_centers, dict) and nonstable_centers:
        print(f"  nonstable_prior_centers keys={sorted([str(k) for k in nonstable_centers.keys()])}")
    if isinstance(component_priors, dict) and component_priors:
        print(f"  component_priors keys={sorted([str(k) for k in component_priors.keys()])}")

    if all_labels:
        print("\nPriors requested (per label, pre-run):")
        for lab in all_labels:
            lk = str(lab).lower()
            pieces: List[str] = []

            # Bounds always apply (flat prior)
            if any(nk in lk for nk in negative_keys_l):
                pieces.append("bounds=flat(-inf,+inf)")
            else:
                pieces.append("bounds=flat([0,+inf])")

            if lk in ("iso", "isotropic"):
                if "iso_prior_sigma_dex" in cfg:
                    pieces.append(
                        f"iso_log10_ratio_gauss(mode={cfg.get('iso_prior_mode', 'f')}, sigma_dex={cfg.get('iso_prior_sigma_dex')})"
                    )
                if cfg.get("iso_anchor", None) is not None:
                    pieces.append(f"iso_anchor={bool(cfg.get('iso_anchor'))} (center depends on energy bin)")
                if cfg.get("iso_floor_e2", None) is not None:
                    try:
                        floor = float(cfg.get("iso_floor_e2"))
                        if floor > 0:
                            pieces.append(f"iso_floor_e2={floor:.6g} (=> f_iso lower bound depends on energy bin)")
                    except Exception:
                        pass

            if cfg.get("nonstable_prior_sigma", None) is not None:
                # nonstable never applies to iso (enforced in mcmc_helper.py)
                if lk not in ("iso", "isotropic"):
                    # If explicit centers provided in config, apply to those keys.
                    if (nonstable_centers_l is not None) and (lk in nonstable_centers_l):
                        pieces.append(
                            f"nonstable_log10_ratio_gauss(center={nonstable_centers_l.get(lk)}, sigma_dex={cfg.get('nonstable_prior_sigma')})"
                        )
                    else:
                        # Otherwise, run_mcmc.py defaults to applying this to gas/ics/ps (if present) centered on Totani init f0.
                        if lk in ("gas", "ics", "ps"):
                            pieces.append(
                                f"nonstable_log10_ratio_gauss(center=TotaniInit(f0), sigma_dex={cfg.get('nonstable_prior_sigma')})"
                            )

            if isinstance(component_priors, dict):
                spec = None
                if component_priors_l is not None:
                    spec = component_priors_l.get(lk, None)
                if isinstance(spec, dict):
                    if ("center_e2" in spec) and ("sigma_dex" in spec):
                        pieces.append(
                            f"component_log10_ratio_gauss(mode={spec.get('mode','f')}, center_e2={spec.get('center_e2')} (converted per bin), sigma_dex={spec.get('sigma_dex')})"
                        )
                    elif ("center" in spec) and ("sigma_dex" in spec):
                        pieces.append(
                            f"component_log10_ratio_gauss(mode={spec.get('mode','f')}, center={spec.get('center')}, sigma_dex={spec.get('sigma_dex')})"
                        )
            print(f"  {lab}: " + " | ".join(pieces))


def _maybe_run_plots(*, cfg: Dict[str, Any], cfg_dir: str, dry_run: bool) -> None:
    plot_cfgs = cfg.get("plot_configs", None)
    if plot_cfgs is None:
        return
    if not isinstance(plot_cfgs, list):
        raise SystemExit("Config key 'plot_configs' must be a list of paths")

    # Resolve common overrides from the merged config
    mcmc_dir = _abspath_from(cfg_dir, _pop_path(cfg, "mcmc_dir"))
    counts = _abspath_from(cfg_dir, _pop_path(cfg, "counts"))
    expo = _abspath_from(cfg_dir, _pop_path(cfg, "expo"))
    templates_dir = _abspath_from(cfg_dir, _pop_path(cfg, "templates_dir"))
    ext_mask = _abspath_from(cfg_dir, _pop_path(cfg, "ext_mask"))
    mcmc_stat = cfg.get("mcmc_stat", None)

    # Lazy import so plotting deps only load when requested
    from figures.make_figures_from_config import main as _plot_main

    for pcfg in plot_cfgs:
        pcfg_abs = _abspath_from(cfg_dir, pcfg)
        argv_plot: List[str] = ["--config", str(pcfg_abs)]
        if mcmc_dir is not None:
            argv_plot += ["--mcmc-dir", str(mcmc_dir)]
        if counts is not None:
            argv_plot += ["--counts", str(counts)]
        if expo is not None:
            argv_plot += ["--expo", str(expo)]
        if templates_dir is not None:
            argv_plot += ["--templates-dir", str(templates_dir)]
        if ext_mask is not None:
            argv_plot += ["--ext-mask", str(ext_mask)]
        if mcmc_stat is not None:
            argv_plot += ["--mcmc-stat", str(mcmc_stat)]

        if dry_run:
            print("make_figures_from_config.py", " ".join(argv_plot))
        else:
            _plot_main(argv_plot)


def _pop_path(cfg: Dict[str, Any], key: str) -> Optional[str]:
    v = cfg.get(key, None)
    return None if v is None else str(v)


def _add_arg(argv: List[str], flag: str, value: Optional[Union[str, int, float, bool]] = None) -> None:
    if value is None:
        return
    argv.append(str(flag))
    argv.append(str(value))


def _add_boolopt(argv: List[str], flag: str, value: Optional[bool]) -> None:
    if value is None:
        return
    argv.append(str(flag) if bool(value) else str("--no-" + str(flag).lstrip("-")))


def _build_argv_from_structured(cfg: Dict[str, Any], *, cfg_dir: str) -> List[str]:
    """Build argv for run_mcmc.py from structured config fields.

    Supported keys (top-level):
      - mcmc_dir, counts, expo, templates_dir, ext_mask
      - roi_lon, roi_lat, include_disk (bool), disk_cut, cell_deg, binsz
      - energy_bins ("all" or list[int])
      - nwalkers, nsteps, burn, thin
      - early_stop (bool), require_autocorr (bool), autocorr_target (float), autocorr_check_every (int), autocorr_min_steps (int)
      - labels (list[str])
      - iso_target_e2 (float or null)
      - iso_anchor (bool), iso_anchor_e2 (float), iso_floor_e2 (float), iso_prior_sigma_dex (float), iso_prior_mode (str)
      - negative_keys (list[str]), tighten_negative_bounds (bool)
      - progress (bool), no_plots (bool)
      - verbosity (int: 0,1,2)
      - extra_argv (list[str]) appended verbatim
    """

    argv: List[str] = []

    mcmc_dir = _abspath_from(cfg_dir, _pop_path(cfg, "mcmc_dir"))
    counts = _abspath_from(cfg_dir, _pop_path(cfg, "counts"))
    expo = _abspath_from(cfg_dir, _pop_path(cfg, "expo"))
    templates_dir = _abspath_from(cfg_dir, _pop_path(cfg, "templates_dir"))
    ext_mask = _abspath_from(cfg_dir, _pop_path(cfg, "ext_mask"))

    _add_arg(argv, "--outdir", mcmc_dir)
    _add_arg(argv, "--counts", counts)
    _add_arg(argv, "--expo", expo)
    _add_arg(argv, "--templates-dir", templates_dir)
    _add_arg(argv, "--ext-mask", ext_mask)

    _add_arg(argv, "--roi-lon", cfg.get("roi_lon", None))
    _add_arg(argv, "--roi-lat", cfg.get("roi_lat", None))
    _add_arg(argv, "--disk-cut", cfg.get("disk_cut", None))
    _add_arg(argv, "--cell-deg", cfg.get("cell_deg", None))
    _add_arg(argv, "--binsz", cfg.get("binsz", None))

    include_disk = cfg.get("include_disk", None)
    if include_disk is not None and (not bool(include_disk)):
        argv.append("--exclude-disk")

    _add_arg(argv, "--nwalkers", cfg.get("nwalkers", None))
    _add_arg(argv, "--nsteps", cfg.get("nsteps", None))
    _add_arg(argv, "--burn", cfg.get("burn", None))
    _add_arg(argv, "--thin", cfg.get("thin", None))
    _add_arg(argv, "--nprocs", cfg.get("nprocs", None))

    if "early_stop" in cfg:
        _add_boolopt(argv, "--early-stop", bool(cfg.get("early_stop")))
    if "require_autocorr" in cfg:
        if bool(cfg.get("require_autocorr")):
            argv.append("--require-autocorr")
    if "autocorr_target" in cfg:
        _add_arg(argv, "--autocorr-target", cfg.get("autocorr_target"))
    if "autocorr_check_every" in cfg:
        _add_arg(argv, "--autocorr-check-every", cfg.get("autocorr_check_every"))
    if "autocorr_min_steps" in cfg:
        _add_arg(argv, "--autocorr-min-steps", cfg.get("autocorr_min_steps"))

    mcmc_stat = cfg.get("mcmc_stat", None)
    if mcmc_stat is not None:
        _add_arg(argv, "--mcmc-stat", str(mcmc_stat))

    if "iso_target_e2" in cfg:
        _add_arg(argv, "--iso-target-e2", cfg.get("iso_target_e2"))

    labels = cfg.get("labels", None)
    if labels is not None:
        if not isinstance(labels, list):
            raise SystemExit("Config key 'labels' must be a list")
        argv.append("--labels")
        argv.extend([str(x) for x in labels])

    negative_keys = cfg.get("negative_keys", None)
    if negative_keys is not None:
        if not isinstance(negative_keys, list):
            raise SystemExit("Config key 'negative_keys' must be a list")
        argv.append("--negative-keys")
        argv.extend([str(x) for x in negative_keys])

    if "tighten_negative_bounds" in cfg:
        _add_boolopt(argv, "--tighten-negative-bounds", bool(cfg.get("tighten_negative_bounds")))

    if "iso_anchor" in cfg:
        _add_boolopt(argv, "--iso-anchor", bool(cfg.get("iso_anchor")))
    if "iso_anchor_e2" in cfg:
        _add_arg(argv, "--iso-anchor-e2", cfg.get("iso_anchor_e2"))
    if "iso_floor_e2" in cfg:
        _add_arg(argv, "--iso-floor-e2", cfg.get("iso_floor_e2"))
    if "iso_prior_sigma_dex" in cfg:
        _add_arg(argv, "--iso-prior-sigma-dex", cfg.get("iso_prior_sigma_dex"))
    if "iso_prior_mode" in cfg:
        _add_arg(argv, "--iso-prior-mode", cfg.get("iso_prior_mode"))

    if "nonstable_prior_sigma" in cfg:
        _add_arg(argv, "--nonstable-prior-sigma", cfg.get("nonstable_prior_sigma"))
    if "nonstable_prior_centers" in cfg:
        # Pass as JSON string; run_mcmc.py will validate/parse
        _add_arg(argv, "--nonstable-prior-centers", json.dumps(cfg.get("nonstable_prior_centers")))

    if "component_priors" in cfg:
        # Pass as JSON string; run_mcmc.py will validate/parse
        _add_arg(argv, "--component-priors", json.dumps(cfg.get("component_priors")))

    if "verbosity" in cfg:
        _add_arg(argv, "--verbosity", cfg.get("verbosity"))

    if "progress" in cfg:
        _add_boolopt(argv, "--progress", bool(cfg.get("progress")))
    if cfg.get("no_plots", False):
        argv.append("--no-plots")

    extra_argv = cfg.get("extra_argv", None)
    if extra_argv is not None:
        if not isinstance(extra_argv, list):
            raise SystemExit("Config key 'extra_argv' must be a list")
        argv.extend([str(x) for x in extra_argv])

    return argv


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run MCMC fits from a JSON config.")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument(
        "--base-config",
        default=None,
        help="Optional path to a base JSON config with shared parameters; overridden by --config.",
    )
    ap.add_argument("--energy-bin", type=int, default=None, help="Override: run only this single energy bin")
    ap.add_argument("--dry-run", action="store_true", help="Print the resolved commands but do not execute")

    args = ap.parse_args(argv)

    cfg_path = os.path.abspath(str(args.config))
    cfg_dir = os.path.dirname(cfg_path)
    run_cfg = _load_json(cfg_path)

    base_cfg: Dict[str, Any] = {}
    base_path = args.base_config or run_cfg.get("base_config", None)
    if base_path is not None:
        base_abs = _abspath_from(cfg_dir, base_path)
        base_cfg = _load_json(str(base_abs))

    cfg = _deep_merge(base_cfg, run_cfg)

    # Backward-compat: if 'argv' list is provided, use it as-is (after resolving relative paths for --counts).
    if "argv" in cfg and cfg.get("argv") is not None:
        base_argv = cfg.get("argv")
        if not isinstance(base_argv, list):
            raise SystemExit("Config 'argv' must be a list")
        base_argv = [str(x) for x in base_argv]

        counts_path = cfg.get("counts", None)
        if counts_path is not None:
            counts_abs = _abspath_from(cfg_dir, counts_path)
            for i, x in enumerate(base_argv):
                if x == "--counts" and i + 1 < len(base_argv):
                    base_argv[i + 1] = str(counts_abs)
                    break
            else:
                base_argv = ["--counts", str(counts_abs)] + base_argv
            counts_path = str(counts_abs)
        else:
            counts_path = None
            for i, x in enumerate(base_argv):
                if x == "--counts" and i + 1 < len(base_argv):
                    counts_path = str(_abspath_from(cfg_dir, base_argv[i + 1]))
                    base_argv[i + 1] = str(counts_path)
                    break
            if counts_path is None:
                raise SystemExit("Provide 'counts' in config or include --counts <path> in config argv")
    else:
        base_argv = _build_argv_from_structured(cfg, cfg_dir=cfg_dir)
        counts_path = cfg.get("counts", None)
        if counts_path is None:
            raise SystemExit("Structured config must set 'counts'")
        counts_path = str(_abspath_from(cfg_dir, counts_path))

    _print_prior_summary(cfg=cfg, base_argv=base_argv)

    if args.energy_bin is not None:
        energy_bins = [int(args.energy_bin)]
    else:
        energy_bins = _energy_bins_from_cfg(cfg, counts_path=str(counts_path))

    for k in energy_bins:
        argv_k = list(base_argv) + ["--energy-bin", str(int(k))]
        if args.dry_run:
            print("run_mcmc.py", " ".join(argv_k))
        else:
            _run_one(argv_k)

    # Optional automation: generate plots after successful MCMC completion.
    _maybe_run_plots(cfg=cfg, cfg_dir=cfg_dir, dry_run=bool(args.dry_run))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
