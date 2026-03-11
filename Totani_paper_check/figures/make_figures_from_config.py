#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.mcmc_plotter import make_plots_from_mcmc


def _repo_dir() -> str:
    return os.environ.get(
        "REPO_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )


def _data_dir(repo_dir: str) -> str:
    return os.path.join(repo_dir, "fermi_data", "totani")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    v = cfg.get(key, default)
    return default if v is None else v


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Make Totani paper figures using a single driver and a per-figure JSON config. "
            "The config controls labels, I/O paths, and naming."
        )
    )
    ap.add_argument("--config", required=True, help="Path to a JSON config file")

    ap.add_argument("--mcmc-dir", default=None, help="Override config mcmc_dir")
    ap.add_argument("--outdir", default=None, help="Override config outdir")
    ap.add_argument("--templates-dir", default=None, help="Override config templates_dir")
    ap.add_argument("--counts", default=None, help="Override config counts")
    ap.add_argument("--expo", default=None, help="Override config expo")
    ap.add_argument("--ext-mask", default=None, help="Override config ext_mask")

    ap.add_argument("--mcmc-stat", default=None, choices=["f_ml", "f_p50", "f_p16", "f_p84"], help="Override config mcmc_stat")

    ap.add_argument("--roi-lon", type=float, default=None)
    ap.add_argument("--roi-lat", type=float, default=None)
    ap.add_argument("--disk-cut", type=float, default=None)
    ap.add_argument("--binsz", type=float, default=None)

    args = ap.parse_args(argv)

    cfg_path = os.path.abspath(str(args.config))
    cfg_dir = os.path.dirname(cfg_path)
    cfg = _load_json(cfg_path)

    repo_dir = _repo_dir()
    data_dir = _data_dir(repo_dir)

    fig = str(_get(cfg, "fig", "fig"))
    plot_style = str(_get(cfg, "plot_style", "totani"))

    mcmc_dir = args.mcmc_dir or _get(cfg, "mcmc_dir", None)
    if mcmc_dir is None:
        raise SystemExit("Config must set 'mcmc_dir' or pass --mcmc-dir")
    if not os.path.isabs(str(mcmc_dir)):
        mcmc_dir = os.path.abspath(os.path.join(cfg_dir, str(mcmc_dir)))

    outdir = args.outdir or _get(cfg, "outdir", None)
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), f"plots_{fig}")
    if not os.path.isabs(str(outdir)):
        outdir = os.path.abspath(os.path.join(cfg_dir, str(outdir)))

    counts = args.counts or _get(cfg, "counts", os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"))
    expo = args.expo or _get(cfg, "expo", os.path.join(data_dir, "processed", "expcube_1000to1000000.fits"))

    templates_dir = args.templates_dir or _get(cfg, "templates_dir", os.path.join(data_dir, "processed", "templates"))
    ext_mask = args.ext_mask or _get(cfg, "ext_mask", os.path.join(data_dir, "processed", "templates", "mask_extended_sources.fits"))

    mcmc_stat = args.mcmc_stat or str(_get(cfg, "mcmc_stat", "f_ml"))

    roi_lon = float(args.roi_lon) if args.roi_lon is not None else float(_get(cfg, "roi_lon", 60.0))
    roi_lat = float(args.roi_lat) if args.roi_lat is not None else float(_get(cfg, "roi_lat", 60.0))
    disk_cut = float(args.disk_cut) if args.disk_cut is not None else float(_get(cfg, "disk_cut", 10.0))
    binsz = float(args.binsz) if args.binsz is not None else float(_get(cfg, "binsz", 0.125))

    labels = _get(cfg, "labels", None)
    if labels is not None:
        labels = [str(x) for x in labels]

    exclude_disk_in_plot = bool(_get(cfg, "exclude_disk_in_plot", False))

    os.makedirs(outdir, exist_ok=True)

    make_plots_from_mcmc(
        fig=fig,
        counts_path=str(counts),
        expo_path=str(expo),
        templates_dir=str(templates_dir),
        mcmc_dir=str(mcmc_dir),
        outdir=str(outdir),
        mcmc_stat=str(mcmc_stat),
        plot_style=str(plot_style),
        ext_mask_path=str(ext_mask) if ext_mask else None,
        roi_lon=float(roi_lon),
        roi_lat=float(roi_lat),
        disk_cut=float(disk_cut),
        binsz=float(binsz),
        labels=labels,
        exclude_disk_in_plot=bool(exclude_disk_in_plot),
    )

    print(f"✓ Done: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
