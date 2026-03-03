#!/usr/bin/env python3

import argparse
import os
import json
import subprocess
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.wcs import WCS

from totani_helpers.totani_io import (
    load_mask_any_shape,
    lonlat_grids,
    read_counts_and_ebounds,
)


def _safe_lognorm(img: np.ndarray):
    z = np.asarray(img, float)
    pos = z[np.isfinite(z) & (z > 0)]
    if pos.size == 0:
        return None
    vmin = float(np.nanpercentile(pos, 1))
    vmax = float(np.nanpercentile(pos, 99.9))
    if (not np.isfinite(vmin)) or vmin <= 0:
        vmin = float(np.nanmin(pos))
    if (not np.isfinite(vmax)) or vmax <= vmin:
        vmax = float(np.nanmax(pos))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= 0:
        return None
    return LogNorm(vmin=max(vmin, 1e-300), vmax=max(vmax, vmin * 1.0001))


def _read_cube(path: str, expected_shape: Tuple[int, int, int]):
    from astropy.io import fits

    with fits.open(path) as h:
        cube = np.asarray(h[0].data, float)
    if cube.shape != expected_shape:
        raise RuntimeError(f"{path} shape {cube.shape} != expected {expected_shape}")
    return cube


def _corr_matrix(templates_2d: List[np.ndarray], mask2d: np.ndarray) -> np.ndarray:
    vecs = []
    for t in templates_2d:
        v = np.asarray(t, float)[mask2d]
        v = v[np.isfinite(v)]
        if v.size == 0:
            vecs.append(np.zeros(int(np.sum(mask2d)), float))
            continue
        vv = np.asarray(t, float)[mask2d]
        mu = float(np.nanmean(vv))
        sig = float(np.nanstd(vv))
        if (not np.isfinite(sig)) or sig == 0:
            sig = 1.0
        vecs.append((vv - mu) / sig)
    X = np.vstack(vecs)
    return np.corrcoef(X)


def _maybe_git_hash(repo_dir: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        s = (r.stdout or "").strip()
        return s if s else None
    except Exception:
        return None


def _autodetect_ics_subcomponents(templates_dir: str) -> List[Tuple[str, str]]:
    """Return a list of (name, path) for any recognized ICS subcomponent templates.

    This is intentionally conservative: it only adds items if the file exists.
    """
    out: List[Tuple[str, str]] = []
    candidates = {
        "ics_opt": [
            "mu_ics_opt_counts.fits",
            "mu_ics_optical_counts.fits",
        ],
        "ics_ir": [
            "mu_ics_ir_counts.fits",
            "mu_ics_infrared_counts.fits",
        ],
        "ics_cmb": [
            "mu_ics_cmb_counts.fits",
        ],
    }

    for key, fnames in candidates.items():
        for fn in fnames:
            p = os.path.join(str(templates_dir), fn)
            if os.path.exists(p):
                out.append((key, p))
                break
    return out


def main():
    repo_dir = os.environ.get("REPO_PATH")
    if repo_dir is None:
        raise SystemExit("REPO_PATH not set (source Totani_paper_check/setup.sh)")

    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    templates_dir_default = os.path.join(data_dir, "processed", "templates")

    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=templates_dir_default)
    ap.add_argument("--ext-mask", default=os.path.join(templates_dir_default, "mask_extended_sources.fits"))

    ap.add_argument("--Egev", type=float, default=21.0)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0)

    ap.add_argument(
        "--halo-label",
        default="nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
    )

    ap.add_argument("--ics-opt", default=None)
    ap.add_argument("--ics-ir", default=None)
    ap.add_argument("--ics-cmb", default=None)

    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "plots_template_grid_21gev"))
    args = ap.parse_args()

    counts, hdr, _Emin, _Emax, Ectr_mev, _dE = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0
    k = int(np.argmin(np.abs(Ectr_gev - float(args.Egev))))

    lon2d, lat2d = lonlat_grids(wcs, ny, nx)
    roi2d = (np.abs(lon2d) <= float(args.roi_lon)) & (np.abs(lat2d) <= float(args.roi_lat))
    disk_keep = np.abs(lat2d) >= float(args.disk_cut)

    ext_keep3d = np.ones((nE, ny, nx), dtype=bool)
    if args.ext_mask and os.path.exists(str(args.ext_mask)):
        ext_keep3d = load_mask_any_shape(str(args.ext_mask), (nE, ny, nx)).astype(bool)

    mask2d = roi2d & disk_keep & ext_keep3d[k]

    items: List[Tuple[str, str]] = [
        ("gas", os.path.join(args.templates_dir, "mu_gas_counts.fits")),
        ("ps", os.path.join(args.templates_dir, "mu_ps_counts.fits")),
        ("ics", os.path.join(args.templates_dir, "mu_ics_counts.fits")),
        ("iso", os.path.join(args.templates_dir, "mu_iso_counts.fits")),
        ("loopA", os.path.join(args.templates_dir, "mu_loopA_counts.fits")),
        ("loopB", os.path.join(args.templates_dir, "mu_loopB_counts.fits")),
        ("loopI", os.path.join(args.templates_dir, "mu_loopI_counts.fits")),
        ("fb_pos", os.path.join(args.templates_dir, "mu_fb_pos_counts.fits")),
        ("fb_neg", os.path.join(args.templates_dir, "mu_fb_neg_counts.fits")),
        ("halo", os.path.join(args.templates_dir, f"mu_{str(args.halo_label)}_counts.fits")),
    ]

    if args.ics_opt:
        items.append(("ics_opt", str(args.ics_opt)))
    if args.ics_ir:
        items.append(("ics_ir", str(args.ics_ir)))
    if args.ics_cmb:
        items.append(("ics_cmb", str(args.ics_cmb)))

    # If not explicitly provided, attempt to autodetect common filenames.
    if (not args.ics_opt) and (not args.ics_ir) and (not args.ics_cmb):
        items.extend(_autodetect_ics_subcomponents(str(args.templates_dir)))

    names: List[str] = []
    maps: List[np.ndarray] = []

    for name, path in items:
        if not os.path.exists(path):
            print(f"[skip] missing: {name} -> {path}")
            continue
        cube = _read_cube(path, expected_shape=(nE, ny, nx))
        img = np.asarray(cube[k], float)
        img = np.where(mask2d, img, np.nan)
        names.append(str(name))
        maps.append(img)

    if not maps:
        raise SystemExit("No templates found to plot.")

    os.makedirs(args.outdir, exist_ok=True)

    # Reproducibility outputs: dump config + cache intermediates.
    cfg = {
        "counts": str(args.counts),
        "templates_dir": str(args.templates_dir),
        "ext_mask": str(args.ext_mask),
        "Egev_requested": float(args.Egev),
        "Egev_selected": float(Ectr_gev[k]),
        "k_selected": int(k),
        "roi_lon": float(args.roi_lon),
        "roi_lat": float(args.roi_lat),
        "disk_cut": float(args.disk_cut),
        "halo_label": str(args.halo_label),
        "mask2d_npix": int(np.sum(mask2d)),
        "inputs": [{"name": n, "path": p} for (n, p) in items],
        "git_hash": _maybe_git_hash(str(repo_dir)),
    }
    out_cfg = os.path.join(args.outdir, f"template_grid_E{Ectr_gev[k]:.2f}GeV_config.json")
    with open(out_cfg, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    n = len(maps)
    ncol = 3
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.7 * nrow), squeeze=False)
    for i, (name, img) in enumerate(zip(names, maps)):
        ax = axes[i // ncol][i % ncol]
        norm = _safe_lognorm(img)
        im = ax.imshow(img, origin="lower", cmap="magma", norm=norm)
        ax.set_title(f"{name}  k={k}  E={Ectr_gev[k]:.2f} GeV")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.tight_layout()
    out_png = os.path.join(args.outdir, f"template_grid_E{Ectr_gev[k]:.2f}GeV.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("✓ wrote", out_png)

    C = _corr_matrix(maps, mask2d=mask2d)
    fig = plt.figure(figsize=(0.55 * len(names) + 4.5, 0.55 * len(names) + 4.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(C, origin="lower", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title(f"ROI correlation matrix @ {Ectr_gev[k]:.2f} GeV")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr")
    fig.tight_layout()
    out_png = os.path.join(args.outdir, f"template_corr_E{Ectr_gev[k]:.2f}GeV.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("✓ wrote", out_png)

    out_cache = os.path.join(args.outdir, f"template_grid_E{Ectr_gev[k]:.2f}GeV_cache.npz")
    np.savez_compressed(
        out_cache,
        names=np.asarray(names, dtype=object),
        k=np.int32(k),
        Egev=np.float32(Ectr_gev[k]),
        mask2d=mask2d.astype(np.uint8),
        corr=C.astype(np.float32),
    )
    print("✓ wrote", out_cache)
    print("✓ wrote", out_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
