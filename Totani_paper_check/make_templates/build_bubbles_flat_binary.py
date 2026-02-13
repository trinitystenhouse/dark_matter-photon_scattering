#!/usr/bin/env python3
"""build_bubbles_flat_binary.py

Build *truly flat* Fermi bubbles templates from a pure binary mask.

This script is intentionally independent of the SCA-derived bubbles builder.
It enforces the convention:
- spatial template is binary within each lobe (north/south) and zero elsewhere
- normalize spatial template so sum over ROI pixels = 1
- convert to dN/dE per sr via division by (omega * dE)
- build mu via multiplication by (expo * omega * dE)

That guarantees the reconstructed intensity
  I_rec = mu / (expo * omega * dE)
 is constant within the lobe mask (up to floating point noise).
"""

import argparse
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.path import Path

from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
)


def _write_primary_with_bunit(path, data, hdr_in, bunit):
    hdr = hdr_in.copy()
    hdr["BUNIT"] = str(bunit)
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=hdr).writeto(path, overwrite=True)


def _roi_box_mask(lon_deg, lat_deg, roi_lon, roi_lat):
    return (np.abs(lon_deg) <= float(roi_lon)) & (np.abs(lat_deg) <= float(roi_lat))


def _read_vertices_lonlat(path):
    pts = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 3:
        raise RuntimeError(f"Need >=3 vertices in {path}")
    return np.asarray(pts, dtype=float)


def _polygon_mask_from_lonlat_vertices(wcs, ny, nx, verts_lonlat_deg):
    lon = np.mod(verts_lonlat_deg[:, 0], 360.0)
    lat = verts_lonlat_deg[:, 1]
    x, y = wcs.world_to_pixel_values(lon, lat)
    poly = Path(np.vstack([x, y]).T)
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    return poly.contains_points(pts).reshape(ny, nx)


def _debug_template_stats(name, T, m):
    vals = np.asarray(T, dtype=float)[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print(f"[dbg] {name}: EMPTY")
        return
    med = float(np.median(vals))
    p16, p84 = np.percentile(vals, [16, 84])
    rel_std = float(np.std(vals) / med) if med != 0 else float("inf")
    print(f"[dbg] {name} spatial median={med:.6e}  16-84={p16:.6e},{p84:.6e}  rel_std/median={rel_std:.4e}")


def _debug_flatness_stats(tag, x, m):
    vals = np.asarray(x, dtype=float)[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print(f"[dbg] {tag}: EMPTY")
        return
    med = float(np.median(vals))
    p16, p84 = np.percentile(vals, [16, 84])
    rel_std = float(np.std(vals) / med) if med != 0 else float("inf")
    print(f"[dbg] {tag} median={med:.6e}  16-84={p16:.6e},{p84:.6e}  rel_std/median={rel_std:.4e}")


def _build_and_write_products(*, outdir, prefix, hdr, roi2d, mask2d, expo, omega, dE_mev, Ectr_mev):
    nE, ny, nx = expo.shape
    if mask2d.shape != (ny, nx):
        raise RuntimeError(f"mask2d shape {mask2d.shape} != {(ny, nx)}")

    m = np.asarray(mask2d, dtype=bool) & np.asarray(roi2d, dtype=bool)
    if not np.any(m):
        raise RuntimeError(f"Empty mask within ROI for prefix='{prefix}'")

    # Raw binary spatial indicator (for debugging only)
    fb_spatial = np.zeros((ny, nx), float)
    fb_spatial[m] = 1.0
    fb_spatial[~roi2d] = 0.0

    # Debug (requested): verify binary-ness within N/S before any weighting.
    yy, xx = np.mgrid[:ny, :nx]
    lon_deg, lat_deg = WCS(hdr).celestial.pixel_to_world_values(xx, yy)
    lat_deg = np.asarray(lat_deg, dtype=float)
    for tag, mm in [("N", m & (lat_deg > 0)), ("S", m & (lat_deg < 0))]:
        v = fb_spatial[mm]
        if v.size == 0:
            print(f"[FB:{prefix}:{tag}] fb_spatial: EMPTY")
            continue
        med = float(np.median(v))
        rel = float(np.std(v) / med) if med != 0 else float("inf")
        p16, p84 = np.percentile(v, [16, 84])
        print(
            f"[FB:{prefix}:{tag}] fb_spatial rel_std/median = {rel:.4e}  "
            f"16-84 {float(p16):.6g} {float(p84):.6g}"
        )

    # Solid-angle-weighted spatial template. This is the key step:
    # It ensures dnde = T/(omega*dE) is constant inside the mask.
    T = np.zeros((ny, nx), float)
    T[m] = omega[m]
    T[~roi2d] = 0.0

    # Normalize over ROI pixels (sum(T) over masked pixels = 1)
    s = float(np.nansum(T[m]))
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"Normalization failed for prefix='{prefix}' (sum={s})")
    T /= s

    # Debug: the omega-weighted T is not expected to be flat inside the lobe on a CAR grid
    # (omega varies with latitude). The key quantity is dnde (= I_rec) which should be flat.
    k_dbg = int(np.clip(nE // 2, 0, nE - 1))
    dnde_dbg = T / (omega * dE_mev[k_dbg])
    _debug_flatness_stats(f"{prefix}:dnde(k={k_dbg}):N", dnde_dbg, m & (lat_deg > 0))
    _debug_flatness_stats(f"{prefix}:dnde(k={k_dbg}):S", dnde_dbg, m & (lat_deg < 0))

    dnde = np.empty((nE, ny, nx), float)
    for k in range(nE):
        dnde[k] = T / (omega * dE_mev[k])

    e2dnde = dnde * (Ectr_mev[:, None, None] ** 2)
    mu_counts = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    _write_primary_with_bunit(os.path.join(outdir, f"{prefix}_mask.fits"), m.astype(np.int16), hdr, "dimensionless")
    _write_primary_with_bunit(os.path.join(outdir, f"{prefix}_template.fits"), T, hdr, "dimensionless")
    _write_primary_with_bunit(os.path.join(outdir, f"mu_{prefix}_counts.fits"), mu_counts, hdr, "counts")
    _write_primary_with_bunit(os.path.join(outdir, f"{prefix}_dnde.fits"), dnde, hdr, "ph cm-2 s-1 sr-1 MeV-1")
    _write_primary_with_bunit(os.path.join(outdir, f"{prefix}_E2dnde.fits"), e2dnde, hdr, "MeV cm-2 s-1 sr-1")

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
def main():
    default_counts = os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits")
    default_expo = os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits")
    default_outdir = os.path.join(DATA_DIR, "processed", "templates")
    here = os.path.dirname(__file__)
    verts_n = os.path.join(here, "bubble_vertices_north.txt")
    verts_s = os.path.join(here, "bubble_vertices_south.txt")

    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=default_counts)
    ap.add_argument("--expo", default=default_expo)
    ap.add_argument("--mask", default=None, help="Binary bubbles mask FITS (2D).")
    ap.add_argument("--verts-north", default=verts_n, help="North lobe polygon vertices file (lon lat per line, deg)")
    ap.add_argument("--verts-south", default=verts_s, help="South lobe polygon vertices file (lon lat per line, deg)")
    ap.add_argument("--outdir", default=default_outdir)
    ap.add_argument("--prefix", default="bubbles_flat_binary")

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--binsz", type=float, default=0.125)

    args = ap.parse_args()

    if not os.path.exists(str(args.counts)):
        raise SystemExit(f"Counts file not found: {args.counts}")
    if not os.path.exists(str(args.expo)):
        raise SystemExit(f"Exposure cube not found: {args.expo}")

    os.makedirs(args.outdir, exist_ok=True)

    counts, hdr, Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    wcs = WCS(hdr).celestial
    nE, ny, nx = counts.shape

    with fits.open(args.expo) as he:
        expo_raw = np.array(he[0].data, dtype=np.float64)
        E_expo_mev = read_expcube_energies_mev(he)
    expo = resample_exposure_logE_interp(expo_raw, E_expo_mev, Ectr_mev)

    omega = pixel_solid_angle_map(wcs, ny, nx, binsz_deg=float(args.binsz))

    # Build bubbles mask from either a FITS mask file or vertices.
    if args.mask is not None:
        mask_path = str(args.mask)
        if not os.path.exists(mask_path):
            raise SystemExit(
                "Bubbles mask file not found: "
                + mask_path
                + "\n(If you passed '$REPO_PATH/...' make sure REPO_PATH is exported in this shell.)"
            )

        fb_mask = fits.getdata(mask_path).astype(bool)
        if fb_mask.ndim == 3:
            fb_mask = fb_mask[0].astype(bool)
        if fb_mask.shape != (ny, nx):
            raise SystemExit(f"Mask shape {fb_mask.shape} != {(ny, nx)}")
    else:
        if (args.verts_north is None) or (args.verts_south is None):
            raise SystemExit("Pass either --mask or both --verts-north and --verts-south")
        if not os.path.exists(str(args.verts_north)):
            raise SystemExit(f"verts-north file not found: {args.verts_north}")
        if not os.path.exists(str(args.verts_south)):
            raise SystemExit(f"verts-south file not found: {args.verts_south}")

        verts_n = _read_vertices_lonlat(str(args.verts_north))
        verts_s = _read_vertices_lonlat(str(args.verts_south))
        mask_n_poly = _polygon_mask_from_lonlat_vertices(wcs, ny, nx, verts_n)
        mask_s_poly = _polygon_mask_from_lonlat_vertices(wcs, ny, nx, verts_s)
        fb_mask = mask_n_poly | mask_s_poly

    yy, xx = np.mgrid[:ny, :nx]
    lon_deg, lat_deg = wcs.pixel_to_world_values(xx, yy)
    lon_deg = ((np.asarray(lon_deg, dtype=float) + 180.0) % 360.0) - 180.0
    lat_deg = np.asarray(lat_deg, dtype=float)

    roi2d = _roi_box_mask(lon_deg, lat_deg, args.roi_lon, args.roi_lat)

    if args.mask is not None:
        mask_n = fb_mask & roi2d & (lat_deg > 0)
        mask_s = fb_mask & roi2d & (lat_deg < 0)
        print("[info] mask:", str(args.mask))
    else:
        mask_n = mask_n_poly & roi2d
        mask_s = mask_s_poly & roi2d
        print("[info] verts-north:", str(args.verts_north))
        print("[info] verts-south:", str(args.verts_south))

    print("[info] ROI: |l|<=", float(args.roi_lon), "|b|<=", float(args.roi_lat))
    print("[info] pixels in ROI:", int(np.sum(roi2d)))
    print("[info] pixels in FB mask (ROI):", int(np.sum(fb_mask & roi2d)))
    print("[info] pixels in FB North/South:", int(np.sum(mask_n)), int(np.sum(mask_s)))

    # Products:
    # - combined (for FB_FLAT use)
    # - north-only and south-only for diagnostics
    base = str(args.prefix)

    _build_and_write_products(
        outdir=str(args.outdir),
        prefix=base,
        hdr=hdr,
        roi2d=roi2d,
        mask2d=(mask_n | mask_s),
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
    )
    _build_and_write_products(
        outdir=str(args.outdir),
        prefix=f"{base}_N",
        hdr=hdr,
        roi2d=roi2d,
        mask2d=mask_n,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
    )
    _build_and_write_products(
        outdir=str(args.outdir),
        prefix=f"{base}_S",
        hdr=hdr,
        roi2d=roi2d,
        mask2d=mask_s,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
    )

    print("✓ Wrote templates to:")
    print(" ", str(args.outdir))
    print("✓ Prefix:")
    print(" ", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
