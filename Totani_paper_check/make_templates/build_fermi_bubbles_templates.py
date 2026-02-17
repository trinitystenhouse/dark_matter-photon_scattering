#!/usr/bin/env python3
"""
Build Totani-style *flat* Fermi Bubbles (FB) template on the counts CCUBE grid.

Totani 2025 method (step 1 of bubble construction):
  - Assume a flat (isotropic) bubble template with sharp edges within a rough bubble area,
    and fit it with other components (including the disk |b|<10° at this stage). :contentReference[oaicite:7]{index=7}
This script ONLY builds the flat template products in the same conventions as your NFW builder:
  - fb_flat_dnde.fits      : dN/dE [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - fb_flat_E2dnde.fits    : E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
  - mu_fb_flat_counts.fits : expected counts per bin per pixel [counts]
with:
  mu = dnde * exposure * Omega_pix * dE

No unnecessary normalisations:
  - no pixel-sum normalisation
  - no unit-mean normalisation
  - dnde is exactly 1 inside the bubble mask and 0 outside (sharp edges).

You MUST supply a bubble mask FITS (0/1 map) defining the rough FB area.
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Optional dependency for WCS reprojection
try:
    from reproject import reproject_interp
except Exception as e:
    reproject_interp = None

from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
)

def load_and_reproject_mask(mask_fits, target_header, ny, nx):
    """
    Returns float mask on target (ny,nx), values in [0,1].
    Uses nearest-neighbour-like behaviour by thresholding after interpolation.
    """
    if reproject_interp is None:
        raise RuntimeError(
            "Missing dependency 'reproject'. Install with:\n"
            "  pip install reproject\n"
            "or conda install -c conda-forge reproject"
        )

    with fits.open(mask_fits) as hm:
        # allow either primary or first image extension
        hdu = hm[0]
        if hdu.data is None and len(hm) > 1:
            hdu = hm[1]
        mask_data = np.array(hdu.data, dtype=float)
        mask_wcs_full = WCS(hdu.header)

    # Reduce to 2D + celestial WCS if the input has extra (non-celestial) axes.
    # Common case: a 3D cube with shape (nE, ny, nx) or (ny, nx, nE).
    mask_wcs = mask_wcs_full.celestial
    if mask_data.ndim > 2:
        # Try to pick the first plane in a deterministic way.
        # If the last two axes match the target image, assume leading axes are non-spatial.
        if mask_data.shape[-2:] == (ny, nx):
            mask_data = mask_data.reshape((-1, ny, nx))[0]
        elif mask_data.shape[:2] == (ny, nx):
            mask_data = mask_data[:, :, 0]
        else:
            # Fallback: squeeze any singleton axes then take the first index until 2D.
            mask_data = np.squeeze(mask_data)
            while mask_data.ndim > 2:
                mask_data = mask_data[0]

    if mask_data.shape != (ny, nx):
        # Reproject onto CCUBE celestial grid
        reproj, _ = reproject_interp((mask_data, mask_wcs), target_header, shape_out=(ny, nx))
        reproj = np.nan_to_num(reproj, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # Already on target grid; just sanitize NaNs.
        reproj = np.nan_to_num(mask_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Make it sharp-edged again
    mask01 = (reproj >= 0.5).astype(np.float32)
    return mask01

def main():
    ap = argparse.ArgumentParser()
    repo_dir = os.environ.get("REPO_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    default_counts = os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits")
    default_expo = os.path.join(data_dir, "processed", "expcube_1000to1000000.fits")
    default_bubble_mask = os.path.join(data_dir, "processed", "templates", "bubbles_flat_binary_mask.fits")
    default_outdir = os.path.join(data_dir, "processed", "templates")

    ap.add_argument("--counts", default=default_counts, help="Counts CCUBE (authoritative WCS + EBOUNDS)")
    ap.add_argument("--expo", default=default_expo, help="Exposure cube (expcube)")
    ap.add_argument("--bubble-mask", default=default_bubble_mask, help="FITS mask defining FB rough area (0/1 map)")
    ap.add_argument("--outdir", default=default_outdir)

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)

    ap.add_argument(
        "--expo-sampling",
        choices=["center", "edge-mean"],
        default="center",
        help="Exposure per bin: at CCUBE bin center, or mean of exposure at (Emin,Emax).",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    out_dnde   = os.path.join(args.outdir, "fb_flat_dnde.fits")
    out_e2dnde = os.path.join(args.outdir, "fb_flat_E2dnde.fits")
    out_mu     = os.path.join(args.outdir, "mu_fb_flat_counts.fits")

    # --- Read CCUBE header + EBOUNDS (authoritative energy binning) ---
    with fits.open(args.counts) as hc:
        hdr = hc[0].header
        wcs_cel = WCS(hdr).celestial
        ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]

        if "EBOUNDS" not in hc:
            raise RuntimeError("Counts CCUBE missing EBOUNDS extension.")
        eb = hc["EBOUNDS"].data
        eb_hdu = hc["EBOUNDS"].copy()

    Emin_mev = np.array(eb["E_MIN"], dtype=float) / 1000.0
    Emax_mev = np.array(eb["E_MAX"], dtype=float) / 1000.0
    dE_mev   = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    nE       = int(Ectr_mev.size)

    # --- Read exposure cube + energies, resample to CCUBE binning ---
    with fits.open(args.expo) as he:
        expo_raw = np.array(he[0].data, dtype=np.float64)
        E_expo_mev = read_expcube_energies_mev(he)

    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure spatial shape {expo_raw.shape[1:]} != counts {(ny, nx)}")

    if args.expo_sampling == "center":
        expo = resample_exposure_logE_interp(expo_raw, E_expo_mev, Ectr_mev)
        expo_comment = "Exposure at CCUBE Ectr (logE interp, clamped)."
    else:
        expo_min = resample_exposure_logE_interp(expo_raw, E_expo_mev, Emin_mev)
        expo_max = resample_exposure_logE_interp(expo_raw, E_expo_mev, Emax_mev)
        expo = 0.5 * (expo_min + expo_max)
        expo_comment = "Exposure = 0.5*(expo(Emin)+expo(Emax)) with logE interp (clamped)."

    if expo.shape[0] != nE:
        raise RuntimeError(f"Resampled exposure has {expo.shape[0]} planes, expected {nE}")

    # --- Build ROI mask (Totani ROI is |l|<=60, 10<=|b|<=60 for halo search,
    # but the *flat bubble construction step* fits the whole ROI including disk.
    # Here we just enforce your analysis map ROI box; disk cuts happen in fitter masks.) :contentReference[oaicite:8]{index=8}
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs_cel.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0
    roi_box = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)

    # --- Load and reproject bubble region mask onto CCUBE grid ---
    # Treat the provided mask as the “rough area” with sharp edges. :contentReference[oaicite:9]{index=9}
    fb_mask01 = load_and_reproject_mask(args.bubble_mask, wcs_cel.to_header(), ny, nx)
    fb_mask01[~roi_box] = 0.0

    # --- Solid angle map ---
    omega = pixel_solid_angle_map(wcs_cel, ny, nx, args.binsz)

    # --- Define flat FB intensity template (no extra normalisations) ---
    # dnde = 1 inside bubble, 0 outside (sharp edges)
    fb_dnde = np.broadcast_to(fb_mask01[None, :, :], (nE, ny, nx)).astype(np.float64)

    # E^2 dnde
    fb_E2dnde = fb_dnde * (Ectr_mev[:, None, None] ** 2)

    # Counts-shape template for fitting: 0/1 indicator repeated over energy.
    # (Keep physical dnde/E2dnde for flux conversions/plotting; NNLS wants a shape-only component.)
    mu_fb = np.broadcast_to(fb_mask01[None, :, :], (nE, ny, nx)).astype(np.float64)

    # -------------------------
    # Write outputs (include EBOUNDS)
    # -------------------------
    def _write_cube(path, data, bunit, comments):
        phdu = fits.PrimaryHDU(data.astype(np.float32), header=hdr)
        phdu.header["BUNIT"] = bunit
        for c in comments:
            phdu.header["COMMENT"] = c
        fits.HDUList([phdu, eb_hdu]).writeto(path, overwrite=True)

    comments = [
        "Totani-style flat Fermi Bubbles template: constant inside bubble mask, zero outside (sharp edges).",
        "No additional normalisations applied (no unit-sum, no unit-mean).",
        "dnde = 1 ph cm^-2 s^-1 sr^-1 MeV^-1 inside mask; 0 outside; coefficient sets intensity per bin.",
        expo_comment,
        "mu_fb_flat_counts is a shape-only 0/1 cube for NNLS fitting (coefficient absorbs normalization).",
    ]

    _write_cube(out_dnde, fb_dnde, "ph cm-2 s-1 sr-1 MeV-1", comments)
    _write_cube(out_e2dnde, fb_E2dnde, "MeV cm-2 s-1 sr-1", ["E^2 dN/dE version of flat FB template."] + comments)
    _write_cube(out_mu, mu_fb, "counts", ["Expected counts for flat FB template."] + comments)

    print("✓ wrote", out_dnde)
    print("✓ wrote", out_e2dnde)
    print("✓ wrote", out_mu)
    print("mu_fb total counts:", float(np.nansum(mu_fb)))
    print("mu_fb per-bin:", np.nansum(mu_fb, axis=(1, 2)))

if __name__ == "__main__":
    main()
