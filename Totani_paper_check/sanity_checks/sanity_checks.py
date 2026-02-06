#!/usr/bin/env python3
"""
Sanity checks for Totani-style Fermi LAT cube products.

This script answers: "Is the dataset sane?"
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

# -------------------------
# Paths
# -------------------------
REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani", "processed")

COUNTS_CCUBE = os.path.join(DATA_DIR, "counts_ccube_1000to1000000.fits")
EXPCUBE      = os.path.join(DATA_DIR, "expcube_1000to1000000.fits")
COUNTS_CMAP  = os.path.join(DATA_DIR, "counts_cmap_1000to1000000.fits")

OUTDIR = "sanity_plots"
os.makedirs(OUTDIR, exist_ok=True)


def load_image_hdu(path, prefer_extname=None):
    """
    Load an image HDU from a FITS file robustly.
    If prefer_extname is given and exists, use it; otherwise pick the first image HDU with data.
    Returns (data, header, hdu_index, hdu_name).
    """
    with fits.open(path) as hdul:
        # optional preferred extname
        if prefer_extname is not None and prefer_extname in hdul:
            h = hdul[prefer_extname]
            if h.data is None:
                raise RuntimeError(f"{path}: preferred HDU '{prefer_extname}' has no data")
            return np.asarray(h.data), h.header, hdul.index_of(prefer_extname), prefer_extname

        # otherwise pick first ImageHDU/PrimaryHDU that has 2D/3D data
        for i, h in enumerate(hdul):
            if h.data is None:
                continue
            if getattr(h, "is_image", False) or isinstance(h, (fits.PrimaryHDU, fits.ImageHDU)):
                arr = np.asarray(h.data)
                if arr.ndim in (2, 3):
                    name = h.name if hasattr(h, "name") else f"HDU{i}"
                    return arr, h.header, i, name

        raise RuntimeError(f"{path}: no 2D/3D image HDU found")


# -------------------------
# Load counts (should be (E, Y, X))
# -------------------------
counts = fits.getdata(COUNTS_CCUBE).astype(float)
hdr_counts = fits.getheader(COUNTS_CCUBE)

if counts.ndim != 3:
    raise RuntimeError(f"Counts cube should be 3D (nE,ny,nx), got shape {counts.shape}")

nE, ny, nx = counts.shape

# -------------------------
# Load exposure robustly
# -------------------------
# Print exposure HDU inventory (super useful)
print("\n=== EXPCUBE HDU LIST ===")
with fits.open(EXPCUBE) as hdul:
    hdul.info()

expo, hdr_expo, expo_hdu_idx, expo_hdu_name = load_image_hdu(EXPCUBE)
expo = expo.astype(float)

print("\n=== COUNTS CCUBE ===")
print("Shape:", counts.shape)
print("Total counts:", int(np.nansum(counts)))
print("Min / Max:", np.nanmin(counts), np.nanmax(counts))
if np.nansum(counts) == 0:
    raise RuntimeError("❌ Counts cube is EMPTY")

print("\nCounts per energy bin:")
for i in range(nE):
    print(f"  bin {i:02d}: {int(np.nansum(counts[i]))}")

print("\n=== EXPOSURE CUBE ===")
print(f"Loaded exposure from HDU {expo_hdu_idx} ('{expo_hdu_name}')")
print("Shape:", expo.shape)
print("Finite fraction:", np.isfinite(expo).mean())
print("Min / Max:", np.nanmin(expo), np.nanmax(expo))

# -------------------------
# Energy bin sanity
# -------------------------
with fits.open(COUNTS_CCUBE) as hdul:
    if "EBOUNDS" not in hdul:
        raise RuntimeError("Counts cube missing EBOUNDS extension")
    eb = hdul["EBOUNDS"].data
    emin = eb["E_MIN"].astype(float)  # MeV
    emax = eb["E_MAX"].astype(float)  # MeV
    ectr = np.sqrt(emin * emax) / 1000.0  # GeV

print("\n=== ENERGY BINS ===")
print("N bins:", len(ectr))
print("GeV bin centers:", np.round(ectr, 3))

# -------------------------
# CMAP quick-look (integrated counts)
# -------------------------
if os.path.exists(COUNTS_CMAP):
    cmap = fits.getdata(COUNTS_CMAP).astype(float)

    plt.figure(figsize=(6, 5))
    plt.imshow(np.log10(np.maximum(cmap, 1)), origin="lower", cmap="inferno")
    plt.colorbar(label=r"$\log_{10}(\mathrm{counts})$")
    plt.title("Integrated counts (CMAP)")
    plt.xlabel("pixel x")
    plt.ylabel("pixel y")
    plt.tight_layout()
    out = os.path.join(OUTDIR, "counts_cmap_log.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("\n✓ Wrote:", out)

# -------------------------
# Per-bin sky structure check
# -------------------------
idxs = [0, nE // 2, nE - 1]

# Determine how to index exposure
expo_is_3d = (expo.ndim == 3)
if expo.ndim == 3 and expo.shape[0] != nE:
    print(f"\n⚠️ Exposure has {expo.shape[0]} energy planes but counts has {nE}.")
    print("   This script will still plot, but you should resample exposure energies for real analysis.")

for k in idxs:
    c = counts[k]

    if expo_is_3d:
        kk = min(k, expo.shape[0] - 1)
        a = expo[kk]
    else:
        a = expo  # 2D exposure map

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(np.log10(np.maximum(c, 1)), origin="lower", cmap="inferno")
    axes[0].set_title(f"Counts (E~{ectr[k]:.2g} GeV)")

    axes[1].imshow(np.log10(np.maximum(a, 1e-30)), origin="lower", cmap="viridis")
    axes[1].set_title("Exposure" + (" (2D)" if expo.ndim == 2 else f" (bin {min(k, expo.shape[0]-1)})"))

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(OUTDIR, f"bin_{k:02d}_counts_exposure.png")
    plt.savefig(out, dpi=150)
    plt.close()

    print("✓ Wrote:", out)

# -------------------------
# Global consistency ratios
# -------------------------
print("\n=== GLOBAL CONSISTENCY ===")
counts_nonzero_frac = (counts > 0).mean()
expo_nonzero_frac = (expo > 0).mean()

print("Counts nonzero fraction:", counts_nonzero_frac)
print("Exposure nonzero fraction:", expo_nonzero_frac)

print("\nAll sanity checks finished.")
print("Next step: flux maps + |b|<10° mask + Fig.1 panels.")
