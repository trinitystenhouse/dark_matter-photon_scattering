#!/usr/bin/env python3
import numpy as np
from astropy.io import fits

TXT_IN   = "../fermi_data/processed/energy_edges_mev.txt"
FITS_OUT = "../fermi_data/processed/energy_ebounds.fits"

edges = np.loadtxt(TXT_IN, dtype=float)  # MeV
if edges.ndim != 1 or edges.size < 2:
    raise ValueError("Need >=2 edges (one per line) in MeV.")

e_min = edges[:-1].astype(np.float32)
e_max = edges[1:].astype(np.float32)
n = len(e_min)

cols = fits.ColDefs([
    fits.Column(name="CHANNEL", format="1J", array=np.arange(1, n+1, dtype=np.int32)),
    fits.Column(name="E_MIN",   format="1E", unit="MeV", array=e_min),
    fits.Column(name="E_MAX",   format="1E", unit="MeV", array=e_max),
])

# IMPORTANT: gtbin (your build) expects an HDU named ENERGYBINS
energybins = fits.BinTableHDU.from_columns(cols)
energybins.name = "ENERGYBINS"
energybins.header["EXTNAME"] = "ENERGYBINS"

hdul = fits.HDUList([fits.PrimaryHDU(), energybins])
hdul.writeto(FITS_OUT, overwrite=True)

print(f"Wrote {FITS_OUT} with {n} bins")
print("First/last edges (MeV):", edges[0], edges[-1])
