import numpy as np
from astropy.io import fits

n_bins = 13
E0_center_gev = 1.51
dlog10E = 0.227637288133

i = np.arange(n_bins)
Ecen_gev = E0_center_gev * 10**(i * dlog10E)

Elow_gev  = Ecen_gev * 10**(-dlog10E / 2.0)
Ehigh_gev = Ecen_gev * 10**(+dlog10E / 2.0)

# write in MeV because Fermitools energy inputs are typically MeV
with open("totani_13bins_gev_edges.txt", "w") as f:
    for lo, hi in zip(Elow_gev * 1e3, Ehigh_gev * 1e3):
        f.write(f"{lo:.6f} {hi:.6f}\n")

# Also write an OGIP-style EBOUNDS FITS so Fermitools can use it for binning.
# This avoids gtbindef (which depends on PFILES being configured).
emin_mev = (Elow_gev * 1e3).astype(np.float64)
emax_mev = (Ehigh_gev * 1e3).astype(np.float64)

for j in range(n_bins - 1):
    if not (emax_mev[j] < emin_mev[j + 1]):
        emax_mev[j] = np.nextafter(emin_mev[j + 1], -np.inf)
    if not (emin_mev[j] < emax_mev[j]):
        emin_mev[j] = np.nextafter(emax_mev[j], -np.inf)

channel = np.arange(1, n_bins + 1, dtype=np.int32)

cols = fits.ColDefs(
    [
        fits.Column(name="CHANNEL", format="J", array=channel),
        fits.Column(name="E_MIN", format="D", unit="MeV", array=emin_mev),
        fits.Column(name="E_MAX", format="D", unit="MeV", array=emax_mev),
    ]
)

eb_hdu = fits.BinTableHDU.from_columns(cols, name="EBOUNDS")
eb_hdu.header["EXTNAME"] = "EBOUNDS"
eb_hdu.header["HDUCLASS"] = "OGIP"
eb_hdu.header["HDUCLAS1"] = "RESPONSE"
eb_hdu.header["HDUCLAS2"] = "EBOUNDS"
eb_hdu.header["HDUVERS"] = "1.2.1"
eb_hdu.header["CHANTYPE"] = "PHA"
eb_hdu.header["DETCHANS"] = int(n_bins)

energybins_hdu = fits.BinTableHDU.from_columns(cols, name="ENERGYBINS")
energybins_hdu.header["EXTNAME"] = "ENERGYBINS"
energybins_hdu.header["DETCHANS"] = int(n_bins)

prim = fits.PrimaryHDU()
hdul = fits.HDUList([prim, eb_hdu, energybins_hdu])
hdul.writeto("totani_13bins.fits", overwrite=True)

print("first low edge [GeV] =", Elow_gev[0])
print("last high edge [GeV] =", Ehigh_gev[-1])
print("wrote totani_13bins_gev_edges.txt")
print("wrote totani_13bins.fits")