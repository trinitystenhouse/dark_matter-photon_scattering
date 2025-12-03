#!/usr/bin/env python3
"""
Extract photon spectrum from processed Fermi LAT data
Prepares data for Figure 6 recreation
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def extract_spectrum(fits_file, energy_bins=None):
    """
    Extract photon spectrum from FITS file.
    
    Parameters:
    -----------
    fits_file : str
        Path to processed FITS file
    energy_bins : array-like, optional
        Energy bin edges in GeV. If None, uses log-spaced bins from 50-500 GeV
    
    Returns:
    --------
    bin_centers : array
        Energy bin centers in GeV
    flux : array
        E^2 * dN/dE in GeV cm^-2 s^-1
    flux_err : array
        Uncertainty in flux
    """
    
    # Read FITS file
    print(f"Reading: {fits_file}")
    with fits.open(fits_file) as hdul:
        # Print file structure
        print("\nFITS structure:")
        hdul.info()
        
        # Get EVENTS extension
        events = hdul['EVENTS'].data
        
        print(f"\nTotal events: {len(events)}")
        print(f"Available columns: {events.columns.names}")
        
        # Extract energy (in MeV, convert to GeV)
        energy_mev = events['ENERGY']
        energy_gev = energy_mev / 1000.0
        
        print(f"\nEnergy range: {energy_gev.min():.1f} - {energy_gev.max():.1f} GeV")
        
        # Get time range for exposure calculation
        time = events['TIME']
        exposure_time = time.max() - time.min()  # seconds
        print(f"Time range: {exposure_time / (365.25 * 24 * 3600):.2f} years")
        
        # Get GTI (Good Time Intervals) for accurate exposure
        gti = hdul['GTI'].data
        total_gti = np.sum(gti['STOP'] - gti['START'])
        print(f"Total good time: {total_gti / (365.25 * 24 * 3600):.2f} years")
    
    # Define energy bins (log-spaced from 50 to 500 GeV)
    if energy_bins is None:
        n_bins = 100
        energy_bins = np.logspace(np.log10(50), np.log10(500), n_bins + 1)
    
    # Histogram the events
    counts, bin_edges = np.histogram(energy_gev, bins=energy_bins)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean
    bin_widths = np.diff(bin_edges)
    
    # Calculate differential flux: dN/dE
    # counts / (exposure_time * bin_width * effective_area)
    # For now, use simplified calculation (will need proper exposure map)
    
    # Assume effective area ~ 8000 cm^2 (typical for Fermi LAT at high energy)
    # This is a simplification - proper analysis needs gtexpcube2
    effective_area = 8000.0  # cm^2
    
    # dN/dE in units of photons / (GeV cm^2 s)
    dNdE = counts / (total_gti * bin_widths * effective_area)
    
    # E^2 * dN/dE in units of GeV / (cm^2 s)
    flux = bin_centers**2 * dNdE
    
    # Poisson errors
    flux_err = flux / np.sqrt(counts + 1)  # Add 1 to avoid division by zero
    
    return bin_centers, flux, flux_err, counts


def plot_spectrum(energy, flux, flux_err, output_file='spectrum.png'):
    """Plot the extracted spectrum."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points with error bars
    ax.errorbar(energy, flux, yerr=flux_err, fmt='o', 
                color='black', markersize=6, capsize=3,
                label='Fermi LAT data')
    
    ax.set_xlabel('Energy [GeV]', fontsize=14)
    ax.set_ylabel(r'$E^2 \, dN/dE$ [GeV cm$^{-2}$ s$^{-1}$]', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Spectrum plot saved: {output_file}")
    
    return fig, ax


def save_spectrum_data(energy, flux, flux_err, counts, output_file='spectrum_data.txt'):
    """Save spectrum to text file for further analysis."""
    
    header = "Energy [GeV]    E^2*dN/dE [GeV/cm^2/s]    Error    Counts"
    data = np.column_stack([energy, flux, flux_err, counts])
    
    np.savetxt(output_file, data, header=header, 
               fmt=['%.2f', '%.6e', '%.6e', '%d'])
    
    print(f"✓ Spectrum data saved: {output_file}")


def main():
    """Main execution."""
    
    fits_file = 'fermi_data/processed/GC_filtered_merged.fits'
    
    print("="*60)
    print("FERMI LAT SPECTRUM EXTRACTION")
    print("="*60)
    print()
    
    # Extract spectrum
    energy, flux, flux_err, counts = extract_spectrum(fits_file)
    
    print("\n" + "="*60)
    print("EXTRACTED SPECTRUM")
    print("="*60)
    print(f"\n{'Energy [GeV]':<15} {'Flux':<20} {'Error':<20} {'Counts':<10}")
    print("-"*70)
    for e, f, fe, c in zip(energy, flux, flux_err, counts):
        print(f"{e:<15.1f} {f:<20.6e} {fe:<20.6e} {c:<10.0f}")
    
    # Save data
    save_spectrum_data(energy, flux, flux_err, counts)
    
    # Plot spectrum
    plot_spectrum(energy, flux, flux_err)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Review spectrum_data.txt and spectrum.png
2. For proper analysis, you need to:
   - Generate exposure map: gtltcube + gtexpcube2
   - Calculate proper effective area vs energy
   - Apply instrument response functions
   
3. Compare with Figure 6 from the paper:
   - Fit power-law model (no DM)
   - Fit power-law + DM scattering model
   - Calculate residuals
   
4. Use recreate_fig6.py as template for fitting
    """)


if __name__ == "__main__":
    main()
