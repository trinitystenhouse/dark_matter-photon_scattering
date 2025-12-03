#!/usr/bin/env python3
"""
Batch processing script for Fermi LAT data
Processes photon files in chunks with proper cuts applied
"""

import os
import subprocess
import glob

# Configuration
DATA_DIR = "fermi_data"
PROCESSED_DIR = "fermi_data/processed"
SPACECRAFT_FILE = None  # Will be set automatically

# Analysis parameters for Galactic Center
RA = 266.415  # Galactic Center in J2000
DEC = -29.0061
ROI_RADIUS = 10.0  # degrees
EMIN = 50000  # MeV (50 GeV)
EMAX = 500000  # MeV (500 GeV)
ZMAX = 90.0  # Maximum zenith angle (degrees)
EVENT_CLASS = 128  # SOURCE class (Pass 8)
EVENT_TYPE = 3  # Front+back, all PSF and energy subclasses

# Good time interval filter
GTI_FILTER = "(DATA_QUAL>0)&&(LAT_CONFIG==1)"


def setup_directories():
    """Create necessary directories."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"✓ Directories created: {DATA_DIR}, {PROCESSED_DIR}")


def find_spacecraft_file():
    """Find the spacecraft file (SC*.fits)."""
    global SPACECRAFT_FILE
    sc_files = glob.glob(f"{DATA_DIR}/*_SC*.fits")
    if not sc_files:
        raise FileNotFoundError(
            f"No spacecraft file found in {DATA_DIR}. "
            "Download it from the Fermi LAT query."
        )
    SPACECRAFT_FILE = sc_files[0]
    print(f"✓ Found spacecraft file: {SPACECRAFT_FILE}")
    return SPACECRAFT_FILE


def get_photon_files():
    """Get list of photon files (PH*.fits)."""
    ph_files = sorted(glob.glob(f"{DATA_DIR}/*_PH*.fits"))
    if not ph_files:
        raise FileNotFoundError(
            f"No photon files found in {DATA_DIR}. "
            "Download them from the Fermi LAT query."
        )
    print(f"✓ Found {len(ph_files)} photon file(s)")
    return ph_files


def create_file_list(ph_files, output_file="events_list.txt"):
    """Create a text file listing all photon files."""
    list_path = os.path.join(DATA_DIR, output_file)
    with open(list_path, 'w') as f:
        for ph_file in ph_files:
            f.write(f"{ph_file}\n")
    print(f"✓ Created file list: {list_path}")
    return list_path


def run_gtselect(input_file, output_file, batch_num=None):
    """
    Run gtselect to apply event selection cuts.
    
    Cuts applied:
    - Time range (from input file)
    - Energy range: 50-500 GeV
    - ROI: 10 degrees around Galactic Center
    - Zenith angle: < 90 degrees
    - Event class: SOURCE (128)
    - Event type: 3 (front+back, all PSF/energy)
    """
    label = f" (batch {batch_num})" if batch_num else ""
    print(f"\n{'='*60}")
    print(f"Running gtselect{label}...")
    print(f"{'='*60}")
    
    cmd = [
        "gtselect",
        f"infile={input_file}",
        f"outfile={output_file}",
        f"ra={RA}",
        f"dec={DEC}",
        f"rad={ROI_RADIUS}",
        f"tmin=INDEF",  # Use time range from file
        f"tmax=INDEF",
        f"emin={EMIN}",
        f"emax={EMAX}",
        f"zmax={ZMAX}",
        f"evclass={EVENT_CLASS}",
        f"evtype={EVENT_TYPE}",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode != 0:
        print(f"ERROR in gtselect:")
        # Decode with error handling
        try:
            stderr = result.stderr.decode('utf-8', errors='replace')
            print(stderr)
        except Exception:
            print(result.stderr)
        return False
    
    print(f"✓ gtselect completed: {output_file}")
    return True


def run_gtmktime(input_file, output_file, scfile, batch_num=None):
    """
    Run gtmktime to apply good time interval cuts.
    
    Cuts applied:
    - DATA_QUAL > 0 (good data quality)
    - LAT_CONFIG == 1 (normal science operations)
    - ROI-based zenith angle cut
    """
    label = f" (batch {batch_num})" if batch_num else ""
    print(f"\n{'='*60}")
    print(f"Running gtmktime{label}...")
    print(f"{'='*60}")
    
    cmd = [
        "gtmktime",
        f"scfile={scfile}",
        f"evfile={input_file}",
        f"outfile={output_file}",
        "roicut=yes",  # Apply ROI-based zenith cut
        f"filter={GTI_FILTER}",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode != 0:
        print(f"ERROR in gtmktime:")
        # Decode with error handling
        try:
            stderr = result.stderr.decode('utf-8', errors='replace')
            print(stderr)
        except Exception:
            print(result.stderr)
        return False
    
    print(f"✓ gtmktime completed: {output_file}")
    return True


def process_single_file(ph_file, scfile, batch_num):
    """Process a single photon file through the full pipeline."""
    basename = os.path.basename(ph_file).replace('.fits', '')
    
    # Step 1: gtselect
    selected_file = os.path.join(PROCESSED_DIR, f"{basename}_selected.fits")
    if not run_gtselect(ph_file, selected_file, batch_num):
        return None
    
    # Step 2: gtmktime
    final_file = os.path.join(PROCESSED_DIR, f"{basename}_filtered.fits")
    if not run_gtmktime(selected_file, final_file, scfile, batch_num):
        return None
    
    # Clean up intermediate file
    os.remove(selected_file)
    print(f"✓ Cleaned up intermediate file: {selected_file}")
    
    return final_file


def process_all_files(ph_files, scfile):
    """Process all photon files in batches."""
    processed_files = []
    
    print(f"\n{'='*60}")
    print(f"Processing {len(ph_files)} photon file(s)")
    print(f"{'='*60}\n")
    
    for i, ph_file in enumerate(ph_files, 1):
        print(f"\n>>> Processing file {i}/{len(ph_files)}: {os.path.basename(ph_file)}")
        
        final_file = process_single_file(ph_file, scfile, i)
        if final_file:
            processed_files.append(final_file)
        else:
            print(f"⚠ Failed to process: {ph_file}")
    
    return processed_files


def merge_processed_files(processed_files, output_file="GC_filtered_merged.fits"):
    """Merge all processed files into a single file."""
    if len(processed_files) == 1:
        print(f"\n✓ Only one file processed, no merge needed")
        return processed_files[0]
    
    print(f"\n{'='*60}")
    print(f"Merging {len(processed_files)} processed files...")
    print(f"{'='*60}")
    
    # Create file list for merging
    merge_list = os.path.join(PROCESSED_DIR, "merge_list.txt")
    with open(merge_list, 'w') as f:
        for pf in processed_files:
            f.write(f"{pf}\n")
    
    output_path = os.path.join(PROCESSED_DIR, output_file)
    
    # Use gtselect with @filelist to merge
    cmd = [
        "gtselect",
        f"infile=@{merge_list}",
        f"outfile={output_path}",
        f"ra={RA}",
        f"dec={DEC}",
        f"rad={ROI_RADIUS}",
        "tmin=INDEF",
        "tmax=INDEF",
        f"emin={EMIN}",
        f"emax={EMAX}",
        f"zmax={ZMAX}",
        f"evclass={EVENT_CLASS}",
        f"evtype={EVENT_TYPE}",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in merge:")
        print(result.stderr)
        return None
    
    print(f"✓ Merged file created: {output_path}")
    return output_path


def print_summary(processed_files, merged_file):
    """Print processing summary."""
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"\n✓ Processed {len(processed_files)} file(s)")
    print(f"✓ Final merged file: {merged_file}")
    print(f"\nCuts applied:")
    print(f"  - ROI: {ROI_RADIUS}° around (RA={RA}, Dec={DEC})")
    print(f"  - Energy: {EMIN/1000}-{EMAX/1000} GeV")
    print(f"  - Zenith angle: < {ZMAX}°")
    print(f"  - Event class: {EVENT_CLASS} (SOURCE)")
    print(f"  - Event type: {EVENT_TYPE}")
    print(f"  - GTI filter: {GTI_FILTER}")
    print(f"\nNext steps:")
    print(f"  1. Generate exposure map with gtltcube")
    print(f"  2. Extract spectrum for analysis")
    print(f"  3. Run recreate_fig6.py with real data")
    print(f"\n{'='*60}\n")


def main():
    """Main processing pipeline."""
    print("\n" + "="*60)
    print("FERMI LAT BATCH PROCESSING PIPELINE")
    print("="*60 + "\n")
    
    try:
        # Setup
        setup_directories()
        scfile = find_spacecraft_file()
        ph_files = get_photon_files()
        
        # Process files
        processed_files = process_all_files(ph_files, scfile)
        
        if not processed_files:
            print("\n❌ No files were successfully processed!")
            return 1
        
        # Merge if multiple files
        if len(processed_files) > 1:
            merged_file = merge_processed_files(processed_files)
        else:
            merged_file = processed_files[0]
        
        # Summary
        print_summary(processed_files, merged_file)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure you have:")
        print("  1. Downloaded photon files (*_PH*.fits)")
        print("  2. Downloaded spacecraft file (*_SC*.fits)")
        print(f"  3. Placed them in: {DATA_DIR}/")
        return 1
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
