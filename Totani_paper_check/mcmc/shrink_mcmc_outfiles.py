#!/usr/bin/env python3
"""
Shrink existing MCMC .npz files by recompressing and optionally dropping chain/logprob.

This script:
1. Recompresses .npz files using np.savez_compressed
2. Downcasts float64 arrays to float32
3. Optionally drops chain and logprob arrays (which are large)
4. Creates backups before modifying files (optional)

Usage:
    python shrink_mcmc_outfiles.py mcmc_results/mcmc_results_k*.npz --drop-chain --drop-logprob
    python shrink_mcmc_outfiles.py mcmc_results/ --backup-suffix .bak
"""

import argparse
import glob
import os
import shutil
import numpy as np


def shrink_npz(
    input_path,
    output_path=None,
    drop_chain=False,
    drop_logprob=False,
    backup_suffix=None,
):
    """
    Shrink an MCMC .npz file by recompressing and optionally dropping arrays.
    
    Args:
        input_path: Path to input .npz file
        output_path: Path to output file (if None, overwrites input)
        drop_chain: If True, don't save 'chain' array
        drop_logprob: If True, don't save 'logprob' array
        backup_suffix: If provided, create backup with this suffix before overwriting
    
    Returns:
        (old_size_mb, new_size_mb): File sizes before and after
    """
    if output_path is None:
        output_path = input_path
        
    # Get original size
    old_size = os.path.getsize(input_path)
    
    # Create backup if requested and overwriting
    if backup_suffix and (output_path == input_path):
        backup_path = input_path + backup_suffix
        shutil.copy2(input_path, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # Load existing data
    data = np.load(input_path, allow_pickle=True)
    
    # Build new save dict
    save_dict = {}
    for key in data.files:
        # Skip chain/logprob if requested
        if drop_chain and key == "chain":
            print(f"  Dropping 'chain' array")
            continue
        if drop_logprob and key == "logprob":
            print(f"  Dropping 'logprob' array")
            continue
        
        arr = data[key]
        
        # Downcast numeric arrays to float32
        if isinstance(arr, np.ndarray) and arr.dtype in (np.float64, np.float32):
            save_dict[key] = arr.astype(np.float32)
        else:
            save_dict[key] = arr
    
    # Save compressed
    np.savez_compressed(output_path, **save_dict)
    
    # Get new size
    new_size = os.path.getsize(output_path)
    
    return old_size / 1e6, new_size / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Input .npz files or directories containing them")
    ap.add_argument("--drop-chain", action="store_true", help="Drop 'chain' array to save space")
    ap.add_argument("--drop-logprob", action="store_true", help="Drop 'logprob' array to save space")
    ap.add_argument("--backup-suffix", default=None, help="Create backup with this suffix (e.g., .bak)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    args = ap.parse_args()
    
    # Collect all .npz files
    npz_files = []
    for path in args.paths:
        if os.path.isdir(path):
            npz_files.extend(glob.glob(os.path.join(path, "*.npz")))
        elif os.path.isfile(path):
            npz_files.append(path)
        else:
            # Try as glob pattern
            npz_files.extend(glob.glob(path))
    
    npz_files = sorted(set(npz_files))
    
    if not npz_files:
        print("No .npz files found!")
        return
    
    print(f"Found {len(npz_files)} .npz file(s)")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - no files will be modified]\n")
    
    total_old_mb = 0
    total_new_mb = 0
    
    for npz_file in npz_files:
        print(f"\nProcessing: {npz_file}")
        
        if args.dry_run:
            old_size = os.path.getsize(npz_file) / 1e6
            print(f"  Current size: {old_size:.2f} MB")
            print(f"  Would compress and downcast to float32")
            if args.drop_chain:
                print(f"  Would drop 'chain' array")
            if args.drop_logprob:
                print(f"  Would drop 'logprob' array")
            continue
        
        try:
            old_mb, new_mb = shrink_npz(
                npz_file,
                drop_chain=args.drop_chain,
                drop_logprob=args.drop_logprob,
                backup_suffix=args.backup_suffix,
            )
            
            total_old_mb += old_mb
            total_new_mb += new_mb
            
            reduction = 100 * (1 - new_mb / old_mb) if old_mb > 0 else 0
            print(f"  {old_mb:.2f} MB → {new_mb:.2f} MB ({reduction:.1f}% reduction)")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"Total: {total_old_mb:.2f} MB → {total_new_mb:.2f} MB")
        if total_old_mb > 0:
            total_reduction = 100 * (1 - total_new_mb / total_old_mb)
            print(f"Overall reduction: {total_reduction:.1f}%")
            print(f"Space saved: {total_old_mb - total_new_mb:.2f} MB")


if __name__ == "__main__":
    main()
