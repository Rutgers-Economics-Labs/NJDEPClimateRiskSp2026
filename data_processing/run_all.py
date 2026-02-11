"""
run_all.py
----------
Master pipeline that runs all data processing scripts in order
and prints a summary of all cleaned files produced.
"""

import os
import sys
import time

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process_boundaries import process_boundaries
from process_census import process_census
from process_finance import process_finance
from process_climate import process_climate
from process_fema_crs import process_fema_crs


def get_file_size_mb(filepath):
    """Get file size in MB."""
    size = os.path.getsize(filepath)
    return size / (1024 * 1024)


def list_output_files(output_dir):
    """Recursively list all files in the output directory."""
    files = []
    for root, dirs, filenames in os.walk(output_dir):
        for f in filenames:
            if not f.startswith("."):
                files.append(os.path.join(root, f))
    return sorted(files)


def main():
    start = time.time()
    output_dir = os.path.join(PROJECT_ROOT, "data", "data_cleaned")

    print("╔" + "═" * 58 + "╗")
    print("║  NJ MUNICIPAL BOND RESILIENCE — DATA PROCESSING PIPELINE  ║")
    print("╚" + "═" * 58 + "╝")
    print()

    all_outputs = []

    # 1. Boundaries (fast, needed for spatial joins later)
    try:
        all_outputs.extend(process_boundaries())
    except Exception as e:
        print(f"\n❌ Boundaries processing failed: {e}")

    print()

    # 2. Census data
    try:
        all_outputs.extend(process_census())
    except Exception as e:
        print(f"\n❌ Census processing failed: {e}")

    print()

    # 3. Financial data
    try:
        all_outputs.extend(process_finance())
    except Exception as e:
        print(f"\n❌ Finance processing failed: {e}")

    try:
        all_outputs.extend(process_fema_crs())
    except Exception as e:
        print(f"\n❌ FEMA CRS processing failed: {e}")

    print()

    # 4. Climate / SLR data (slowest — large GDB files)
    try:
        all_outputs.extend(process_climate())
    except Exception as e:
        print(f"\n❌ Climate processing failed: {e}")

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - start
    all_files = list_output_files(output_dir)

    print()
    print("╔" + "═" * 58 + "╗")
    print("║  PIPELINE SUMMARY                                         ║")
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Output directory: {os.path.relpath(output_dir, PROJECT_ROOT)}/")
    print(f"  Total files created: {len(all_files)}")
    print()

    if all_files:
        print("  Files in data_cleaned/:")
        for f in all_files:
            size = get_file_size_mb(f)
            rel = os.path.relpath(f, PROJECT_ROOT)
            print(f"    {rel:<50} {size:>8.2f} MB")

    print()
    print("  Done! ✓")


if __name__ == "__main__":
    main()
