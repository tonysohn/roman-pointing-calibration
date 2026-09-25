#!/usr/bin/env python3
import argparse
import concurrent.futures
import glob
import os

from roman_pointing import extract_wfi_sources


def process_single_file(args):
    """
    Worker function to process a single ASDF file.
    Runs independently on a separate CPU core.
    """
    filepath, strategy, target_dir = args  # Unpack the arguments tuple directly

    basename = os.path.basename(filepath)
    parts = basename.upper().split("_")
    sca_name = next(
        (part for part in parts if part.startswith("WFI") and len(part) == 5), None
    )

    if not sca_name:
        return f"[!] Could not parse SCA name from filename: {basename}"

    dict_key = f"{sca_name}_FULL"

    try:
        catalog = extract_wfi_sources(
            asdf_filepath=filepath,
            centroid_method=strategy,
            save_diagnostic_plot=True,
            plot_outdir=target_dir,
        )

        if len(catalog) > 0:
            fmt_catalog = catalog.copy()
            for col in fmt_catalog.colnames:
                if fmt_catalog[col].dtype.kind in "fc":
                    fmt_catalog[col].format = "%.4f"

            out_filename = os.path.join(target_dir, f"{basename}_catalog.ecsv")
            fmt_catalog.write(out_filename, format="ascii.ecsv", overwrite=True)
            return f"[{dict_key}] Successfully exported {len(fmt_catalog)} sources to {out_filename}"
        else:
            return f"[{dict_key}] Skipped: No valid sources extracted."

    except Exception as e:
        return f"[{dict_key}] Error processing {basename}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract WFI sources from ASDF files using Multiprocessing."
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=".",
        help="Target directory containing the ASDF files (defaults to current directory)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="gaussian",
        choices=["gaussian", "epsf"],
        help="Extraction strategy: 'gaussian' (fast) or 'epsf' (high-fidelity)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of CPU cores to use. Defaults to all available cores.",
    )
    args = parser.parse_args()

    search_pattern = os.path.join(args.target_dir, "*_cal.asdf")
    input_files = sorted(glob.glob(search_pattern))

    if not input_files:
        print(f"No calibrated ASDF files found in '{args.target_dir}'.")
        return

    # Determine optimal worker count
    max_cores = os.cpu_count() or 1
    num_workers = args.workers if args.workers else min(len(input_files), max_cores)

    print(
        f"Found {len(input_files)} calibrated files in '{args.target_dir}' to process."
    )
    print(f"Using extraction strategy: '{args.strategy}'")
    print(f"Spinning up {num_workers} parallel workers...\n")

    # Package arguments as a list of tuples
    worker_args = [
        (filepath, args.strategy, args.target_dir) for filepath in input_files
    ]

    # Execute batch processing in parallel directly on the top-level function
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_single_file, worker_args)

        for output_message in results:
            print(output_message)

    print("\nBatch extraction complete.")


if __name__ == "__main__":
    main()
