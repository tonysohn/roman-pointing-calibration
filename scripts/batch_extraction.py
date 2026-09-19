#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np

from roman_pointing import extract_wfi_sources


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract WFI sources from ASDF files."
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
    args = parser.parse_args()

    phot_catalogs = {}

    # Locate all calibrated files dynamically in the target directory
    search_pattern = os.path.join(args.target_dir, "*_cal.asdf")
    input_files = sorted(glob.glob(search_pattern))

    if not input_files:
        print(f"No calibrated ASDF files found in '{args.target_dir}'.")
        return

    print(
        f"Found {len(input_files)} calibrated files in '{args.target_dir}' to process."
    )
    print(f"Using extraction strategy: '{args.strategy}'\n")

    for filepath in input_files:
        basename = os.path.basename(filepath)

        parts = basename.upper().split("_")
        sca_name = next(
            (part for part in parts if part.startswith("WFI") and len(part) == 5), None
        )

        if sca_name:
            dict_key = f"{sca_name}_FULL"

            catalog = extract_wfi_sources(
                asdf_filepath=filepath,
                centroid_method=args.strategy,
                save_diagnostic_plot=True,
                plot_outdir=args.target_dir,
            )

            if len(catalog) > 0:
                phot_catalogs[dict_key] = catalog

                fmt_catalog = catalog.copy()
                for col in fmt_catalog.colnames:
                    if fmt_catalog[col].dtype.kind in "fc":
                        fmt_catalog[col].format = "%.4f"

                # Save output ECSV in the same directory as the target ASDFs
                out_filename = os.path.join(args.target_dir, f"{basename}_catalog.ecsv")
                fmt_catalog.write(out_filename, format="ascii.ecsv", overwrite=True)
                print(f"  -> Exported formatted ECSV catalog: {out_filename}")
            else:
                print(f"Skipping {dict_key}: No valid sources extracted.")
        else:
            print(f"Could not parse SCA name from filename: {basename}")

    print("\nBatch extraction complete.")


if __name__ == "__main__":
    main()
