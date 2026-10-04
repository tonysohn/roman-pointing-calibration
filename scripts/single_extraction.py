#!/usr/bin/env python3
import argparse
import os

from roman_pointing import extract_wfi_sources


def main():
    parser = argparse.ArgumentParser(
        description="Run single extraction on a specific ASDF file for debugging."
    )
    parser.add_argument("filepath", help="Path to the ASDF file to process.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="gaussian",
        choices=["gaussian", "epsf"],
        help="Extraction strategy: 'gaussian' (fast) or 'epsf' (high-fidelity)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=".",
        help="Target directory to save the catalog and plot.",
    )
    args = parser.parse_args()

    print(f"Running single extraction on: {args.filepath}")
    print(f"Using extraction strategy: '{args.strategy}'")

    # Running directly without a try/except block to expose the full traceback
    catalog = extract_wfi_sources(
        asdf_filepath=args.filepath,
        centroid_method=args.strategy,
        save_diagnostic_plot=True,
        plot_outdir=args.target_dir,
    )

    if len(catalog) > 0:
        fmt_catalog = catalog.copy()
        for col in fmt_catalog.colnames:
            if fmt_catalog[col].dtype.kind in "fc":
                fmt_catalog[col].format = "%.4f"

        basename = os.path.basename(args.filepath)
        out_filename = os.path.join(args.target_dir, f"{basename}_catalog.ecsv")
        fmt_catalog.write(out_filename, format="ascii.ecsv", overwrite=True)

        print(f"Successfully exported {len(fmt_catalog)} sources to {out_filename}")
    else:
        print("Skipped: No valid sources extracted.")


if __name__ == "__main__":
    main()
