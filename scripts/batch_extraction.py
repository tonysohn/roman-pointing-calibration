import glob
import os

import numpy as np
from roman_pointing import extract_wfi_sources

# 1. Initialize the dictionary required by the pipeline
phot_catalogs = {}

# =========================================================================
# CONFIGURATION
# =========================================================================
# Choose your extraction strategy: 'gaussian' (fast) or 'epsf' (high-fidelity)
CENTROID_STRATEGY = "gaussian"
# =========================================================================

# 2. Locate all perturbed files in the directory
perturbed_files = glob.glob("*_perturbed_cal.asdf")
perturbed_files.sort()

print(f"Found {len(perturbed_files)} perturbed files to process.\n")
print(f"Using extraction strategy: '{CENTROID_STRATEGY}'\n")

# 3. Loop through and extract sources
for filepath in perturbed_files:
    basename = os.path.basename(filepath)

    # Extract the SCA name from the filename for the output ECSV naming
    parts = basename.upper().split("_")
    sca_name = next(
        (part for part in parts if part.startswith("WFI") and len(part) == 5), None
    )

    if sca_name:
        dict_key = f"{sca_name}_FULL"

        # Run the source extraction tool with the selected strategy
        catalog = extract_wfi_sources(
            asdf_filepath=filepath,
            centroid_method=CENTROID_STRATEGY,
            save_diagnostic_plot=True,
            plot_outdir="./",
        )

        # Store the resulting Astropy Table in the dictionary
        if len(catalog) > 0:
            phot_catalogs[dict_key] = catalog

            # --- FILE EXPORT FORMATTING ---
            # Create a copy so we don't truncate the pipeline's internal 64-bit precision
            fmt_catalog = catalog.copy()

            # Truncate all floating point numbers to 4 decimal places
            for col in fmt_catalog.colnames:
                if fmt_catalog[col].dtype.kind in "fc":  # if float or complex
                    fmt_catalog[col].format = "%.4f"

            # Write with fixed-width formatting for perfect column alignment
            out_filename = f"{basename}_catalog.ecsv"
            fmt_catalog.write(out_filename, format="ascii.ecsv", overwrite=True)
            print(f"  -> Exported formatted ECSV catalog: {out_filename}")
        else:
            print(f"Skipping {dict_key}: No valid sources extracted.")
    else:
        print(f"Could not parse SCA name from filename: {basename}")

print("\nBatch extraction complete.")
