import glob
import os

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
            centroid_method=CENTROID_STRATEGY,  # Pass the toggle here!
            save_diagnostic_plot=True,
            plot_outdir="./",
        )

        # Store the resulting Astropy Table in the dictionary
        if len(catalog) > 0:
            phot_catalogs[dict_key] = catalog
            catalog.write(
                f"{basename}_catalog.ecsv", format="ascii.ecsv", overwrite=True
            )
        else:
            print(f"Skipping {dict_key}: No valid sources extracted.")
    else:
        print(f"Could not parse SCA name from filename: {basename}")

print(
    f"\nExtraction complete. Successfully built phot_catalogs with {len(phot_catalogs)} SCAs."
)
