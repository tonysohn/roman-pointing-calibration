#!/usr/bin/env python3
import argparse
import concurrent.futures
import dataclasses
import glob
import os

import numpy as np
from astropy.table import Table

from roman_pointing.crossmatch import (
    RobustMatchConfig,
    apply_affine_transformation,
    find_detector_consensus_matches,
    project_reference_catalog_to_detector,
    project_with_calibrated_siaf,
)


def process_single_detector(
    asdf_path,
    ref_catalog,
    output_dir,
    config,
    global_seed,
    yaml_file=None,
    dV2=0.0,
    dV3=0.0,
):
    """Isolated worker function executing a seeded fine-match on a dedicated CPU core."""
    basename = os.path.basename(asdf_path)
    base_prefix = basename.replace("_cal.asdf", "")

    # Use base_prefix to avoid duplicating the suffix
    catalog_path = f"{base_prefix}_cal.asdf_catalog.ecsv"
    if not os.path.exists(catalog_path):
        return None, 0, None, f"Missing {catalog_path}"

    parts = basename.upper().split("_")
    det_name = next((p for p in parts if p.startswith("WFI") and len(p) == 5), None)
    if not det_name:
        return None, 0, None, "Could not parse detector name"

    obs_cat = Table.read(catalog_path, format="ascii.ecsv")
    if "flux" in obs_cat.colnames:
        obs_cat.sort("flux", reverse=True)

    obs_pixels = np.column_stack(
        [
            obs_cat["x"] + 1 if "x" in obs_cat.colnames else obs_cat["x_centroid"] + 1,
            obs_cat["y"] + 1 if "y" in obs_cat.colnames else obs_cat["y_centroid"] + 1,
        ]
    )
    try:
        if yaml_file:
            sub_cat, pred_pixels = project_with_calibrated_siaf(
                asdf_path, ref_catalog, yaml_file, config, dV2, dV3
            )
        else:
            sub_cat, pred_pixels = project_reference_catalog_to_detector(
                asdf_path, ref_catalog, config
            )

        # 2. SEEDED MATCH: Shrink the search window to 400 pixels around the global seed.
        # This isolates the true peak from background noise, allowing the sharp 4-pixel bin to work everywhere.
        seeded_config = dataclasses.replace(config, window_padding_pix=400.0)

        cat_idxs, obs_idxs, best_affine = find_detector_consensus_matches(
            pred_pixels,
            obs_pixels,
            seeded_config.primary_bin_pix,
            seed=global_seed,
            config=seeded_config,
        )

        matched_x = obs_pixels[obs_idxs, 0]
        matched_y = obs_pixels[obs_idxs, 1]
        matched_ra = np.asarray(sub_cat["ra_epoch"])[cat_idxs]
        matched_dec = np.asarray(sub_cat["dec_epoch"])[cat_idxs]
        matched_mag = np.asarray(sub_cat["phot_g_mean_mag"])[cat_idxs]

        count = len(matched_x)

        # 3. Save matching table
        out_table = Table(
            [matched_x, matched_y, matched_ra, matched_dec, matched_mag],
            names=("x", "y", "ra_epoch", "dec_epoch", "phot_g_mean_mag"),
        )
        ecsv_filename = f"{base_prefix}_cal.asdf_xmatch.ecsv"
        ecsv_path = os.path.join(output_dir, ecsv_filename)
        out_table.write(ecsv_path, format="ascii.ecsv", overwrite=True)

        # 4. Generate DS9 Region File (.reg)
        reg_filename = f"{base_prefix}_cal.asdf_xmatch.reg"
        reg_path = os.path.join(output_dir, reg_filename)

        detector_center = np.array([4088 / 2, 4088 / 2])
        matched_pred_pixels = apply_affine_transformation(
            pred_pixels[cat_idxs], best_affine, detector_center, config
        )

        with open(reg_path, "w") as f_reg:
            f_reg.write("# Region file format: DS9 version 4.1\n")
            f_reg.write(
                'global dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
            )
            f_reg.write("image\n")

            for x, y in zip(matched_x, matched_y):
                f_reg.write(f"circle({x:.2f},{y:.2f},6) # color=green\n")

            for rx, ry in matched_pred_pixels:
                f_reg.write(f"circle({rx:.2f},{ry:.2f},3) # color=red\n")

        return det_name, count, ecsv_filename, None

    except Exception as err:
        return det_name, 0, None, str(err)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Robust Cross-Matcher for Roman WFI")
    parser.add_argument(
        "--yaml", type=str, default=None, help="Path to calibrated YAML"
    )
    args = parser.parse_args()

    print("--- Robust Native gWCS Cross-Match Pipeline (Parallelized) ---")
    if args.yaml:
        if not os.path.exists(args.yaml):
            raise FileNotFoundError(
                f"CRITICAL: Bootstrap YAML not found at '{args.yaml}'."
            )
        print(
            f"\nBOOTSTRAP MODE ENABLED: Projecting with calibrated models from {args.yaml}"
        )
    else:
        print("\nBLIND MODE ENABLED: Projecting with native ASDF gWCS")

    gaia_file = "gaia_dr3_commissioning_field_wide.ecsv"
    print(f"Loading reference catalog: {gaia_file}")
    ref_catalog = Table.read(gaia_file, format="ascii.ecsv")

    asdf_files = sorted(glob.glob("*_f146_cal.asdf"))
    if not asdf_files:
        print("Error: No calibrated ASDF files found.")
        return

    output_dir = "crossmatch_results"
    os.makedirs(output_dir, exist_ok=True)
    config = RobustMatchConfig()

    # --- UNIFIED SCOUT PASS ---
    scout_file = next((f for f in asdf_files if "WFI01" in f.upper()), asdf_files[0])
    scout_name = os.path.basename(scout_file)
    print(
        f"\nExecuting native scout match on {scout_name} to lock global attitude offset..."
    )

    scout_cat_path = f"{scout_name.replace('_cal.asdf', '')}_cal.asdf_catalog.ecsv"
    scout_obs = Table.read(scout_cat_path, format="ascii.ecsv")
    if "flux" in scout_obs.colnames:
        scout_obs.sort("flux", reverse=True)

    scout_pixels = np.column_stack(
        [
            scout_obs["x"] + 1
            if "x" in scout_obs.colnames
            else scout_obs["x_centroid"] + 1,
            scout_obs["y"] + 1
            if "y" in scout_obs.colnames
            else scout_obs["y_centroid"] + 1,
        ]
    )

    # Always use the native ASDF projection for the scout to avoid warping the scout itself
    _, scout_pred_pixels = project_reference_catalog_to_detector(
        scout_file, ref_catalog, config
    )
    _, _, scout_affine = find_detector_consensus_matches(
        scout_pred_pixels,
        scout_pixels,
        config.primary_bin_pix,
        seed=None,
        config=config,
    )
    dx_pix, dy_pix = scout_affine[2]
    print(f"Scout pixel shift locked: dx={dx_pix:.1f}, dy={dy_pix:.1f} pixels.")

    # Convert the raw pixel shift into a physical V2/V3 sky correction factor
    import pysiaf

    rsiaf = pysiaf.Siaf("Roman")
    scout_aper = rsiaf["WFI01_FULL"]
    v2_cen, v3_cen = scout_aper.sci_to_tel(2044.5, 2044.5)
    v2_shift, v3_shift = scout_aper.sci_to_tel(2044.5 + dx_pix, 2044.5 + dy_pix)

    dV2 = v2_shift - v2_cen
    dV3 = v3_shift - v3_cen
    print(f"Global V2/V3 correction locked: dV2={dV2:.3f}, dV3={dV3:.3f} arcsec.")

    # In Bootstrap Mode, the physical dV2/dV3 correction pre-aligns the catalog.
    # Therefore, the pixel seed passed to the workers must be 0.0 to prevent double-shifting.
    global_seed = np.array([0.0, 0.0]) if args.yaml else np.array([dx_pix, dy_pix])
    passed_dV2 = dV2 if args.yaml else 0.0
    passed_dV3 = dV3 if args.yaml else 0.0

    print(f"\nDispatching {len(asdf_files)} SCAs to worker pool (4 at a time)")
    print(f"============================================================")
    print(f"{'Detector':<12} | {'Matches':<8} | {'Output Files':<30}")
    print(f"------------------------------------------------------------")

    total_global_matches = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                process_single_detector,
                asdf_path,
                ref_catalog,
                output_dir,
                config,
                global_seed,
                args.yaml,
                passed_dV2,  # Added physical V2 correction
                passed_dV3,  # Added physical V3 correction
            ): asdf_path
            for asdf_path in asdf_files
        }

        for future in concurrent.futures.as_completed(futures):
            det_name, count, ecsv_filename, error = future.result()

            if not det_name:
                continue

            if error:
                print(f"{det_name:<12} | ERROR    | Failed: {error}")
            else:
                print(f"{det_name:<12} | {count:<8} | {ecsv_filename}")
                total_global_matches += count

    print(f"============================================================")
    print(f"Total Cross-Matched Pairs Generated: {total_global_matches}")
    print(f"Results, tables, and region files successfully written to '{output_dir}/'.")


if __name__ == "__main__":
    main()
