#!/usr/bin/env python3
import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import asdf
import matplotlib.pyplot as plt
from astropy.table import Table

# Import the core matching functions from your crossmatch.py
from roman_pointing.crossmatch import (
    RobustMatchConfig,
    apply_affine_transformation,
    find_detector_consensus_matches,
    project_catalog,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run robust crossmatch on a single SCA for debugging."
    )
    parser.add_argument(
        "--asdf_file", required=True, help="Path to the Level 2 ASDF image."
    )
    parser.add_argument(
        "--extracted_catalog", required=True, help="Path to the extracted ECSV catalog."
    )
    parser.add_argument(
        "--reference_catalog",
        required=True,
        help="Path to the reference Gaia ECSV catalog.",
    )
    parser.add_argument(
        "--siaf", type=str, default=None, help="Optional YAML/XML SIAF to apply."
    )
    parser.add_argument(
        "--gwcs",
        action="store_true",
        help="Use gWCS embedded in ASDF instead of PySIAF",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="crossmatch_results",
        help="Directory to save the matched output.",
    )

    args = parser.parse_args()

    basename = os.path.basename(args.asdf_file)
    det_name = next(
        (p for p in basename.upper().split("_") if p.startswith("WFI") and len(p) == 5),
        "UNKNOWN",
    )

    print(f"\n[SINGLE CROSSMATCH]")
    print(f"Detector  : {det_name}")
    print(f"ASDF File : {basename}")
    print(f"Extracted : {os.path.basename(args.extracted_catalog)}")
    print(f"Reference : {os.path.basename(args.reference_catalog)}")

    if args.gwcs:
        print(
            "\nGWCS MODE ENABLED: Bypassing PySIAF. Projecting catalogs using embedded Level-2 ASDF gWCS."
        )
        if args.siaf:
            print("  [WARNING] --siaf flag ignored because --gwcs is active.")
    elif args.siaf:
        print(
            f"\nBOOTSTRAP MODE ENABLED: Projecting with calibrated models from {args.siaf}"
        )

    os.makedirs(args.target_dir, exist_ok=True)

    # 1. Load the Catalogs
    print("\nLoading catalogs...")
    ref_cat = Table.read(args.reference_catalog, format="ascii.ecsv")
    obs_cat = Table.read(args.extracted_catalog, format="ascii.ecsv")

    if "flux" in obs_cat.colnames:
        obs_cat.sort("flux", reverse=True)

    # Ensure X/Y are arrays for the KDTree (Handle Photutils 3.0 column names safely)
    x_col = "x" if "x" in obs_cat.colnames else "x_centroid"
    y_col = "y" if "y" in obs_cat.colnames else "y_centroid"

    # Add +1 to convert from 0-based extraction pixels to 1-based SIAF/DS9 pixels
    obs_coords = np.column_stack((obs_cat[x_col] + 1, obs_cat[y_col] + 1))

    # 2. Project Gaia to the focal plane
    print("Projecting reference catalog to detector space...")
    config = RobustMatchConfig()
    projected_cat, pred_coords = project_catalog(
        args.asdf_file,
        ref_cat,
        config=config,
        custom_siaf_filepath=args.siaf,
        use_gwcs=args.gwcs,
    )

    if len(pred_coords) < config.min_total_matches:
        print(
            f"FAIL: Only {len(pred_coords)} reference stars projected onto detector. Cannot match."
        )
        return

    # 3. Execute 2D Histogram Voting and Affine Optimization (No try/except to expose tracebacks)
    print("Executing 2D histogram voting and robust matching...")
    p_idx, o_idx, best_affine = find_detector_consensus_matches(
        pred_coords, obs_coords, bin_width=config.primary_bin_pix, config=config
    )

    # 4. Save Matched ECSV Catalog
    matched_x = obs_coords[o_idx][:, 0]
    matched_y = obs_coords[o_idx][:, 1]
    matched_ra = np.asarray(projected_cat["ra_epoch"])[p_idx]
    matched_dec = np.asarray(projected_cat["dec_epoch"])[p_idx]
    matched_mag = np.asarray(projected_cat["phot_g_mean_mag"])[p_idx]

    out_table = Table(
        [matched_x, matched_y, matched_ra, matched_dec, matched_mag],
        names=("x", "y", "ra_epoch", "dec_epoch", "phot_g_mean_mag"),
    )

    ecsv_file = os.path.join(args.target_dir, f"{basename}_xmatch.ecsv")
    out_table.write(ecsv_file, format="ascii.ecsv", overwrite=True)

    # 5. Generate Diagnostic PNG
    print("Generating diagnostic overlay plot...")
    png_file = os.path.join(args.target_dir, f"{basename}_matches.png")
    with asdf.open(args.asdf_file, lazy_load=True) as f:
        data_small = f["roman"]["data"][::8, ::8].astype(np.float32)
        ny, nx = f["roman"]["data"].shape

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    lo, hi = np.nanpercentile(data_small, [5, 99.5])
    ax.imshow(
        data_small,
        origin="lower",
        cmap="gray",
        vmin=lo,
        vmax=hi,
        extent=(1, nx, 1, ny),
    )
    ax.scatter(matched_x, matched_y, s=20, facecolors="none", edgecolors="cyan", lw=0.5)
    ax.set(title=f"{det_name}: {len(matched_x)} Gaia matches", xlabel="x", ylabel="y")
    fig.savefig(png_file, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 6. Generate DS9 Region File
    print("Writing DS9 region file...")
    reg_file = os.path.join(args.target_dir, f"{basename}_xmatch.reg")
    detector_center = np.array([4088 / 2, 4088 / 2])

    # Calculate exactly where the Gaia stars (red) fell relative to the extracted stars (green)
    matched_pred_pixels = apply_affine_transformation(
        pred_coords[p_idx], best_affine, detector_center, config
    )

    with open(reg_file, "w") as f_reg:
        f_reg.write("# Region file format: DS9 version 4.1\n")
        f_reg.write(
            'global dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
        )
        f_reg.write("image\n")

        # Plot observed image stars in green
        for x, y in zip(matched_x, matched_y):
            f_reg.write(f"circle({x:.2f},{y:.2f},6) # color=green\n")

        # Plot transformed catalog stars in red
        for rx, ry in matched_pred_pixels:
            f_reg.write(f"circle({rx:.2f},{ry:.2f},3) # color=red\n")

    print(f"\nSUCCESS: Matched {len(matched_x)} sources.")
    print(f"Saved matched catalog to : {ecsv_file}")
    print(f"Saved diagnostic image to: {png_file}")
    print(f"Saved DS9 region file to : {reg_file}\n")


if __name__ == "__main__":
    main()
