#!/usr/bin/env python3
"""
run_calibration.py

Master Commissioning Pipeline for the Roman Space Telescope pointing calibration.
Executes two sequential tasks:
  1. WFI Macroscopic Alignment: Solves the geometric layout of the 18 SCAs.
  2. FGS Boresight Calibration: Derives the updated Body-to-FGS quaternion
     using Wahba's problem and the WFI cross-matched star catalogs.
"""

import glob
import os
import warnings

import numpy as np
import pysiaf
import roman_datamodels as rdm
from astropy.table import Table
from scipy.spatial.transform import Rotation as R

# Import the core modules from the pipeline
from roman_pointing import (
    align_wfi,
    apply_dva_scale_to_catalog,
    calibrate_roman_fgs_alignment,
    export_alignment_to_yaml,
    fetch_local_commissioning_gaia,
)


def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    # Toggle to enable/disable Differential Velocity Aberration correction
    apply_dva = True

    # We use (0,0,0) as the default to test the pipeline's blind recovery
    manual_offsets = {"d_ra_arcsec": 0.0, "d_dec_arcsec": 0.0, "d_pa_arcsec": 0.0}

    # =========================================================================
    # 1. DATA INGEST & INITIALIZATION
    # =========================================================================
    print("--- 1. DATA INGEST ---")
    roman_siaf = pysiaf.Siaf("Roman")

    # Reconstruct the phot_catalogs dictionary from the ECSV files
    ecsv_files = sorted(glob.glob("*_perturbed_cal.asdf_catalog.ecsv"))
    print(f"Found {len(ecsv_files)} extracted catalogs to align.")

    phot_catalogs = {}
    for filepath in ecsv_files:
        basename = os.path.basename(filepath)
        # Parse out the WFI chip ID (e.g., 'WFI01')
        parts = basename.upper().split("_")
        sca_name = next(
            (part for part in parts if part.startswith("WFI") and len(part) == 5), None
        )

        if sca_name:
            dict_key = f"{sca_name}_FULL"
            phot_catalogs[dict_key] = Table.read(filepath, format="ascii.ecsv")
        else:
            print(f"Warning: Could not parse SCA name from {basename}")

    if not phot_catalogs:
        print("Error: No valid catalogs loaded into memory. Exiting.")
        return

    # Extract nominal pointing metadata to anchor the reference catalog query
    original_asdfs = glob.glob("*_perturbed_cal.asdf")
    if not original_asdfs:
        print("Error: No original ASDF files found. Cannot extract pointing metadata.")
        return

    original_asdf = original_asdfs[0]
    print(f"\nExtracting reference observation metadata from: {original_asdf}")

    with rdm.open(original_asdf) as f:
        # 1. Pointing Meta
        ra_v1 = f.meta.pointing.ra_v1
        dec_v1 = f.meta.pointing.dec_v1
        pa_v3 = f.meta.pointing.pa_v3

        # 2. DVA Meta
        try:
            dva_scale = f.meta.velocity_aberration.scale_factor
            dva_ra_ref = f.meta.velocity_aberration.ra_reference
            dva_dec_ref = f.meta.velocity_aberration.dec_reference
            has_dva_meta = True
        except AttributeError:
            has_dva_meta = False
            print(
                "Warning: DVA metadata not found in ASDF. DVA correction will be skipped."
            )

        # 3. Spacecraft Velocity Meta (Moved inside the context manager!)
        try:
            v_x = f.meta.ephemeris.velocity_x
            v_y = f.meta.ephemeris.velocity_y
            v_z = f.meta.ephemeris.velocity_z
            velocity_kms = np.array([v_x, v_y, v_z])
        except AttributeError:
            print("Warning: Velocity missing from ASDF meta. Using hardcoded fallback.")
            velocity_kms = np.array([-4.33382, 26.99558, 11.70196])

        # 4. Observation Date
        try:
            obs_date_str = f.meta.observation.date_start
        except AttributeError:
            obs_date_str = "2026-09-20T00:00:00"

    pointing_info = {"RA_V1": ra_v1, "DEC_V1": dec_v1, "PA_V3": pa_v3}
    print(
        f"Telemetry Pointing: RA={ra_v1:.5f} deg, Dec={dec_v1:.5f} deg, PA={pa_v3:.5f} deg"
    )

    # ---------------------------------------------------------
    # FLIGHT TELEMETRY INGESTION
    # ---------------------------------------------------------
    # Telemetry Quaternion (SCF_AC_SDR_QBJ -> Body to ECI)
    acs_telemetry_qbj = np.array([0.70515723, 0.08269192, -0.68259625, -0.17314064])

    # Nominal BAM Telemetry (SCF_AC_FGS_TBL_Qb -> FGS to Body)
    q_b2fgs_nominal = np.array(
        [
            -0.1859673417539929,
            +0.6837984564491885,
            -0.1800546332580956,
            +0.6822141509826322,
        ]
    )
    # ---------------------------------------------------------

    # Load the Gaia DR3 catalog
    try:
        ref_catalog = fetch_local_commissioning_gaia(
            obs_date_str, local_csv_path="gaia_dr3_commissioning_field.ecsv"
        )

        # --- APPLY DIFFERENTIAL VELOCITY ABERRATION ---
        if apply_dva and has_dva_meta:
            print("  -> Applying Differential Velocity Aberration (DVA) to Gaia...")
            app_ra, app_dec = apply_dva_scale_to_catalog(
                np.asarray(ref_catalog["ra_epoch"]),
                np.asarray(ref_catalog["dec_epoch"]),
                dva_ra_ref,
                dva_dec_ref,
                dva_scale,
            )
            # Overwrite the catalog columns so align_wfi sees the Apparent Sky
            ref_catalog["ra_epoch"] = app_ra
            ref_catalog["dec_epoch"] = app_dec
        # ----------------------------------------------

    except Exception as e:
        print(f"Error loading local Gaia catalog: {e}")
        return

    # =========================================================================
    # --- FAULT-TOLERANCE TEST: SIMULATE DEAD SCAs ---
    # =========================================================================
    scas_to_kill = ["WFI05_FULL", "WFI12_FULL", "WFI17_FULL"]
    print("\n--- INJECTING HARDWARE FAILURES ---")
    for bad_sca in scas_to_kill:
        if bad_sca in phot_catalogs:
            print(
                f"Sabotaging {bad_sca}: Truncating to 3 stars to force alignment failure."
            )
            # Keep only the first 3 stars (Astropy Table slicing)
            phot_catalogs[bad_sca] = phot_catalogs[bad_sca][:3]
    # =========================================================================

    # =========================================================================
    # 2. WFI MACROSCOPIC ALIGNMENT (Local Geometry)
    # =========================================================================
    print("\n--- 2. RUNNING WFI ALIGNMENT ---")

    calibrated_siaf_params, attitude_results, matched_pairs_log = align_wfi(
        phot_catalogs=phot_catalogs,
        ref_catalog=ref_catalog,
        pointing_info=pointing_info,
        user_offsets=manual_offsets,
        max_iterations=5,
        debug=False,  # Set to True for verbose histogram/solver output
    )

    # Export to SIAF YAML
    output_yaml = export_alignment_to_yaml(
        calibrated_siaf_params, output_prefix="calibrated_roman_siaf"
    )

    # Calculate Recovered Deltas
    mean_dec_rad = np.deg2rad(attitude_results["DEC_V1"])
    d_ra_arcsec = (attitude_results["RA_V1"] - ra_v1) * np.cos(mean_dec_rad) * 3600.0
    d_dec_arcsec = (attitude_results["DEC_V1"] - dec_v1) * 3600.0
    d_pa_arcsec = (attitude_results["PA_V3"] - pa_v3) * 3600.0

    print("\n========================================================")
    print("           MEASURED GLOBAL OFFSETS (DELTAS)       ")
    print("========================================================")
    print(f"Δ RA  (V1): {d_ra_arcsec:8.2f} arcsec")
    print(f"Δ Dec (V1): {d_dec_arcsec:8.2f} arcsec")
    print(f"Δ PA  (V3): {d_pa_arcsec:8.2f} arcsec")

    # =========================================================================
    # 3. FGS BORESIGHT CALIBRATION (Spacecraft Geometry)
    # =========================================================================
    print("\n--- 3. RUNNING FGS BORESIGHT CALIBRATION ---")

    # Format the WFI cross-matched stars for the Wahba FGS solver
    # (Assuming matched_pairs_log format: [SCA, X, Y, RA, Dec, Flux, Mag, V2, V3, ResV2, ResV3])
    ref_stars_radec = np.array([[row[3], row[4]] for row in matched_pairs_log])
    measured_v2_v3 = np.array([[row[7], row[8]] for row in matched_pairs_log])

    print(f"Feeding {len(ref_stars_radec)} cross-matched stars into Wahba's Problem...")

    try:
        # Execute the vectorized Boresight solver using raw telemetry
        q_b2fgs_calibrated = calibrate_roman_fgs_alignment(
            reference_stars_radec=ref_stars_radec,
            measured_v2_v3=measured_v2_v3,  # Passed perfectly as Telescope V2/V3
            q_eci2b=acs_telemetry_qbj,
            v_sc_eci_kms=velocity_kms,
            wfi_cen_aper=roman_siaf["WFI_CEN"],
            q_b2fgs_old=q_b2fgs_nominal,
        )
        print("\n========================================================")
        print("           FGS BORESIGHT CALIBRATION RESULTS             ")
        print("========================================================")
        print(f"Updated BAM Telemetry (SCF_AC_FGS_TBL_Qb):")
        print(
            f"[{q_b2fgs_calibrated[0]:.9f}, {q_b2fgs_calibrated[1]:.9f}, {q_b2fgs_calibrated[2]:.9f}, {q_b2fgs_calibrated[3]:.9f}]"
        )

        # 1. Calculate the rotation required to align Old FGS to New FGS
        q_nom = R.from_quat(q_b2fgs_nominal)
        q_cal = R.from_quat(q_b2fgs_calibrated)
        delta_q = q_cal * q_nom.inv()

        # 2. Project the Boresight Vector (V1 axis: [1, 0, 0])
        # This tells us exactly how much the boresight (V1) moved on the sky
        v1_nominal = np.array([1, 0, 0])
        v1_calibrated = delta_q.apply(v1_nominal)

        # Calculate the angular separation between the old and new boresight
        # Use dot product for small angles: theta = arccos(u.v)
        cos_theta = np.clip(np.dot(v1_nominal, v1_calibrated), -1.0, 1.0)
        boresight_shift_arcsec = np.degrees(np.arccos(cos_theta)) * 3600.0

        print(f"Total Boresight Shift (V1 Bore): {boresight_shift_arcsec:.3f} arcsec")
        print("--------------------------------------------------------\n")

    except Exception as e:
        print(f"FGS Boresight Calibration failed: {e}")

    print("\n========================================================")
    print("           INTERPRETATION OF RECOVERED DELTAS             ")
    print("========================================================")
    print("The values above are INVERSE CORRECTION VECTORS.")
    print("To nullify observed systematic biases and achieve")
    print("a zero-mean calibrated state, these corrections")
    print("should be applied to the spacecraft configuration.")
    print("")
    print(" - WFI Alignment: Add ΔRA/ΔDec/ΔPA to observation headers.")
    print(" - FGS Boresight: Apply updated BAM (SCF_AC_FGS_TBL_Qb)")
    print("                  quaternion to Flight Software.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
