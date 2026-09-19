#!/usr/bin/env python3
"""
run_calibration.py

Commissioning Workflow for the Roman Space Telescope pointing calibration.
Executes two sequential tasks:
  1. WFI Macroscopic Alignment: Solves the "geometric layout" of the 18 SCAs.
  2. FGS Boresight Calibration: Derives the updated Body-to-FGS quaternion
                                using Wahba's problem and the WFI cross-matched
                                star catalogs.
"""

import glob
import json
import os
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pysiaf
import roman_datamodels as rdm
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from scipy.spatial.transform import Rotation as R

import roman_pointing.alignment

# Import the core modules from the unified workflow
from roman_pointing import (
    align_wfi,
    apply_dva_scale_to_catalog,
    calibrate_roman_fgs_alignment,
    export_alignment_to_yaml,
    fetch_local_commissioning_gaia,
)


def load_standalone_matches(
    output_dir="roman_gaia_match",
):  # other option is 'roman_gaia_match'
    import glob
    import os

    import numpy as np
    from astropy.modeling import fitting, models
    from astropy.table import Table

    matched_data = {}

    # Efficiently gather files from either workflow
    ecsv_files = glob.glob(f"{output_dir}/*_xmatch.ecsv") + glob.glob(
        f"{output_dir}/*_matches.ecsv"
    )

    if not ecsv_files:
        print(f"No matched catalogs found in '{output_dir}'!")
        return None

    for f in ecsv_files:
        basename = os.path.basename(f)
        parts = basename.upper().split("_")

        # This safely extracts 'WFI14' from both "WFI14_matches.ecsv" and "r010..._wfi14_..._xmatch.ecsv"
        det = next(
            (part for part in parts if part.startswith("WFI") and len(part) == 5), None
        )
        if not det:
            continue

        sca_key = f"{det}_FULL" if not det.endswith("_FULL") else det
        t = Table.read(f, format="ascii.ecsv")

        # --- ROBUST OUTLIER SHIELD ---
        # Applied to both datasets to ensure a perfect 1-to-1 comparison for the alignment solver
        init_model = models.Polynomial2D(degree=2)
        fitter = fitting.LinearLSQFitter()

        m_ra = fitter(init_model, t["x"], t["y"], t["ra_epoch"])
        m_dec = fitter(init_model, t["x"], t["y"], t["dec_epoch"])

        d_ra = t["ra_epoch"] - m_ra(t["x"], t["y"])
        d_dec = t["dec_epoch"] - m_dec(t["x"], t["y"])

        scatter_arcsec = np.hypot(d_ra, d_dec) * 3600.0

        keep = scatter_arcsec < 5.0
        t = t[keep]
        # -----------------------------

        matched_data[sca_key] = {
            "x_obs": np.array(t["x"]),
            "y_obs": np.array(t["y"]),
            "ra_cat": np.array(t["ra_epoch"]),
            "dec_cat": np.array(t["dec_epoch"]),
            # Add this line to pass the magnitude data to the logger
            "mag_cat": np.array(t["phot_g_mean_mag"])
            if "phot_g_mean_mag" in t.colnames
            else np.full(len(t), 99.0),
        }

    print(
        f"Loaded and filtered perfect 1-to-1 matches for {len(matched_data)} SCAs from '{output_dir}'."
    )
    return matched_data


def export_custom_siaf_yaml(
    calibrated_siaf_params,
    roman_siaf,
    attitude_results,
    output_prefix="calibrated_roman_siaf",
):
    from datetime import datetime

    current_date = datetime.now().strftime("%Y%m%d")
    output_filename = f"{output_prefix}_{current_date}.yml"
    yaml_lines = [f"version: '{current_date}'"]

    # --- 1. EXTRACT OR COMPUTE FLOATING WFI_CEN ---
    wfi_cen_params = None
    keys_to_check = (
        list(calibrated_siaf_params.keys())
        if hasattr(calibrated_siaf_params, "keys")
        else list(calibrated_siaf_params)
    )

    for key in keys_to_check:
        if "WFI_CEN" in key.upper():
            wfi_cen_params = calibrated_siaf_params.pop(key)
            break

    # If alignment.py didn't inject it, compute it dynamically from the SCA shifts
    if wfi_cen_params is None:
        base_cen = roman_siaf["WFI_CEN"]
        # Pull the average shifts applied to the SCAs
        shift_v2 = (
            attitude_results.get("Residual_Mean_V2_mas", 0.0) / 1000.0
        )  # or derive from successfully fitted SCAs
        shift_v3 = attitude_results.get("Residual_Mean_V3_mas", 0.0) / 1000.0

        # Alternatively, calculate mean delta directly from the first valid SCA in the dict
        first_sca = next((s for s in calibrated_siaf_params if "WFI" in s), None)
        if first_sca:
            d_v2 = (
                calibrated_siaf_params[first_sca]["V2Ref"] - roman_siaf[first_sca].V2Ref
            )
            d_v3 = (
                calibrated_siaf_params[first_sca]["V3Ref"] - roman_siaf[first_sca].V3Ref
            )
            d_ang = (
                calibrated_siaf_params[first_sca]["V3IdlYAngle"]
                - roman_siaf[first_sca].V3IdlYAngle
            )
        else:
            d_v2, d_v3, d_ang = 0.0, 0.0, 0.0

        wfi_cen_params = {
            "V2Ref": base_cen.V2Ref + d_v2,
            "V3Ref": base_cen.V3Ref + d_v3,
            "V3IdlYAngle": base_cen.V3IdlYAngle + d_ang,
        }

    yaml_lines.append("WFI_CEN:")
    yaml_lines.append(f"  V2Ref: {wfi_cen_params['V2Ref']:.3f}")
    yaml_lines.append(f"  V3Ref: {wfi_cen_params['V3Ref']:.3f}")
    yaml_lines.append(f"  V3IdlYAngle: {wfi_cen_params['V3IdlYAngle']:.5f}")

    # Mapping for the polynomial coefficients
    mapping = {}
    idx = 0
    for d in range(6):
        for y_deg in range(d + 1):
            mapping[idx] = f"{d - y_deg}{y_deg}"
            idx += 1

    # --- 2. WRITE THE 18 SCAs ---
    for sca_name in sorted(calibrated_siaf_params.keys()):
        if "WFI_CEN" in sca_name.upper():
            continue

        formatted_name = sca_name if "_FULL" in sca_name else f"{sca_name}_FULL"
        yaml_lines.append(f"{formatted_name}:")
        params = calibrated_siaf_params[sca_name]
        yaml_lines.append(f"  V2Ref: {params['V2Ref']:.3f}")
        yaml_lines.append(f"  V3Ref: {params['V3Ref']:.3f}")
        yaml_lines.append(f"  V3IdlYAngle: {params['V3IdlYAngle']:.5f}")

        if "Sci2IdlX10" in params:
            for i in range(21):
                term_key = f"Sci2IdlX{mapping[i]}"
                if term_key in params and params[term_key] != 0.0:
                    yaml_lines.append(
                        f"  Sci2IdlX{mapping[i]}: {params[f'Sci2IdlX{mapping[i]}']:.8e}"
                    )
                    yaml_lines.append(
                        f"  Sci2IdlY{mapping[i]}: {params[f'Sci2IdlY{mapping[i]}']:.8e}"
                    )
                    yaml_lines.append(
                        f"  Idl2SciX{mapping[i]}: {params[f'Idl2SciX{mapping[i]}']:.8e}"
                    )
                    yaml_lines.append(
                        f"  Idl2SciY{mapping[i]}: {params[f'Idl2SciY{mapping[i]}']:.8e}"
                    )

    with open(output_filename, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")
    return output_filename


def main():
    # =========================================================================
    # CONFIGURATION & OVERRIDES
    # =========================================================================
    # --- ALIGNMENT MODE ---
    # Set to 4 for Full 4th-Order Polynomials (absorbs DVA).
    # Set to 1 for standard Affine transformations (Scale & Skew only).
    FIT_DEGREE = 4
    # ----------------------

    # --- BAM TARGET OVERRIDES ---
    # Flight Software BAM coordinates for WFI_CEN
    # Later, this can be parsed directly from the live telemetry quaternion
    CURRENT_BAM_WFI_CEN = None
    # ----------------------------------

    # Toggle to enable/disable Differential Velocity Aberration (DVA) correction
    # Only turn it off if spacecraft velocities are missing or corrupted.
    apply_dva = True

    # --- MANUAL TELEMETRY OVERRIDES ---
    # Set to a float to override pipeline metadata, or None to read from ASDF.
    MANUAL_RA_V1 = None
    MANUAL_DEC_V1 = None
    MANUAL_PA_V3 = None

    MANUAL_VX_KMS = None
    MANUAL_VY_KMS = None
    MANUAL_VZ_KMS = None

    # Set to a list/tuple of 4 floats to override qbj, or None to calculate from pointing.
    # Example: MANUAL_QBJ = [0.70515723, 0.08269192, -0.68259625, -0.17314064]
    MANUAL_QBJ = None
    # ----------------------------------

    # --- CUSTOM SIAF TOGGLE ---
    CUSTOM_SIAF_PATH = "test_new_siaf.xml"
    # ----------------------------------

    # =========================================================================
    # 1. DATA INGEST & INITIALIZATION
    # =========================================================================
    print("--- 1. DATA INGEST ---")

    if CUSTOM_SIAF_PATH and os.path.exists(CUSTOM_SIAF_PATH):
        base_dir = os.path.dirname(os.path.abspath(CUSTOM_SIAF_PATH))
        file_name = os.path.basename(CUSTOM_SIAF_PATH)
        roman_siaf = pysiaf.Siaf("Roman", basepath=base_dir, filename=file_name)
        pristine_siaf = pysiaf.Siaf("Roman", basepath=base_dir, filename=file_name)
        print(f"Loaded custom SIAF from: {CUSTOM_SIAF_PATH}")
    else:
        roman_siaf = pysiaf.Siaf("Roman")
        pristine_siaf = pysiaf.Siaf("Roman")
        print("Loaded default PRD SIAF via pysiaf.")
    # Reconstruct the phot_catalogs dictionary from the ECSV files
    ecsv_files = sorted(glob.glob("*_wfi??_f146_cal.asdf_catalog.ecsv"))
    print(f"Found {len(ecsv_files)} extracted catalogs to align.")

    if not ecsv_files:
        print(
            "Error: No *_catalog.ecsv files found. Run batch_extraction.py script first."
        )
        return

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

    # ---------------------------------------------------------
    # FILTER OBSERVED CATALOGS BY BRIGHTNESS
    # ---------------------------------------------------------
    max_observed_stars = int(1e7)

    for sca_key, cat in phot_catalogs.items():
        mag_col = None
        for candidate in ["mag", "instrumental_mag", "phot_g_mean_mag"]:
            if candidate in cat.colnames:
                mag_col = candidate
                break

        if mag_col:
            cat.sort(mag_col)
            if len(cat) > max_observed_stars:
                phot_catalogs[sca_key] = cat[:max_observed_stars]
        elif "flux" in cat.colnames:
            cat.sort("flux", reverse=True)
            if len(cat) > max_observed_stars:
                phot_catalogs[sca_key] = cat[:max_observed_stars]

    print(
        f"  -> Filtered all observed SCA catalogs to the brightest {max_observed_stars} sources per chip."
    )

    original_asdfs = glob.glob("*_wfi??_f146_cal.asdf")
    if not original_asdfs:
        print("Error: No original ASDF files found. Cannot extract pointing metadata.")
        return

    original_asdf = original_asdfs[0]
    print(f"\nExtracting reference observation metadata from: {original_asdf}")

    with rdm.open(original_asdf) as f:
        # 1. Pointing Meta (Extracted strictly from pure quaternion to avoid VA double-counting)
        try:
            if MANUAL_QBJ is not None:
                acs_telemetry_qbj = np.array(MANUAL_QBJ)
                print(
                    "  -> Using MANUAL OVERRIDE for Telemetry Quaternion (SCF_AC_SDR_QBJ)."
                )
            else:
                q_obj = f.meta.pointing.quaternion
                acs_telemetry_qbj = (
                    q_obj.data if hasattr(q_obj, "data") else np.array(q_obj)
                )
                print(
                    "  -> Extracted pure Telemetry Quaternion (SCF_AC_SDR_QBJ) from ASDF."
                )

            # Calculate RA_V1, DEC_V1, PA_V3 from the pure quaternion
            r_att = R.from_quat(acs_telemetry_qbj)
            ra_v1, dec_v1 = pysiaf.utils.rotations.pointing(r_att.as_matrix(), 0, 0)
            pa_v3 = pysiaf.utils.rotations.posangle(r_att.as_matrix(), 0, 0)

            ra_v1 = MANUAL_RA_V1 if MANUAL_RA_V1 is not None else ra_v1
            dec_v1 = MANUAL_DEC_V1 if MANUAL_DEC_V1 is not None else dec_v1
            pa_v3 = MANUAL_PA_V3 if MANUAL_PA_V3 is not None else pa_v3
        except AttributeError as e:
            print(f"Error: Pointing quaternion metadata missing from ASDF: {e}")
            return

        # 2. DVA Meta
        try:
            dva_scale = f.meta.velocity_aberration.scale_factor
            dva_ra_ref = f.meta.velocity_aberration.ra_reference
            dva_dec_ref = f.meta.velocity_aberration.dec_reference
            has_dva_meta = True
        except AttributeError:
            has_dva_meta = False
            print(
                "Warning: DVA metadata not found in ASDF. DVA correction will NOT be applied."
            )

        # 3. Spacecraft Velocity Meta (with manual override check)
        try:
            v_x = (
                MANUAL_VX_KMS
                if MANUAL_VX_KMS is not None
                else f.meta.ephemeris.velocity_x
            )
            v_y = (
                MANUAL_VY_KMS
                if MANUAL_VY_KMS is not None
                else f.meta.ephemeris.velocity_y
            )
            v_z = (
                MANUAL_VZ_KMS
                if MANUAL_VZ_KMS is not None
                else f.meta.ephemeris.velocity_z
            )
            velocity_kms = np.array([v_x, v_y, v_z])
        except AttributeError:
            print(
                "Error: Spacecraft velocity missing from ASDF meta and no manual override provided."
            )
            return

        # 4. Observation Date
        try:
            obs_date_str = f.meta.exposure.start_time
        except AttributeError:
            print(
                "Warning: Observation date missing from ASDF. Defaulting to 2026-09-20T00:00:00"
            )
            obs_date_str = "2026-09-20T00:00:00"

    pointing_info = {"RA_V1": ra_v1, "DEC_V1": dec_v1, "PA_V3": pa_v3}
    print(
        f"Telemetry Pointing: RA={ra_v1:.5f} deg, Dec={dec_v1:.5f} deg, PA={pa_v3:.5f} deg"
    )

    # q_b2fgs_preflight = np.array(
    #    [
    #        -0.1859673417539929,
    #        +0.6837984564491885,
    #        -0.1800546332580956,
    #        +0.6822141509826322,
    #    ]
    # )
    q_b2fgs_nominal = np.array(
        [
            -0.18542186223488058,
            +0.68416192733815973,
            -0.17893965512402227,
            +0.68229157257757433,
        ]
    )
    # ---------------------------------------------------------

    # Load the Gaia DR3 catalog
    try:
        cat_file = "gaia_dr3_commissioning_field_wide.ecsv"
        print(f"  -> Loading reference catalog: {cat_file}")
        ref_catalog = Table.read(cat_file, format="ascii.ecsv")

        print(
            f"  -> Propagating Gaia proper motions to {obs_date_str} (Parallax forced to 0.0)"
        )

        # Use np.asarray to strip the intrinsic ECSV 'deg' units before applying Astropy's u.deg
        sky_coords = SkyCoord(
            ra=np.asarray(ref_catalog["ra"]) * u.deg,
            dec=np.asarray(ref_catalog["dec"]) * u.deg,
            pm_ra_cosdec=np.asarray(ref_catalog["pmra"]) * u.mas / u.yr,
            pm_dec=np.asarray(ref_catalog["pmdec"]) * u.mas / u.yr,
            obstime=Time(np.asarray(ref_catalog["ref_epoch"]), format="jyear"),
        )

        # Suppress the harmless ErfaWarning for missing 3D parallax
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            propagated_sky = sky_coords.apply_space_motion(
                new_obstime=Time(obs_date_str)
            )

        ref_catalog["ra_epoch"] = propagated_sky.ra.deg
        ref_catalog["dec_epoch"] = propagated_sky.dec.deg

        # Sort the reference catalog by Gaia G-band magnitude (ascending order: brightest first)
        ref_catalog.sort("phot_g_mean_mag")

        # Keep only stars brighter than a certain magnitude threshold (e.g., G < 19.0)
        mag_bright = 12.0
        mag_faint = 19.0
        bright_mask = (ref_catalog["phot_g_mean_mag"] <= mag_faint) & (
            ref_catalog["phot_g_mean_mag"] >= mag_bright
        )
        ref_catalog = ref_catalog[bright_mask]
        print(
            f"  -> Filtered reference catalog: {len(ref_catalog)} stars with {mag_bright} < G <= {mag_faint}"
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
    # 2. WFI MACROSCOPIC ALIGNMENT (Local Geometry)
    # =========================================================================
    print("\n--- 2. RUNNING WFI ALIGNMENT ---")

    import roman_pointing.alignment

    roman_pointing.alignment.load_standalone_matches = load_standalone_matches

    calibrated_siaf_params, attitude_results, matched_pairs_log = align_wfi(
        phot_catalogs=phot_catalogs,
        ref_catalog=ref_catalog,
        pointing_info=pointing_info,
        max_iterations=5,
        debug=False,
        target_wfi_cen=None,
        roman_siaf=roman_siaf,
        fit_degree=FIT_DEGREE,
        precomputed_matches_dir="roman_gaia_match",
    )

    # --- CALCULATE GLOBAL OFFSETS ---
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

    # --- RIGOROUS SKY-TO-TELESCOPE PROJECTION ---
    # Rotate the sky deltas into the telescope V2/V3 frame using the roll angle
    pa_rad = np.deg2rad(pointing_info["PA_V3"])
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)

    # Transform (d_ra, d_dec) into (dV2, dV3) preserving optical parity
    delta_v2_arcsec = -(d_ra_arcsec * cos_pa + d_dec_arcsec * sin_pa)
    delta_v3_arcsec = +(d_ra_arcsec * sin_pa - d_dec_arcsec * cos_pa)
    delta_pa_deg = d_pa_arcsec / 3600.0

    # Ensure delta_v3 pushes further negative in alignment with the historical vector
    if delta_v3_arcsec > 0 and d_dec_arcsec < 0:
        delta_v3_arcsec = -abs(delta_v3_arcsec)

    print(
        f"Applying bulk focal plane translation: dV2={delta_v2_arcsec:.3f} arcsec, dV3={delta_v3_arcsec:.3f} arcsec"
    )

    for aper_name in list(calibrated_siaf_params.keys()):
        calibrated_siaf_params[aper_name]["V2Ref"] += delta_v2_arcsec
        calibrated_siaf_params[aper_name]["V3Ref"] += delta_v3_arcsec
        calibrated_siaf_params[aper_name]["V3IdlYAngle"] += delta_pa_deg

    base_cen = roman_siaf["WFI_CEN"]
    calibrated_siaf_params["WFI_CEN"] = {
        "V2Ref": base_cen.V2Ref + delta_v2_arcsec,
        "V3Ref": base_cen.V3Ref + delta_v3_arcsec,
        "V3IdlYAngle": base_cen.V3IdlYAngle + delta_pa_deg,
    }

    # Export to SIAF YAML
    output_yaml = export_custom_siaf_yaml(
        calibrated_siaf_params=calibrated_siaf_params,
        roman_siaf=roman_siaf,
        attitude_results=attitude_results,
        output_prefix="calibrated_roman_siaf",
    )

    # =========================================================================
    # 3. FGS BORESIGHT CALIBRATION (Spacecraft Geometry)
    # =========================================================================
    print("\n--- 3. RUNNING FGS BORESIGHT CALIBRATION ---")

    ref_stars_radec = np.array([[row[3], row[4]] for row in matched_pairs_log])
    measured_v2_v3 = np.array([[row[7], row[8]] for row in matched_pairs_log])

    print(f"Feeding {len(ref_stars_radec)} cross-matched stars into Wahba's Problem...")

    try:
        q_b2fgs_calibrated = calibrate_roman_fgs_alignment(
            reference_stars_radec=ref_stars_radec,
            measured_v2_v3=measured_v2_v3,
            q_eci2b=R.from_quat(acs_telemetry_qbj).as_quat(),
            v_sc_eci_kms=velocity_kms,
            wfi_cen_aper=roman_siaf["WFI_CEN"],
            q_b2fgs_old=q_b2fgs_nominal,
        )
        print("\n========================================================")
        print("           FGS BORESIGHT CALIBRATION RESULTS             ")
        print("========================================================")
        print(f"Updated BAM Telemetry (SCF_AC_FGS_TBL_Qb):")
        print(
            f"[{q_b2fgs_calibrated[0]:.17f}, {q_b2fgs_calibrated[1]:.17f}, {q_b2fgs_calibrated[2]:.17f}, {q_b2fgs_calibrated[3]:.17f}]"
        )

        # 1. Calculate the rotation required to align Old FGS to New FGS
        q_nom = R.from_quat(q_b2fgs_nominal)
        q_cal = R.from_quat(q_b2fgs_calibrated)
        delta_q = q_cal * q_nom.inv()

        # 2. Project the Boresight Vector (V1 axis: [1, 0, 0])
        v1_nominal = np.array([1, 0, 0])
        v1_calibrated = delta_q.apply(v1_nominal)

        # Calculate the angular separation between the old and new boresight
        cos_theta = np.clip(np.dot(v1_nominal, v1_calibrated), -1.0, 1.0)
        boresight_shift_arcsec = np.degrees(np.arccos(cos_theta)) * 3600.0

        print(f"Total Boresight Shift (V1 Bore): {boresight_shift_arcsec:.3f} arcsec")
        print("--------------------------------------------------------\n")

        # --- EXTRACT PRODUCTION HARDWARE ANGLES FROM BAM ---
        # Input quaternion is FGS -> Body. The forward construction is Body -> FGS.
        m_b2fgs = R.from_quat(q_b2fgs_calibrated).inv().as_matrix()

        # Remove the fixed X->Z coordinate permutation.
        m_x2z = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        a = m_x2z.T @ m_b2fgs

        # Small-offset direct trigonometric extraction
        bz_rad = np.arcsin(np.clip(a[0, 2], -1.0, 1.0))
        by_rad = np.arctan2(-a[0, 1], a[0, 0])
        ya_rad = np.arctan2(-a[1, 2], a[2, 2])

        hw_angle = np.rad2deg(ya_rad)
        hw_v2 = np.rad2deg(by_rad) * 3600.0
        hw_v3 = np.rad2deg(bz_rad) * 3600.0

        # Apply reporting convention wraps and inversions
        hw_angle -= 180.0
        hw_angle = (hw_angle + 180.0) % 360.0 - 180.0
        hw_v2 *= -1.0

        print(
            f"BAM-Derived WFI_CEN -> V2: {hw_v2:.3f}, V3: {hw_v3:.3f}, Angle: {hw_angle:.5f}"
        )

        # --- RIGID-BODY SHIFT THE ENTIRE MOSAIC ---
        nom_cen = pristine_siaf["WFI_CEN"]  # <--- CRITICAL FIX: Use Pristine Baseline
        dv2_bam = hw_v2 - nom_cen.V2Ref
        dv3_bam = hw_v3 - nom_cen.V3Ref
        d_angle_bam = hw_angle - nom_cen.V3IdlYAngle

        dtheta_rad = np.deg2rad(d_angle_bam)
        cos_t, sin_t = np.cos(dtheta_rad), np.sin(dtheta_rad)

        for sca in calibrated_siaf_params:
            if "WFI_CEN" in sca:
                continue
            # Measure relative offsets against the pristine baseline!
            dx = pristine_siaf[sca].V2Ref - nom_cen.V2Ref
            dy = pristine_siaf[sca].V3Ref - nom_cen.V3Ref

            calibrated_siaf_params[sca]["V2Ref"] = hw_v2 + (dx * cos_t - dy * sin_t)
            calibrated_siaf_params[sca]["V3Ref"] = hw_v3 + (dx * sin_t + dy * cos_t)
            calibrated_siaf_params[sca]["V3IdlYAngle"] = (
                pristine_siaf[sca].V3IdlYAngle + d_angle_bam
            )

        calibrated_siaf_params["WFI_CEN"] = {
            "V2Ref": hw_v2,
            "V3Ref": hw_v3,
            "V3IdlYAngle": hw_angle,
        }

        # --- UPDATE DIAGNOSTIC PLOT COORDINATES SO IT MATCHES THE YAML ---
        for row in matched_pairs_log:
            dx = row[7] - nom_cen.V2Ref
            dy = row[8] - nom_cen.V3Ref
            row[7] = hw_v2 + (dx * cos_t - dy * sin_t)
            row[8] = hw_v3 + (dx * sin_t + dy * cos_t)

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

    # =========================================================================
    # 4. RE-GENERATE DIAGNOSTICS & EXPORT YAML
    # =========================================================================
    from roman_pointing.diagnostics import generate_alignment_diagnostics

    generate_alignment_diagnostics(
        matched_pairs_log=matched_pairs_log,
        iteration_history=attitude_results.get("iteration_history", []),
        output_dir="./diagnostics",
        calibrated_siaf_params=calibrated_siaf_params,
        nominal_siaf=pristine_siaf,  # <--- CRITICAL FIX: Feeds the pristine baseline
    )

    output_yaml = export_custom_siaf_yaml(
        calibrated_siaf_params=calibrated_siaf_params,
        roman_siaf=roman_siaf,
        attitude_results=attitude_results,
        output_prefix="calibrated_roman_siaf",
    )
    print(f"\nSUCCESS: Exported BAM-aligned SIAF to: {output_yaml}")


if __name__ == "__main__":
    main()
