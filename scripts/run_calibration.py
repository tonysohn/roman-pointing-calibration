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
import os
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pysiaf
import roman_datamodels as rdm
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from scipy.spatial.transform import Rotation as R

import roman_pointing.alignment
from roman_pointing import (
    align_wfi,
    apply_dva_scale_to_catalog,
    calibrate_roman_fgs_alignment,
)

# =========================================================================
# GLOBAL CONFIGURATION & OVERRIDES
# =========================================================================
# --- CROSSMATCH INPUT DIRECTORY ---
CROSSMATCH_DIR = "crossmatch_results"

# --- ALIGNMENT MODE ---
# Set to 5 for Full 5th-Order Polynomials (absorbs DVA).
# Set to 1 for standard Affine transformations (Scale & Skew only).
FIT_DEGREE = 5

# Toggle to enable/disable Differential Velocity Aberration (DVA) correction
# Only turn it off if spacecraft velocities are missing or corrupted.
APPLY_DVA = True

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

# --- CUSTOM SIAF TOGGLE ---
CUSTOM_SIAF_PATH = None  # Use this only when testing. Otherwise, start from PRD SIAF.
# =========================================================================


def load_standalone_matches(output_dir=CROSSMATCH_DIR):
    import glob
    import os

    import numpy as np
    from astropy.modeling import fitting, models
    from astropy.table import Table

    matched_data = {}
    ecsv_files = glob.glob(f"{output_dir}/*_xmatch.ecsv") + glob.glob(
        f"{output_dir}/*_matches.ecsv"
    )

    if not ecsv_files:
        print(f"No matched catalogs found in '{output_dir}'!")
        return None

    for f in ecsv_files:
        basename = os.path.basename(f)
        parts = basename.upper().split("_")
        det = next(
            (part for part in parts if part.startswith("WFI") and len(part) == 5), None
        )
        if not det:
            continue

        sca_key = f"{det}_FULL" if not det.endswith("_FULL") else det
        t = Table.read(f, format="ascii.ecsv")

        init_model = models.Polynomial2D(degree=2)
        fitter = fitting.LinearLSQFitter()

        m_ra = fitter(init_model, t["x"], t["y"], t["ra_epoch"])
        m_dec = fitter(init_model, t["x"], t["y"], t["dec_epoch"])

        d_ra = t["ra_epoch"] - m_ra(t["x"], t["y"])
        d_dec = t["dec_epoch"] - m_dec(t["x"], t["y"])

        scatter_arcsec = np.hypot(d_ra, d_dec) * 3600.0

        keep = scatter_arcsec < 5.0
        t = t[keep]

        matched_data[sca_key] = {
            "x_obs": np.array(t["x"]),
            "y_obs": np.array(t["y"]),
            "ra_cat": np.array(t["ra_epoch"]),
            "dec_cat": np.array(t["dec_epoch"]),
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
    fit_degree=5,
):
    from datetime import datetime

    current_date = datetime.now().strftime("%Y%m%d")
    output_filename = f"{output_prefix}_{current_date}.yml"
    yaml_lines = [f"version: '{current_date}'"]

    for special_cen in ["WFI_CEN", "WFI_TILE", "CGI_CEN"]:
        source_key = "WFI_CEN" if special_cen == "WFI_TILE" else special_cen
        if source_key in calibrated_siaf_params:
            params = calibrated_siaf_params[source_key]
            yaml_lines.append(f"{special_cen}:")
            yaml_lines.append(f"  V2Ref: {params['V2Ref']:.3f}")
            yaml_lines.append(f"  V3Ref: {params['V3Ref']:.3f}")
            yaml_lines.append(f"  V3IdlYAngle: {params['V3IdlYAngle']:.5f}")

    num_coeffs = int((fit_degree + 1) * (fit_degree + 2) / 2)
    mapping = {
        idx: f"{d}{y_deg}"
        for idx, (d, y_deg) in enumerate(
            [(d, y) for d in range(fit_degree + 1) for y in range(d + 1)]
        )
    }

    for sca_name in sorted(calibrated_siaf_params.keys()):
        if any(
            skip_str in sca_name.upper()
            for skip_str in ["WFI_CEN", "WFI_TILE", "CGI_CEN"]
        ):
            continue

        formatted_name = sca_name if "_FULL" in sca_name else f"{sca_name}_FULL"
        yaml_lines.append(f"{formatted_name}:")
        params = calibrated_siaf_params[sca_name]
        yaml_lines.append(f"  V2Ref: {params['V2Ref']:.3f}")
        yaml_lines.append(f"  V3Ref: {params['V3Ref']:.3f}")
        yaml_lines.append(f"  V3IdlYAngle: {params['V3IdlYAngle']:.5f}")

        if "Sci2IdlX10" in params:
            for prefix in ["Sci2IdlX", "Sci2IdlY", "Idl2SciX", "Idl2SciY"]:
                for i in range(num_coeffs):
                    suffix = mapping.get(i, "00")
                    key = f"{prefix}{suffix}"
                    val = 0.0 if suffix == "00" else params.get(key, 0.0)
                    yaml_lines.append(f"  {key}: {val:.8e}")

    with open(output_filename, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")
    return output_filename


def main():
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

            r_att = R.from_quat(acs_telemetry_qbj)
            ra_v1, dec_v1 = pysiaf.utils.rotations.pointing(r_att.as_matrix(), 0, 0)
            pa_v3 = pysiaf.utils.rotations.posangle(r_att.as_matrix(), 0, 0)

            ra_v1 = MANUAL_RA_V1 if MANUAL_RA_V1 is not None else ra_v1
            dec_v1 = MANUAL_DEC_V1 if MANUAL_DEC_V1 is not None else dec_v1
            pa_v3 = MANUAL_PA_V3 if MANUAL_PA_V3 is not None else pa_v3
        except AttributeError as e:
            print(f"Error: Pointing quaternion metadata missing from ASDF: {e}")
            return

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

    q_b2fgs_preflight = np.array(
        [
            -0.1859673417539929,
            +0.6837984564491885,
            -0.1800546332580956,
            +0.6822141509826322,
        ]
    )

    try:
        cat_file = "gaia_dr3_commissioning_field_wide.ecsv"
        print(f"  -> Loading reference catalog: {cat_file}")
        ref_catalog = Table.read(cat_file, format="ascii.ecsv")

        print(
            f"  -> Propagating Gaia proper motions to {obs_date_str} (Parallax forced to 0.0)"
        )

        sky_coords = SkyCoord(
            ra=np.asarray(ref_catalog["ra"]) * u.deg,
            dec=np.asarray(ref_catalog["dec"]) * u.deg,
            pm_ra_cosdec=np.asarray(ref_catalog["pmra"]) * u.mas / u.yr,
            pm_dec=np.asarray(ref_catalog["pmdec"]) * u.mas / u.yr,
            obstime=Time(np.asarray(ref_catalog["ref_epoch"]), format="jyear"),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            propagated_sky = sky_coords.apply_space_motion(
                new_obstime=Time(obs_date_str)
            )

        ref_catalog["ra_epoch"] = propagated_sky.ra.deg
        ref_catalog["dec_epoch"] = propagated_sky.dec.deg

        ref_catalog.sort("phot_g_mean_mag")

        mag_bright = 12.0
        mag_faint = 19.0
        bright_mask = (ref_catalog["phot_g_mean_mag"] <= mag_faint) & (
            ref_catalog["phot_g_mean_mag"] >= mag_bright
        )
        ref_catalog = ref_catalog[bright_mask]
        print(
            f"  -> Filtered reference catalog: {len(ref_catalog)} stars with {mag_bright} < G <= {mag_faint}"
        )

        if APPLY_DVA and has_dva_meta:
            print("  -> Applying Differential Velocity Aberration (DVA) to Gaia...")
            app_ra, app_dec = apply_dva_scale_to_catalog(
                np.asarray(ref_catalog["ra_epoch"]),
                np.asarray(ref_catalog["dec_epoch"]),
                dva_ra_ref,
                dva_dec_ref,
                dva_scale,
            )
            ref_catalog["ra_epoch"] = app_ra
            ref_catalog["dec_epoch"] = app_dec

    except Exception as e:
        print(f"Error loading local Gaia catalog: {e}")
        return

    print("\n--- 2. RUNNING WFI ALIGNMENT ---")

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
        precomputed_matches_dir=CROSSMATCH_DIR,
    )

    mean_dec_rad = np.deg2rad(attitude_results["DEC_V1"])
    d_ra_arcsec = (attitude_results["RA_V1"] - ra_v1) * np.cos(mean_dec_rad) * 3600.0
    d_dec_arcsec = (attitude_results["DEC_V1"] - dec_v1) * 3600.0
    d_pa_arcsec = (attitude_results["PA_V3"] - pa_v3) * 3600.0

    print("\n========================================================")
    print("           MEASURED COARSE POINTING OFFSETS       ")
    print("========================================================")
    print(f"Δ RA  (V1): {d_ra_arcsec:8.2f} arcsec")
    print(f"Δ Dec (V1): {d_dec_arcsec:8.2f} arcsec")
    print(f"Δ PA  (V3): {d_pa_arcsec:8.2f} arcsec")

    print("\nApplying rigorous rigid-body delta offsets to the SIAF...")

    pa_rad = np.deg2rad(pointing_info["PA_V3"])
    cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)

    delta_v2_arcsec = -(d_ra_arcsec * cos_pa + d_dec_arcsec * sin_pa)
    delta_v3_arcsec = +(d_ra_arcsec * sin_pa - d_dec_arcsec * cos_pa)
    delta_pa_deg = d_pa_arcsec / 3600.0

    base_cen = roman_siaf["WFI_CEN"]
    dtheta_rad = np.deg2rad(delta_pa_deg)
    cos_t, sin_t = np.cos(dtheta_rad), np.sin(dtheta_rad)

    for sca_name in list(calibrated_siaf_params.keys()):
        if sca_name in ["WFI_CEN", "WFI_TILE", "CGI_CEN"]:
            continue

        v2_local = calibrated_siaf_params[sca_name]["V2Ref"]
        v3_local = calibrated_siaf_params[sca_name]["V3Ref"]

        dx = v2_local - base_cen.V2Ref
        dy = v3_local - base_cen.V3Ref

        calibrated_siaf_params[sca_name]["V2Ref"] = (
            base_cen.V2Ref + delta_v2_arcsec + (dx * cos_t + dy * sin_t)
        )
        calibrated_siaf_params[sca_name]["V3Ref"] = (
            base_cen.V3Ref + delta_v3_arcsec + (-dx * sin_t + dy * cos_t)
        )
        calibrated_siaf_params[sca_name]["V3IdlYAngle"] += delta_pa_deg

    calibrated_siaf_params["WFI_CEN"] = {
        "V2Ref": base_cen.V2Ref + delta_v2_arcsec,
        "V3Ref": base_cen.V3Ref + delta_v3_arcsec,
        "V3IdlYAngle": base_cen.V3IdlYAngle + delta_pa_deg,
    }

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
            q_b2fgs_old=q_b2fgs_preflight,
        )
        print("\n========================================================")
        print("           FGS BORESIGHT CALIBRATION RESULTS             ")
        print("========================================================")
        print(f"Updated BAM Telemetry (SCF_AC_FGS_TBL_Qb):")
        print(
            f"[{q_b2fgs_calibrated[0]:.17f}, {q_b2fgs_calibrated[1]:.17f}, {q_b2fgs_calibrated[2]:.17f}, {q_b2fgs_calibrated[3]:.17f}]"
        )

        q_nom = R.from_quat(q_b2fgs_preflight)
        q_cal = R.from_quat(q_b2fgs_calibrated)
        delta_q = q_cal * q_nom.inv()

        v1_nominal = np.array([1, 0, 0])
        v1_calibrated = delta_q.apply(v1_nominal)

        cos_theta = np.clip(np.dot(v1_nominal, v1_calibrated), -1.0, 1.0)
        boresight_shift_arcsec = np.degrees(np.arccos(cos_theta)) * 3600.0

        print(f"Total Boresight Shift (V1 Bore): {boresight_shift_arcsec:.3f} arcsec")
        print("--------------------------------------------------------\n")

        m_b2fgs = R.from_quat(q_b2fgs_calibrated).inv().as_matrix()

        m_x2z = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        a = m_x2z.T @ m_b2fgs

        bz_rad = np.arcsin(np.clip(a[0, 2], -1.0, 1.0))
        by_rad = np.arctan2(-a[0, 1], a[0, 0])
        ya_rad = np.arctan2(-a[1, 2], a[2, 2])

        hw_angle = np.rad2deg(ya_rad)
        hw_v2 = np.rad2deg(by_rad) * 3600.0
        hw_v3 = np.rad2deg(bz_rad) * 3600.0

        hw_angle -= 180.0
        hw_angle = (hw_angle + 180.0) % 360.0 - 180.0
        hw_v2 *= -1.0

        print(
            f"BAM-Derived WFI_CEN -> V2: {hw_v2:.3f}, V3: {hw_v3:.3f}, Angle: {hw_angle:.5f}"
        )

        m_b2fgs_nom = R.from_quat(q_b2fgs_preflight).inv().as_matrix()
        a_nom = m_x2z.T @ m_b2fgs_nom
        ya_nom_rad = np.arctan2(-a_nom[1, 2], a_nom[2, 2])
        hw_angle_nom = np.rad2deg(ya_nom_rad)
        hw_angle_nom -= 180.0
        hw_angle_nom = (hw_angle_nom + 180.0) % 360.0 - 180.0
        delta_hw_angle = hw_angle - hw_angle_nom

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

    from roman_pointing.diagnostics import generate_alignment_diagnostics

    generate_alignment_diagnostics(
        matched_pairs_log=matched_pairs_log,
        iteration_history=attitude_results.get("iteration_history", []),
        output_dir="./diagnostics",
        calibrated_siaf_params=calibrated_siaf_params,
        nominal_siaf=pristine_siaf,
    )

    pd.DataFrame(
        matched_pairs_log,
        columns=[
            "SCA",
            "X",
            "Y",
            "RA",
            "Dec",
            "Flux",
            "Mag",
            "V2_True",
            "V3_True",
            "ResV2_mas",
            "ResV3_mas",
        ],
    ).to_csv("verification_catalog.csv", index=False)

    print("\nAnchoring rigid mosaic to the newly derived BAM center...")

    try:
        bam_v2, bam_v3, bam_angle = hw_v2, hw_v3, hw_angle
    except NameError:
        print("Warning: FGS Boresight failed. Falling back to pristine SIAF center.")
        bam_v2 = pristine_siaf["WFI_CEN"].V2Ref
        bam_v3 = pristine_siaf["WFI_CEN"].V3Ref
        bam_angle = pristine_siaf["WFI_CEN"].V3IdlYAngle

    step2_cen = calibrated_siaf_params["WFI_CEN"]

    dAngle_bulk = bam_angle - step2_cen["V3IdlYAngle"]
    dtheta_rad = np.deg2rad(dAngle_bulk)
    cos_t, sin_t = np.cos(dtheta_rad), np.sin(dtheta_rad)

    for sca in calibrated_siaf_params:
        if "WFI_CEN" in sca or "CGI_CEN" in sca:
            continue

        dx = calibrated_siaf_params[sca]["V2Ref"] - step2_cen["V2Ref"]
        dy = calibrated_siaf_params[sca]["V3Ref"] - step2_cen["V3Ref"]

        calibrated_siaf_params[sca]["V2Ref"] = bam_v2 + (dx * cos_t + dy * sin_t)
        calibrated_siaf_params[sca]["V3Ref"] = bam_v3 + (-dx * sin_t + dy * cos_t)
        calibrated_siaf_params[sca]["V3IdlYAngle"] += dAngle_bulk

    calibrated_siaf_params["WFI_CEN"] = {
        "V2Ref": bam_v2,
        "V3Ref": bam_v3,
        "V3IdlYAngle": bam_angle,
    }

    if "CGI_CEN" in pristine_siaf.apertures:
        cgi_old = pristine_siaf["CGI_CEN"]
        wfi_old = pristine_siaf["WFI_CEN"]

        v2_wfi_old, v3_wfi_old = wfi_old.V2Ref, wfi_old.V3Ref
        v2_wfi_new, v3_wfi_new = bam_v2, bam_v3

        M_old = pysiaf.utils.rotations.attitude(v2_wfi_old, v3_wfi_old, 0.0, 0.0, 0.0)
        M_new = pysiaf.utils.rotations.attitude(
            v2_wfi_new, v3_wfi_new, 0.0, 0.0, dAngle_bulk
        )

        c2_rad = np.deg2rad(cgi_old.V2Ref / 3600.0)
        c3_rad = np.deg2rad(cgi_old.V3Ref / 3600.0)
        cgi_vec_old = np.array(
            [
                np.cos(c3_rad) * np.cos(c2_rad),
                np.cos(c3_rad) * np.sin(c2_rad),
                np.sin(c3_rad),
            ]
        )

        cgi_vec_sky = np.dot(M_old, cgi_vec_old)
        cgi_vec_new = np.dot(M_new.T, cgi_vec_sky)

        v2_cgi_new = np.rad2deg(np.arctan2(cgi_vec_new[1], cgi_vec_new[0])) * 3600.0
        v3_cgi_new = np.rad2deg(np.arcsin(cgi_vec_new[2])) * 3600.0
        angle_cgi_new = cgi_old.V3IdlYAngle + dAngle_bulk

        calibrated_siaf_params["CGI_CEN"] = {
            "V2Ref": v2_cgi_new,
            "V3Ref": v3_cgi_new,
            "V3IdlYAngle": angle_cgi_new,
        }
        print(
            f"Updated CGI_CEN -> V2Ref: {v2_cgi_new:.3f}, V3Ref: {v3_cgi_new:.3f}, Angle: {angle_cgi_new:.5f}"
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
