#!/usr/bin/env python3
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from astropy.io import fits
from pysiaf.utils import rotations
from scipy.spatial.transform import Rotation as R

# ==========================================
# USER CONFIGURATION
# ==========================================
# --- File I/O ---
INPUT_FITS_FILE = "cgi_simulated_anomaly.fits"
OUTPUT_PREFIX = "roman_cgi_updates"

# --- Ground Baselines ---
BASELINE_CGI_V2 = -1106.133
BASELINE_CGI_V3 = -838.580

# --- OPERATION MODE SELECTION ---
# Options: 'TIER1_WFI_UNIFIED', 'TIER2_FACE_TELEMETRY', 'TIER3_COMMANDED'
OPERATION_MODE = "TIER1_WFI_UNIFIED"

# --- TIER 1 INPUTS (WFI Boresight Solution) ---
# Insert the final global attitude outputs from run_calibration.py
SOLVED_WFI_RA = 80.553429
SOLVED_WFI_DEC = -69.514097
SOLVED_WFI_PA = 75.002227

# --- TIER 2 INPUTS (FACE Telemetry) ---
# Insert downloaded time-averaged FGS quaternions [x, y, z, w]
TELEMETRY_Q1 = -0.18596734
TELEMETRY_Q2 = 0.68379845
TELEMETRY_Q3 = -0.18005463
TELEMETRY_Q4 = 0.68221415

# --- TIER 3 INPUTS (Commanded Fallback) ---
# Insert original commanded pointing parameters
CMD_RA = 80.553429
CMD_DEC = -69.514097
CMD_PA = 75.002227


# ==========================================
# MODULE A: INGESTION
# ==========================================
@dataclass
class CGIFitsData:
    """Data structure to hold the parsed CGI astrometric parameters from FITS files."""

    ra_deg: float
    dec_deg: float
    plate_scale_mas: float
    north_angle_deg: float


def ingest_cgi_fits(fits_path: str) -> CGIFitsData:
    """Reads the CGI DRP FITS product and parses the core astrometric data."""
    with fits.open(fits_path) as hdul:
        data_array = hdul[1].data
        ra = float(data_array[0])
        dec = float(data_array[1])
        plate_scale = float(data_array[2])
        north_angle = float(data_array[3])
        return CGIFitsData(ra, dec, plate_scale, north_angle)


# ==========================================
# MODULE B: KINEMATIC SOLVER
# ==========================================
class CGIAlignmentSolver:
    """Computes the Body-to-CGI alignment offsets and local orientation."""

    def __init__(self, cgi_v2_baseline: float, cgi_v3_baseline: float):
        self.cgi_v2 = cgi_v2_baseline
        self.cgi_v3 = cgi_v3_baseline

    def _apply_kinematics(self, att_matrix, meas_ra, meas_dec, meas_north_angle):
        """
        [CORE ENGINE] Projects sky coordinates into the provided global frame.
        Native 3D spherical projection bypasses 2D delta-addition errors.
        """
        # Project absolute RA/Dec directly to absolute V2/V3
        updated_v2, updated_v3 = rotations.getv2v3(att_matrix, meas_ra, meas_dec)

        # Calculate orientation
        pa_cgi_cen = rotations.posangle(att_matrix, updated_v2, updated_v3)
        v3idlyangle = (meas_north_angle - pa_cgi_cen + 180) % 360 - 180

        return updated_v2, updated_v3, v3idlyangle

    # --- TIER 1: Perfect Unified Solution ---
    def compute_alignment_from_wfi_euler(
        self, wfi_ra, wfi_dec, wfi_pa, meas_ra, meas_dec, meas_north_angle
    ):
        """Tier 1: Uses the fully solved Telescope Boresight Euler angles directly from WFI alignment."""
        att_matrix = rotations.attitude_matrix(0.0, 0.0, wfi_ra, wfi_dec, wfi_pa)
        return self._apply_kinematics(att_matrix, meas_ra, meas_dec, meas_north_angle)

    # --- TIER 2: FGS Telemetry Only ---
    def compute_alignment_from_telemetry(
        self, q1, q2, q3, q4, meas_ra, meas_dec, meas_north_angle
    ):
        """Tier 2: Uses time-averaged FGS FACE quaternions (SCF_AC_EST_FGS_qbr)."""
        # CRITICAL FIX: Telemetry quaternions are Body-to-Sky. Do NOT transpose (.T).
        att_matrix = R.from_quat([q1, q2, q3, q4]).as_matrix()
        return self._apply_kinematics(att_matrix, meas_ra, meas_dec, meas_north_angle)

    # --- TIER 3: Pre-Flight / Fallback ---
    def compute_alignment_from_commanded(
        self, cmd_ra, cmd_dec, cmd_pa, meas_ra, meas_dec, meas_north_angle
    ):
        """Tier 3: Uses FGS Commanded state anchored at the CGI baseline."""
        att_matrix = rotations.attitude_matrix(
            self.cgi_v2, self.cgi_v3, cmd_ra, cmd_dec, cmd_pa
        )
        return self._apply_kinematics(att_matrix, meas_ra, meas_dec, meas_north_angle)


# ==========================================
# MODULE C: MATRIX & YAML EXPORTER
# ==========================================
def calculate_cgi_to_body_matrix(
    v2_ref: float, v3_ref: float, v3idlyangle: float
) -> np.ndarray:
    """
    Computes the 3x3 CGI-to-Body alignment matrix safely by evaluating
    3D Cartesian unit vectors of the CGI axes in the Body frame.
    """
    # +Z Axis (The Boresight vector)
    v2_rad = np.deg2rad(v2_ref / 3600.0)
    v3_rad = np.deg2rad(v3_ref / 3600.0)
    z_body = np.array(
        [
            np.cos(v3_rad) * np.cos(v2_rad),
            np.cos(v3_rad) * np.sin(v2_rad),
            np.sin(v3_rad),
        ]
    )

    # Ideal Coordinate frame angles (IdlX/IdlY) are planar.
    # Extract the rotation around the boresight.
    theta_rad = np.deg2rad(v3idlyangle)

    # Construct standard Body frame basis vectors
    v3_axis = np.array([0, 0, 1])

    # Calculate local +Y axis (Rotated theta from V3 axis around the boresight)
    # Using Rodrigues' rotation formula
    y_body = (
        v3_axis * np.cos(theta_rad)
        + np.cross(z_body, v3_axis) * np.sin(theta_rad)
        + z_body * np.dot(z_body, v3_axis) * (1 - np.cos(theta_rad))
    )
    y_body /= np.linalg.norm(y_body)

    # Calculate local +X axis (Orthogonal to Z and Y)
    x_body = np.cross(y_body, z_body)
    x_body /= np.linalg.norm(x_body)

    # The columns of this matrix are the CGI X, Y, Z axes expressed in the Body frame
    return np.column_stack((x_body, y_body, z_body))


def export_cgi_alignment_to_yaml(
    calibrated_siaf_params: dict, output_prefix: str = "roman_cgi_updates"
) -> str:
    """Exports the calibrated CGI SIAF parameters to a YAML file."""
    current_date = datetime.now().strftime("%Y%m%d")
    output_filename = f"{output_prefix}_{current_date}.yml"
    yaml_lines = [f"version: '{current_date}'"]

    for aperture_name in sorted(calibrated_siaf_params.keys()):
        yaml_lines.append(f"{aperture_name}:")
        params = calibrated_siaf_params[aperture_name]
        yaml_lines.append(f"  V2Ref: {params['V2Ref']:.3f}")
        yaml_lines.append(f"  V3Ref: {params['V3Ref']:.3f}")
        yaml_lines.append(f"  V3IdlYAngle: {params['V3IdlYAngle']:.5f}")
        yaml_lines.append(f"  XSciScale: {params['XSciScale']:.6f}")
        yaml_lines.append(f"  YSciScale: {params['YSciScale']:.6f}")

    with open(output_filename, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"\nSUCCESS: Exported calibrated SIAF to {output_filename}")
    return output_filename


# ==========================================
# COMMISSIONING MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("  ROMAN CGI ALIGNMENT COMMISSIONING PIPELINE       ")
    print("==================================================\n")
    print(f"Processing target DRP product: {INPUT_FITS_FILE}...")

    # Step 1: Ingest Measured Data from CGI DRP
    fits_data = ingest_cgi_fits(INPUT_FITS_FILE)

    # Step 2: Initialize Solver & Select Execution Tier
    solver = CGIAlignmentSolver(BASELINE_CGI_V2, BASELINE_CGI_V3)

    if OPERATION_MODE == "TIER1_WFI_UNIFIED":
        print("-> Executing Tier 1: Unified WFI Boresight Solution")
        new_v2, new_v3, new_v3idlyangle = solver.compute_alignment_from_wfi_euler(
            wfi_ra=SOLVED_WFI_RA,
            wfi_dec=SOLVED_WFI_DEC,
            wfi_pa=SOLVED_WFI_PA,
            meas_ra=fits_data.ra_deg,
            meas_dec=fits_data.dec_deg,
            meas_north_angle=fits_data.north_angle_deg,
        )

    elif OPERATION_MODE == "TIER2_FACE_TELEMETRY":
        print("-> Executing Tier 2: FGS FACE Telemetry Only")
        new_v2, new_v3, new_v3idlyangle = solver.compute_alignment_from_telemetry(
            q1=TELEMETRY_Q1,
            q2=TELEMETRY_Q2,
            q3=TELEMETRY_Q3,
            q4=TELEMETRY_Q4,
            meas_ra=fits_data.ra_deg,
            meas_dec=fits_data.dec_deg,
            meas_north_angle=fits_data.north_angle_deg,
        )

    else:
        print("-> Executing Tier 3: Commanded Fallback Mode")
        new_v2, new_v3, new_v3idlyangle = solver.compute_alignment_from_commanded(
            cmd_ra=CMD_RA,
            cmd_dec=CMD_DEC,
            cmd_pa=CMD_PA,
            meas_ra=fits_data.ra_deg,
            meas_dec=fits_data.dec_deg,
            meas_north_angle=fits_data.north_angle_deg,
        )

    # Step 3: Calculate CGI-to-Body Matrix
    cgi_to_body_matrix = calculate_cgi_to_body_matrix(new_v2, new_v3, new_v3idlyangle)

    print("\n==================================================")
    print("           CGI BORESIGHT ALIGNMENT RESULTS        ")
    print("==================================================")
    print(f"Old Baseline V2/V3 : {BASELINE_CGI_V2:.3f}, {BASELINE_CGI_V3:.3f}")
    print(f"New V2Ref          : {new_v2:.3f}")
    print(f"New V3Ref          : {new_v3:.3f}")
    print(f"New V3IdlYAngle    : {new_v3idlyangle:.5f}")
    print("\nCGI-to-Body Alignment Matrix (BAM equivalent):")
    for row in cgi_to_body_matrix:
        print(f"  [{row[0]:>11.8f}, {row[1]:>11.8f}, {row[2]:>11.8f}]")
    print("==================================================")

    # Step 4: Package and Export
    scale_arcsec = fits_data.plate_scale_mas / 1000.0
    calibrated_params = {
        "CGI_CEN": {
            "V2Ref": new_v2,
            "V3Ref": new_v3,
            "V3IdlYAngle": new_v3idlyangle,
            "XSciScale": scale_arcsec,
            "YSciScale": scale_arcsec,
        }
    }

    export_cgi_alignment_to_yaml(calibrated_params, OUTPUT_PREFIX)
