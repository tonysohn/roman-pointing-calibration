#!/usr/bin/env python3
"""
apply_manual_bam_shift.py

A rapid-response tool to manually inject measured X, Y, and Roll shifts
from the WFI Science frame directly into the FGS Boresight Alignment Matrix (BAM).
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R


def apply_shifts_to_bam(q_b2fgs_old, dx_user, dy_user, droll_user):
    """
    Applies user-measured science frame shifts to the BAM quaternion.
    """
    # 1. Sign Convention Correction
    # User input is (Expected - Measured). To correct the model, we need
    # to move FROM Expected TO Measured, which is (Measured - Expected).
    # Therefore, we invert the user's inputs to get the true correction delta.
    dx_sci = -dx_user
    dy_sci = -dy_user
    droll_sci = -droll_user

    # 2. Coordinate Transformation (Science Frame -> FGS Frame)
    # Science +X is aligned with FGS +X
    # Science +Y is flipped from FGS +Y
    dx_fgs = dx_sci
    dy_fgs = -dy_sci

    # A flipped Y-axis mirrors the XY plane, reversing the direction of Roll (Z-axis)
    droll_fgs = -droll_sci

    # 3. Convert arcseconds to degrees for the rotation solver
    dx_deg = dx_fgs / 3600.0
    dy_deg = dy_fgs / 3600.0
    droll_deg = droll_fgs / 3600.0

    # 4. Construct the Correction Rotation (FGS_old -> FGS_new)
    # Assuming Z is the boresight:
    # A positive shift in FGS X requires a rotation around the FGS Y-axis
    # A positive shift in FGS Y requires a negative rotation around the FGS X-axis
    rot_x = -dy_deg
    rot_y = dx_deg
    rot_z = droll_deg

    # Create the correction quaternion using intrinsic Euler angles
    r_corr = R.from_euler("XYZ", [rot_x, rot_y, rot_z], degrees=True)

    # 5. Apply the correction to the original BAM
    # Math: q_new = q_correction * q_old
    r_old = R.from_quat(q_b2fgs_old)
    r_new = r_corr * r_old

    return r_new.as_quat(), dx_fgs, dy_fgs, droll_fgs


def main():
    parser = argparse.ArgumentParser(
        description="Quickly update the BAM quaternion using science image shifts.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--qbj",
        type=float,
        nargs=4,
        default=[
            -0.1859673417539929,
            0.6837984564491885,
            -0.1800546332580956,
            0.6822141509826322,
        ],
        help="Input Base BAM Quaternion [x y z w]. Defaults to nominal pre-flight BAM.",
    )

    # The help text explicitly enforces your x2 - x1 convention
    parser.add_argument(
        "--dx_arcsec",
        type=float,
        required=True,
        help="Shift in Science X.\nDEFINITION: Expected - Measured (x2 - x1)",
    )
    parser.add_argument(
        "--dy_arcsec",
        type=float,
        required=True,
        help="Shift in Science Y.\nDEFINITION: Expected - Measured (y2 - y1)",
    )
    parser.add_argument(
        "--droll_arcsec",
        type=float,
        default=0.0,
        help="Roll shift in Science XY plane.\nDEFINITION: Expected - Measured",
    )

    args = parser.parse_args()
    q_old = np.array(args.qbj)

    # Run the math solver
    q_new, dx_fgs, dy_fgs, droll_fgs = apply_shifts_to_bam(
        q_old, args.dx_arcsec, args.dy_arcsec, args.droll_arcsec
    )

    # =========================================================
    # TELEMETRY REPORT OUTPUT
    # =========================================================
    print("\n===========================================================")
    print("                   MANUAL BAM UPDATE                       ")
    print("===========================================================")
    print("INPUT DEFINITION: (Expected - Measured)")
    print(
        f"  User Input    : ΔX = {args.dx_arcsec:8.3f}, ΔY = {args.dy_arcsec:8.3f}, ΔRoll = {args.droll_arcsec:8.3f}"
    )

    print("\nAPPLIED CORRECTIONS (Measured - Expected):")
    print(
        f"  FGS Frame     : ΔX = {dx_fgs:8.3f}, ΔY = {dy_fgs:8.3f}, ΔRoll = {droll_fgs:8.3f}"
    )
    print("\nQUATERNION COMPARISON [x, y, z, w]:")
    print("           OLD (Input)                 NEW (Output)      ")
    print("-----------------------------------------------------------")
    print(f"  x: {q_old[0]: 16.12f}     ->  {q_new[0]: 16.12f}")
    print(f"  y: {q_old[1]: 16.12f}     ->  {q_new[1]: 16.12f}")
    print(f"  z: {q_old[2]: 16.12f}     ->  {q_new[2]: 16.12f}")
    print(f"  w: {q_old[3]: 16.12f}     ->  {q_new[3]: 16.12f}")
    print("===========================================================\n")

    print("Copy/Paste Format for Flight Software:")
    print(f"[{q_new[0]:.12f}, {q_new[1]:.12f}, {q_new[2]:.12f}, {q_new[3]:.12f}]\n")


if __name__ == "__main__":
    main()
