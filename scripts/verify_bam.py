#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
from scipy.spatial.transform import Rotation as R

def read_bam_file(filepath):
    """Parses a standard ACS BAM text file to extract the quaternion and metadata."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BAM file not found: {filepath}")
        
    q = [0.0, 0.0, 0.0, 0.0]
    instrument = "UNKNOWN"
    gen_time = "UNKNOWN"
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("# Generated"):
                gen_time = line.replace("# Generated", "").strip()
            elif line.startswith("# Body to") and "Alignment" in line:
                instrument = line.split("Body to")[1].split("Alignment")[0].strip()
            
            # Match the Qb[X] array assignment
            match = re.search(r'Qb\[(\d)\]\s+([-\d\.e]+)', line)
            if match:
                idx = int(match.group(1)) - 1
                q[idx] = float(match.group(2))
                
    # Normalize just in case of microscopic text truncation errors
    q = np.array(q)
    q /= np.linalg.norm(q)
    
    return q, instrument, gen_time

def extract_siaf_angles(q_array):
    """
    Reverse-engineers the ACS flight software Euler sequence 
    (M_X2Z @ Rx(ya) @ Ry(bz) @ Rz(by)) to extract SIAF angles.
    """
    m_b2inst = R.from_quat(q_array).inv().as_matrix()
    m_x2z = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    
    a = m_x2z.T @ m_b2inst
    
    # Extract Euler angles in radians
    bz_rad = np.arcsin(np.clip(a[0, 2], -1.0, 1.0))
    by_rad = np.arctan2(a[0, 1], a[0, 0])
    ya_rad = np.arctan2(-a[1, 2], a[2, 2])
    
    # Convert to SIAF conventions (arcseconds and degrees)
    v2_arcsec = np.rad2deg(by_rad) * 3600.0
    v3_arcsec = np.rad2deg(bz_rad) * 3600.0
    v3idlyangle = np.rad2deg(ya_rad)
    
    # Standardize Zeta angle to [-180, 180] boundary
    v3idlyangle = (v3idlyangle + 180.0) % 360.0 - 180.0
    
    return v2_arcsec, v3_arcsec, v3idlyangle

def main():
    parser = argparse.ArgumentParser(description="Verify and compare two Roman ACS BAM files.")
    parser.add_argument("old_bam", help="Path to the Old/Baseline BAM file.")
    parser.add_argument("new_bam", help="Path to the New/Updated BAM file.")
    args = parser.parse_args()

    # 1. Parse Files
    q_old, inst_old, time_old = read_bam_file(args.old_bam)
    q_new, inst_new, time_new = read_bam_file(args.new_bam)
    
    if inst_old != inst_new:
        print(f"WARNING: Instrument mismatch. Comparing {inst_old} with {inst_new}.")
        
    # 2. Extract SIAF Parameters
    v2_old, v3_old, zeta_old = extract_siaf_angles(q_old)
    v2_new, v3_new, zeta_new = extract_siaf_angles(q_new)
    
    # 3. Calculate Projected Focal Plane Shift
    delta_v2 = v2_new - v2_old
    delta_v3 = v3_new - v3_old
    delta_zeta = zeta_new - zeta_old
    
    focal_plane_shift_arcsec = np.hypot(delta_v2, delta_v3)
    focal_plane_shift_arcmin = focal_plane_shift_arcsec/60.0
    focal_plane_shift_pixels = focal_plane_shift_arcsec/0.11

    # 4. Print Diagnostics
    print("\n=============================================================")
    print(f"                   BAM VERIFICATION REPORT                   ")
    print( "=============================================================")
    print(f"Instrument : {inst_new}")
    print(f"Old File   : {os.path.basename(args.old_bam)}")
    print(f"New File   : {os.path.basename(args.new_bam)}")
    print("-------------------------------------------------------------")
    print("SIAF COORDINATE EXTRACTION:")
    print(f"              {'OLD':>12} | {'NEW':>12} | {'DELTA':>10}")
    print(f"V2Ref       : {v2_old:12.3f} | {v2_new:12.3f} | {(v2_new - v2_old):+10.3f} arcsec")
    print(f"V3Ref       : {v3_old:12.3f} | {v3_new:12.3f} | {(v3_new - v3_old):+10.3f} arcsec")
    print(f"V3IdlYAngle : {zeta_old:12.5f} | {zeta_new:12.5f} | {(zeta_new - zeta_old):+10.5f} deg")
    print("-------------------------------------------------------------")
    print("SPATIAL SHIFT IMPLIED ON THE FOCAL PLANE:")
    print(f"{focal_plane_shift_arcsec:.2f} arcsec ({focal_plane_shift_arcmin:.2f} arcmin) ≃ {focal_plane_shift_pixels:.2f} WFI pixels")
    print("=============================================================\n")

if __name__ == "__main__":
    main()
