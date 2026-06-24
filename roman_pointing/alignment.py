import csv
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pysiaf
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from tqdm import tqdm

from .diagnostics import generate_alignment_diagnostics


def _robust_cross_match(
    v2_obs,
    v3_obs,
    v2_cat,
    v3_cat,
    flux_obs=None,
    mag_cat=None,
    broad_tol=10.0,
    strict_tol=1.0,
    bin_size=0.5,
):

    if len(v2_obs) == 0 or len(v2_cat) == 0:
        return (
            np.zeros_like(v2_obs, dtype=bool),
            np.zeros_like(v2_obs, dtype=int),
            0.0,
            0.0,
        )

    min_v2, max_v2 = np.min(v2_obs) - broad_tol, np.max(v2_obs) + broad_tol
    min_v3, max_v3 = np.min(v3_obs) - broad_tol, np.max(v3_obs) + broad_tol

    in_bounds = (
        (v2_cat >= min_v2)
        & (v2_cat <= max_v2)
        & (v3_cat >= min_v3)
        & (v3_cat <= max_v3)
    )
    v2_c_filt = v2_cat[in_bounds]
    v3_c_filt = v3_cat[in_bounds]
    orig_indices = np.where(in_bounds)[0]

    if len(v2_c_filt) == 0:
        return (
            np.zeros_like(v2_obs, dtype=bool),
            np.zeros_like(v2_obs, dtype=int),
            0.0,
            0.0,
        )

    top_n_obs = min(40, len(v2_obs))
    top_n_cat = min(400, len(v2_c_filt))

    v2_obs_arr, v3_obs_arr = np.asarray(v2_obs), np.asarray(v3_obs)

    if flux_obs is not None and mag_cat is not None:
        mag_c_filt = np.asarray(mag_cat)[in_bounds]

        # FIX: Force missing/0.0 magnitudes to 99.0 so they sink to the bottom of the rank
        mag_c_filt = np.where(
            np.isnan(mag_c_filt) | (mag_c_filt == 0.0), 99.0, mag_c_filt
        )

        obs_sort = np.argsort(np.asarray(flux_obs))[::-1]
        v2_o_sub, v3_o_sub = (
            v2_obs_arr[obs_sort][:top_n_obs],
            v3_obs_arr[obs_sort][:top_n_obs],
        )
        cat_sort = np.argsort(mag_c_filt)
        v2_r_sub, v3_r_sub = (
            v2_c_filt[cat_sort][:top_n_cat],
            v3_c_filt[cat_sort][:top_n_cat],
        )
    else:
        v2_o_sub, v3_o_sub = v2_obs_arr[:top_n_obs], v3_obs_arr[:top_n_obs]
        v2_r_sub, v3_r_sub = v2_c_filt[:top_n_cat], v3_c_filt[:top_n_cat]

    dv2_matrix = v2_r_sub[np.newaxis, :] - v2_o_sub[:, np.newaxis]
    dv3_matrix = v3_r_sub[np.newaxis, :] - v3_o_sub[:, np.newaxis]

    valid_diffs = (np.abs(dv2_matrix) <= broad_tol) & (np.abs(dv3_matrix) <= broad_tol)
    dv2_flat, dv3_flat = dv2_matrix[valid_diffs], dv3_matrix[valid_diffs]

    dv2_bulk, dv3_bulk = 0.0, 0.0
    if len(dv2_flat) > 0:
        bins = np.arange(-broad_tol, broad_tol + bin_size, bin_size)
        H, xedges, yedges = np.histogram2d(dv2_flat, dv3_flat, bins=(bins, bins))

        if np.max(H) >= 5:
            ix, iy = np.unravel_index(np.argmax(H), H.shape)
            if 0 < ix < H.shape[0] - 1 and 0 < iy < H.shape[1] - 1:
                patch = H[ix - 1 : ix + 2, iy - 1 : iy + 2]
                x_grid, y_grid = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
                mass = np.sum(patch)
                dx_cent = np.sum(patch * x_grid) / mass if mass > 0 else 0
                dy_cent = np.sum(patch * y_grid) / mass if mass > 0 else 0
            else:
                dx_cent, dy_cent = 0.0, 0.0
            dv2_bulk = xedges[ix] + (bin_size / 2.0) + (dx_cent * bin_size)
            dv3_bulk = yedges[iy] + (bin_size / 2.0) + (dy_cent * bin_size)

    ref_coords = np.column_stack([v2_c_filt, v3_c_filt])
    tree = cKDTree(ref_coords)
    obs_coords_shifted = np.column_stack([v2_obs_arr + dv2_bulk, v3_obs_arr + dv3_bulk])

    dist_strict, idx_strict_filt = tree.query(
        obs_coords_shifted, distance_upper_bound=strict_tol
    )
    valid_strict = dist_strict < strict_tol

    idx_strict_full = np.zeros_like(idx_strict_filt)
    idx_strict_full[valid_strict] = orig_indices[idx_strict_filt[valid_strict]]

    return valid_strict, idx_strict_full, dv2_bulk, dv3_bulk


def _attitude_residuals(
    delta_params, base_ra, base_dec, base_pa, ra_cat, dec_cat, v2_obs, v3_obs
):
    """
    Objective function for the global Levenberg-Marquardt attitude optimizer.
    ...
    """

    # NOTE: delta_params must be treated as physical sky arcseconds
    cos_dec = np.cos(np.deg2rad(base_dec))
    d_ra_deg = (delta_params[0] / 3600.0) / cos_dec
    d_dec_deg = delta_params[1] / 3600.0
    d_pa_deg = delta_params[2] / 3600.0

    att_matrix = pysiaf.utils.rotations.attitude(
        0, 0, base_ra + d_ra_deg, base_dec + d_dec_deg, base_pa + d_pa_deg
    )
    v2_calc, v3_calc = pysiaf.utils.rotations.getv2v3(
        att_matrix, np.asarray(ra_cat), np.asarray(dec_cat)
    )

    # --- THE SPHERICAL FIX ---
    # 1. Convert reference V3 (arcsec) to radians
    v3_rad = np.deg2rad(v3_calc / 3600.0)

    # 2. Apply the spherical cosine correction to the V2 difference
    dv2 = (v2_calc - v2_obs) * np.cos(v3_rad)
    dv3 = v3_calc - v3_obs

    return np.concatenate([dv2, dv3])


def _fit_sca_alignment(
    v2_obs, v3_obs, v2_cat, v3_cat, v2_fiducial, v3_fiducial, sigma_clip=3.0, max_iter=5
):
    """
    Computes the affine transformation (shift and rotation) for a single SCA.

    Uses an iterative least-squares solver with sigma clipping to find the optimal
    dV2, dV3, and dTheta required to align the observed stellar coordinates with
    the astrometric reference catalog.

    Parameters
    ----------
    v2_obs : array_like
        Observed V2 coordinates of detected stars (arcsec).
    v3_obs : array_like
        Observed V3 coordinates of detected stars (arcsec).
    v2_cat : array_like
        Reference V2 coordinates from the astrometric catalog (arcsec).
    v3_cat : array_like
        Reference V3 coordinates from the astrometric catalog (arcsec).
    v2_fiducial : float
        Nominal V2Ref of the SCA aperture to serve as the rotation center.
    v3_fiducial : float
        Nominal V3Ref of the SCA aperture to serve as the rotation center.
    sigma_clip : float, optional
        Threshold for rejecting outlier matches. Default is 3.0.
    max_iter : int, optional
        Maximum number of clipping iterations. Default is 5.

    Returns
    -------
    dv2 : float
        Calculated V2 shift (arcsec).
    dv3 : float
        Calculated V3 shift (arcsec).
    d_theta_deg : float
        Calculated rotation angle (degrees).
    dv2_err : float
        1-sigma formal uncertainty in the V2 shift.
    dv3_err : float
        1-sigma formal uncertainty in the V3 shift.
    theta_err_deg : float
        1-sigma formal uncertainty in the rotation angle.
    """
    x_obs = np.asarray(v2_obs) - v2_fiducial
    y_obs = np.asarray(v3_obs) - v3_fiducial
    x_ref = np.asarray(v2_cat) - v2_fiducial
    y_ref = np.asarray(v3_cat) - v3_fiducial

    valid = np.ones(len(x_obs), dtype=bool)

    # Iterative Sigma Clipping Loop
    for _ in range(max_iter):
        design_matrix = np.column_stack(
            (x_obs[valid], y_obs[valid], np.ones(np.sum(valid)))
        )
        coeffs_x, _, _, _ = np.linalg.lstsq(design_matrix, x_ref[valid], rcond=None)
        coeffs_y, _, _, _ = np.linalg.lstsq(design_matrix, y_ref[valid], rcond=None)

        x_calc = design_matrix @ coeffs_x
        y_calc = design_matrix @ coeffs_y

        # Calculate radial residuals
        dist = np.hypot(x_ref[valid] - x_calc, y_ref[valid] - y_calc)
        std_dist = np.std(dist)

        if std_dist == 0:
            break

        keep = dist < (sigma_clip * std_dist)
        if np.all(keep):
            break

        # ==============================================================
        # SAFETY CATCH 1: Do not over-clip.
        # We need at least 3 points to solve an affine transformation.
        # ==============================================================
        if np.sum(keep) < 3:
            break  # Stop clipping, retain the current 'valid' mask

        # Update valid mask
        valid_indices = np.where(valid)[0]
        valid[valid_indices[~keep]] = False

    # Final definitive fit with the safe mask
    design_matrix = np.column_stack(
        (x_obs[valid], y_obs[valid], np.ones(np.sum(valid)))
    )
    coeffs_x, _, _, _ = np.linalg.lstsq(design_matrix, x_ref[valid], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(design_matrix, y_ref[valid], rcond=None)

    x_calc = design_matrix @ coeffs_x
    y_calc = design_matrix @ coeffs_y

    # ==============================================================
    # SAFETY CATCH 2: Handle 0 degrees of freedom (exactly 3 stars)
    # ==============================================================
    dof = np.sum(valid) - 3
    if dof > 0:
        MSE_x = np.sum((x_ref[valid] - x_calc) ** 2) / dof
        MSE_y = np.sum((y_ref[valid] - y_calc) ** 2) / dof
    else:
        # If exactly 3 stars remain, the fit is perfectly constrained (MSE = 0)
        MSE_x, MSE_y = 0.0, 0.0

    # ==============================================================
    # SAFETY CATCH 3: Collinear points causing singular matrix
    # ==============================================================
    try:
        cov_matrix = np.linalg.inv(design_matrix.T @ design_matrix)
        err_x = np.sqrt(np.diag(cov_matrix) * MSE_x)
        err_y = np.sqrt(np.diag(cov_matrix) * MSE_y)
    except np.linalg.LinAlgError:
        # Fallback if the remaining stars are arranged in a perfectly straight line
        err_x = np.array([np.nan, np.nan, np.nan])
        err_y = np.array([np.nan, np.nan, np.nan])

    dv2, dv2_err = coeffs_x[2], err_x[2]
    dv3, dv3_err = coeffs_y[2], err_y[2]

    # Extract Rotation
    sin_theta = (coeffs_y[0] - coeffs_x[1]) / 2.0
    d_theta_deg = np.degrees(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))
    theta_err_deg = np.degrees(np.sqrt(err_y[0] ** 2 + err_x[1] ** 2) / 2.0)

    # ------------------------------------------------------------------------------
    # Extract Scale (Magnification) and Skew
    # A perfect, un-distorted match would have coeffs_x[0] = 1.0 and coeffs_y[1] = 1.0
    scale_x = coeffs_x[0]
    scale_y = coeffs_y[1]
    skew = (coeffs_x[1] + coeffs_y[0]) / 2.0

    # Calculate final RMS of the fit (in mas)
    final_rms_mas = (
        np.std(np.hypot(x_ref[valid] - x_calc, y_ref[valid] - y_calc)) * 1000.0
    )
    # ------------------------------------------------------------------------------
    # return dv2, dv3, d_theta_deg, dv2_err, dv3_err, theta_err_deg

    return dv2, dv3, d_theta_deg, scale_x, scale_y, skew, final_rms_mas


def align_wfi(
    phot_catalogs,
    ref_catalog,
    pointing_info,
    user_offsets=None,
    max_iterations=5,
    debug=False,
):
    # Suppress the specific Gaia DR4 warning
    warnings.filterwarnings("ignore", message=".*Gaia archive is in evolution.*")

    roman_siaf = pysiaf.Siaf("Roman")
    user_offsets = user_offsets or {}
    cos_dec = np.cos(np.deg2rad(pointing_info["DEC_V1"]))

    current_ra = pointing_info["RA_V1"] + (
        (user_offsets.get("d_ra_arcsec", 0.0) / 3600.0) / cos_dec
    )
    current_dec = pointing_info["DEC_V1"] + (
        user_offsets.get("d_dec_arcsec", 0.0) / 3600.0
    )
    current_pa = pointing_info["PA_V3"] + (
        user_offsets.get("d_pa_arcsec", 0.0) / 3600.0
    )

    # Extract Gaia magnitudes ONCE before any loops
    mag_gaia = (
        np.asarray(ref_catalog["phot_g_mean_mag"])
        if "phot_g_mean_mag" in ref_catalog.colnames
        else None
    )

    # =========================================================================
    # MACRO-ALIGNMENT LOOP
    # =========================================================================
    iteration_history = []
    pbar = tqdm(range(max_iterations), desc="Solving Global Attitude")
    for i in pbar:
        att_matrix = pysiaf.utils.rotations.attitude(
            0, 0, current_ra, current_dec, current_pa
        )
        v2_gaia, v3_gaia = pysiaf.utils.rotations.getv2v3(
            att_matrix,
            np.asarray(ref_catalog["ra_epoch"]),
            np.asarray(ref_catalog["dec_epoch"]),
        )

        global_v2_obs, global_v3_obs, global_ra_ref, global_dec_ref = [], [], [], []
        valid_scas_found = 0

        if debug and i == 0:
            print(f"\n--- DEBUG LOG: Iteration {i} ---")

        for aper_name, catalog in phot_catalogs.items():
            aper = roman_siaf[aper_name]

            if not aper:
                continue

            v2_obs, v3_obs = aper.sci_to_tel(catalog["x"] + 1, catalog["y"] + 1)
            flux_obs = catalog["flux"] if "flux" in catalog.colnames else None

            tol = 20.0 if i == 0 else 2.0

            valid, idx, dv2_bulk, dv3_bulk = _robust_cross_match(
                v2_obs,
                v3_obs,
                v2_gaia,
                v3_gaia,
                flux_obs=flux_obs,
                mag_cat=mag_gaia,
                broad_tol=tol,
                strict_tol=1.0,
            )

            if debug and i == 0:
                print(
                    f'  [{aper_name}] Histogram Shift -> dV2: {dv2_bulk:6.2f}", dV3: {dv3_bulk:6.2f}" | Strict Matches: {np.sum(valid)}'
                )

            if np.sum(valid) > 5:
                valid_scas_found += 1

                # --- SPEED HACK: Sub-sample to 50 stars max per SCA ---
                valid_indices = np.where(valid)[0][:50]

                global_v2_obs.extend(v2_obs[valid_indices])
                global_v3_obs.extend(v3_obs[valid_indices])
                global_ra_ref.extend(ref_catalog["ra_epoch"][idx[valid_indices]])
                global_dec_ref.extend(ref_catalog["dec_epoch"][idx[valid_indices]])

        print(f"  -> Iteration {i + 1}: Matched {valid_scas_found}/18 SCAs.")

        bounds = ([-60.0, -60.0, -1800.0], [60.0, 60.0, 1800.0])
        global_result = least_squares(
            _attitude_residuals,
            [0.0, 0.0, 0.0],
            args=(
                current_ra,
                current_dec,
                current_pa,
                np.asarray(global_ra_ref),
                np.asarray(global_dec_ref),
                global_v2_obs,
                global_v3_obs,
            ),
            method="trf",
            loss="linear" if i == 0 else "soft_l1",
            f_scale=1.0,
            bounds=bounds,
            ftol=1e-5,
            xtol=1e-5,
        )

        d_ra, d_dec, d_pa = global_result.x

        if debug and i == 0:
            print(
                f'  [SOLVER] Raw Optimizer Output -> dRA: {d_ra:6.2f}", dDec: {d_dec:6.2f}", dPA: {d_pa:6.2f}"'
            )
            print("--------------------------------\n")

        current_ra += (d_ra / 3600.0) / cos_dec
        current_dec += d_dec / 3600.0
        current_pa = (current_pa + d_pa / 3600.0) % 360.0

        step_mag_arcsec = np.sqrt(d_ra**2 + d_dec**2)
        iteration_history.append(step_mag_arcsec)

        pbar.set_postfix({"Res_Mag_arcsec": f"{step_mag_arcsec:.3f}"})
        if np.sqrt(d_ra**2 + d_dec**2 + d_pa**2) < 0.001:
            break

    # =========================================================================
    # POST-OPTIMIZATION & FINAL DIAGNOSTIC SUITE
    # =========================================================================
    diag_dir = "diagnostics"
    os.makedirs(diag_dir, exist_ok=True)

    J = global_result.jac
    cov_global = np.linalg.inv(J.T @ J)
    MSE_global = (global_result.fun**2).mean()
    att_err_arcsec = np.sqrt(np.diagonal(cov_global) * MSE_global)

    locked_att_matrix = pysiaf.utils.rotations.attitude(
        0, 0, current_ra, current_dec, current_pa
    )
    v2_gaia_locked, v3_gaia_locked = pysiaf.utils.rotations.getv2v3(
        locked_att_matrix,
        np.asarray(ref_catalog["ra_epoch"]),
        np.asarray(ref_catalog["dec_epoch"]),
    )

    calibrated_siaf_params = {}
    summary_log_data = []
    matched_pairs_log_data = []
    all_matched_flux = []
    all_matched_mag = []

    # Final Local SCA Alignment Loop
    for aper_name, catalog in phot_catalogs.items():
        aper = roman_siaf[aper_name]
        v2_obs, v3_obs = aper.sci_to_tel(catalog["x"] + 1, catalog["y"] + 1)
        flux_obs = catalog["flux"] if "flux" in catalog.colnames else None

        valid, idx, dv2_bulk, dv3_bulk = _robust_cross_match(
            v2_obs,
            v3_obs,
            v2_gaia_locked,
            v3_gaia_locked,
            flux_obs=flux_obs,
            mag_cat=mag_gaia,
            broad_tol=5.0,
            strict_tol=1.0,
        )

        num_matched = np.sum(valid)
        summary_log_data.append(
            [
                aper_name,
                len(v2_obs),
                num_matched,
                round(dv2_bulk, 3),
                round(dv3_bulk, 3),
            ]
        )

        if num_matched < 5:
            continue

        valid_indices = np.where(valid)[0]

        # --- DIAGNOSTIC: DS9 Region File (.reg) ---
        reg_filename = os.path.join(diag_dir, f"matched_gaia_{aper_name}.reg")
        with open(reg_filename, "w") as f_reg:
            f_reg.write("# Region file format: DS9 version 4.1\n")
            f_reg.write(
                'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
            )
            f_reg.write("image\n")
            x_g, y_g = aper.tel_to_sci(
                v2_gaia_locked[idx[valid]], v3_gaia_locked[idx[valid]]
            )
            for xi, yi in zip(x_g, y_g):
                f_reg.write(f"circle({xi:.2f},{yi:.2f},5)\n")

        # --- DIAGNOSTIC: Constellation Overlay Plot ---
        plt.figure(figsize=(10, 10))
        plt.scatter(
            v2_gaia_locked[idx[valid]],
            v3_gaia_locked[idx[valid]],
            c="blue",
            marker="+",
            s=60,
            label="Gaia Ref",
            alpha=0.6,
        )
        plt.scatter(
            np.asarray(v2_obs)[valid] + dv2_bulk,
            np.asarray(v3_obs)[valid] + dv3_bulk,
            facecolors="none",
            edgecolors="red",
            s=60,
            label="Roman Obs (Shifted)",
        )
        for v_i in valid_indices:
            plt.plot(
                [v2_obs[v_i] + dv2_bulk, v2_gaia_locked[idx[v_i]]],
                [v3_obs[v_i] + dv3_bulk, v3_gaia_locked[idx[v_i]]],
                "g-",
                alpha=0.4,
            )
        plt.title(f"Constellation Match: {aper_name}\n({num_matched} stars matched)")
        plt.xlabel("V2 (arcsec)")
        plt.ylabel("V3 (arcsec)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(diag_dir, f"constellation_{aper_name}.png"),
            bbox_inches="tight",
        )
        plt.close()

        # --- Gather data for Matched Pairs Log and Photometry ---
        for v_i in valid_indices:
            ref_idx = idx[v_i]

            # The calibrated observed V2/V3 (what the FGS function expects)
            v2_cal = v2_obs[v_i] + dv2_bulk
            v3_cal = v3_obs[v_i] + dv3_bulk

            # The residuals relative to the locked Gaia positions (for logging)
            res_v2 = v2_cal - v2_gaia_locked[ref_idx]
            res_v3 = v3_cal - v3_gaia_locked[ref_idx]

            f_val = flux_obs[v_i] if flux_obs is not None else np.nan
            m_val = mag_gaia[ref_idx] if mag_gaia is not None else np.nan

            matched_pairs_log_data.append(
                [
                    aper_name,
                    round(catalog["x"][v_i], 2),
                    round(catalog["y"][v_i], 2),
                    round(ref_catalog["ra_epoch"][ref_idx], 6),
                    round(ref_catalog["dec_epoch"][ref_idx], 6),
                    f_val,
                    m_val,
                    round(v2_cal, 4),
                    round(v3_cal, 4),
                    round(res_v2 * 1000, 2),
                    round(res_v3 * 1000, 2),
                ]
            )

            if not np.isnan(f_val) and not np.isnan(m_val) and m_val != 99.0:
                all_matched_flux.append(f_val)
                all_matched_mag.append(m_val)

        # ---------------------------------------------------------------------
        # THE DYNAMIC DISTORTION EVALUATOR
        # ---------------------------------------------------------------------
        dv2, dv3, d_theta, scale_x, scale_y, skew, sca_rms = _fit_sca_alignment(
            v2_obs[valid],
            v3_obs[valid],
            v2_gaia_locked[idx[valid]],
            v3_gaia_locked[idx[valid]],
            aper.V2Ref,
            aper.V3Ref,
        )

        # Retrieve the Roman polynomial coefficient arrays
        poly_coeffs = aper.get_polynomial_coefficients()

        # Initialize our dictionary entry with the standard 3-parameter updates
        calibrated_siaf_params[aper_name] = {
            "V2Ref": aper.V2Ref + dv2,
            "V3Ref": aper.V3Ref + dv3,
            "V3IdlYAngle": aper.V3IdlYAngle + d_theta,
            "Sci2IdlX10": poly_coeffs["Sci2IdlX"][1],
            "Sci2IdlX01": poly_coeffs["Sci2IdlX"][2],
            "Sci2IdlY10": poly_coeffs["Sci2IdlY"][1],
            "Sci2IdlY01": poly_coeffs["Sci2IdlY"][2],
            "Idl2SciX10": poly_coeffs["Idl2SciX"][1],
            "Idl2SciX01": poly_coeffs["Idl2SciX"][2],
            "Idl2SciY10": poly_coeffs["Idl2SciY"][1],
            "Idl2SciY01": poly_coeffs["Idl2SciY"][2],
        }

        # THE METRIC: Did the physical scale or skew diverge from the PySIAF model?
        # A deviation of 5e-5 (50 ppm) creates a >10 mas error at the chip edge.
        scale_error = max(abs(scale_x - 1.0), abs(scale_y - 1.0))
        skew_error = abs(skew)

        if scale_error > 5e-5 or skew_error > 5e-5:
            print(f"\n  -> [{aper_name}] Scale/Skew anomaly detected.")
            print(
                f"     Scale X: {scale_x:.6f}, Scale Y: {scale_y:.6f}, Skew: {skew:.6f}"
            )
            print(f"     Dynamically updating linear SIAF polynomials...")

            # 1. Update the specific linear indices in our extracted arrays
            poly_coeffs["Sci2IdlX"][1] *= scale_x
            poly_coeffs["Sci2IdlX"][2] += skew
            poly_coeffs["Sci2IdlY"][1] += skew
            poly_coeffs["Sci2IdlY"][2] *= scale_y

            # Build the 2x2 forward transformation matrix
            M_forward = np.array(
                [
                    [poly_coeffs["Sci2IdlX"][1], poly_coeffs["Sci2IdlX"][2]],
                    [poly_coeffs["Sci2IdlY"][1], poly_coeffs["Sci2IdlY"][2]],
                ]
            )
            # Invert it to get the backward coefficients
            M_inverse = np.linalg.inv(M_forward)
            poly_coeffs["Idl2SciX"][1] = M_inverse[0, 0]
            poly_coeffs["Idl2SciX"][2] = M_inverse[0, 1]
            poly_coeffs["Idl2SciY"][1] = M_inverse[1, 0]
            poly_coeffs["Idl2SciY"][2] = M_inverse[1, 1]

            # 2. Update the dictionary for YAML export
            calibrated_siaf_params[aper_name].update(
                {
                    "Sci2IdlX10": poly_coeffs["Sci2IdlX"][1],
                    "Sci2IdlX01": poly_coeffs["Sci2IdlX"][2],
                    "Sci2IdlY10": poly_coeffs["Sci2IdlY"][1],
                    "Sci2IdlY01": poly_coeffs["Sci2IdlY"][2],
                    "Idl2SciX10": poly_coeffs["Idl2SciX"][1],
                    "Idl2SciX01": poly_coeffs["Idl2SciX"][2],
                    "Idl2SciY10": poly_coeffs["Idl2SciY"][1],
                    "Idl2SciY01": poly_coeffs["Idl2SciY"][2],
                }
            )

            # 3. Inject the updated arrays back into the PySIAF aperture
            lower_poly_coeffs = {k.lower(): v for k, v in poly_coeffs.items()}
            aper.set_polynomial_coefficients(**lower_poly_coeffs)

            # 4. Re-calculate the observed coordinates with the bent math
            x_idl = np.asarray(v2_obs) - aper.V2Ref
            y_idl = np.asarray(v3_obs) - aper.V3Ref

            v2_obs_fixed = aper.V2Ref + (x_idl * scale_x + y_idl * skew + dv2)
            v3_obs_fixed = aper.V3Ref + (x_idl * skew + y_idl * scale_y + dv3)

            v2_obs = v2_obs_fixed
            v3_obs = v3_obs_fixed

    # =========================================================================
    # DIAGNOSTIC LOG EXPORTS & SUMMARY PLOT
    # =========================================================================
    with open(os.path.join(diag_dir, "alignment_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "SCA_Name",
                "Total_Obs_Stars",
                "Matched_Stars",
                "Bulk_Shift_V2_arcsec",
                "Bulk_Shift_V3_arcsec",
            ]
        )
        writer.writerows(summary_log_data)

    with open(os.path.join(diag_dir, "matched_pairs.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "SCA",
                "X_pixel",
                "Y_pixel",
                "Gaia_RA_deg",
                "Gaia_Dec_deg",
                "Roman_Flux",
                "Gaia_G_Mag",
                "Res_V2_mas",
                "Res_V3_mas",
            ]
        )
        writer.writerows(matched_pairs_log_data)

    if all_matched_flux and all_matched_mag:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            all_matched_mag, np.log10(all_matched_flux), alpha=0.5, s=15, c="purple"
        )
        plt.title("Photometric Sanity Check: All Matched Pairs")
        plt.xlabel("Gaia G-Band Magnitude")
        plt.ylabel("Log10(Roman Instrumental Flux)")
        plt.gca().invert_xaxis()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(diag_dir, "photometry_sanity_check.png"), bbox_inches="tight"
        )
        plt.close()

    print(
        f"\n  -> Diagnostic suite generated in ./{diag_dir}/ (Logs, DS9 .reg files, and PNG overlays)."
    )

    # =========================================================================
    # ATTITUDE RESULTS INITIALIZATION & DIAGNOSTICS
    # =========================================================================
    valid_scas = list(calibrated_siaf_params.keys())

    attitude_results = {
        "RA_V1": current_ra,
        "DEC_V1": current_dec,
        "PA_V3": current_pa,
        "RA_V1_err_arcsec": att_err_arcsec[0],
        "DEC_V1_err_arcsec": att_err_arcsec[1],
        "PA_V3_err_arcsec": att_err_arcsec[2],
    }

    # =========================================================================
    # THE ZERO-MEAN CONSTRAINT (FIXED ANCHOR PRINCIPLE)
    # =========================================================================
    # Calculate the mean residual shift of the 18 SCAs relative to the model.
    if valid_scas:
        mean_dv2 = np.mean(
            [
                calibrated_siaf_params[s]["V2Ref"] - roman_siaf[s].V2Ref
                for s in valid_scas
            ]
        )
        mean_dv3 = np.mean(
            [
                calibrated_siaf_params[s]["V3Ref"] - roman_siaf[s].V3Ref
                for s in valid_scas
            ]
        )
        mean_dtheta = np.mean(
            [
                calibrated_siaf_params[s]["V3IdlYAngle"] - roman_siaf[s].V3IdlYAngle
                for s in valid_scas
            ]
        )

        # 1. APPLY THE CONSTRAINT
        # Subtract the mean from all local SCA updates to pin WFI_CEN.
        for s in valid_scas:
            calibrated_siaf_params[s]["V2Ref"] -= mean_dv2
            calibrated_siaf_params[s]["V3Ref"] -= mean_dv3
            calibrated_siaf_params[s]["V3IdlYAngle"] -= mean_dtheta

        # 2. LOG THE ABSORBED SHIFT
        attitude_results["Residual_Mean_V2_mas"] = mean_dv2 * 1000.0
        attitude_results["Residual_Mean_V3_mas"] = mean_dv3 * 1000.0

        # 3. INJECT THE SHIFT INTO THE SPACECRAFT POINTING
        # Because we removed the shift from the detectors, the spacecraft must
        # move to compensate. We convert the V2/V3 arcsecond shifts back into
        # RA/Dec degrees using the local declination.
        attitude_results["RA_V1"] += (mean_dv2 / 3600.0) / cos_dec
        attitude_results["DEC_V1"] += mean_dv3 / 3600.0
        attitude_results["PA_V3"] += mean_dtheta / 3600.0

    print("\n--- Alignment Diagnostics Summary ---")
    print(
        f"Global Fit RMS Residual: {np.sqrt(np.mean(global_result.fun**2)):.4f} arcsec"
    )
    print(f"Successful SCA Fits: {len(valid_scas)}/18")

    if "Residual_Mean_V2_mas" in attitude_results:
        print(
            f"Detector Plate Mean Shift (V2): {attitude_results['Residual_Mean_V2_mas']:.3f} mas"
        )
        print(
            f"Detector Plate Mean Shift (V3): {attitude_results['Residual_Mean_V3_mas']:.3f} mas"
        )
    print("--------------------------------------\n")

    print("  -> Generating diagnostic plots...")
    generate_alignment_diagnostics(
        matched_pairs_log=matched_pairs_log_data,
        iteration_history=iteration_history,  # e.g., [11.2, 0.49, 0.03, 0.03, 0.04]
        output_dir="./diagnostics",
    )

    return calibrated_siaf_params, attitude_results, matched_pairs_log_data


def export_alignment_to_yaml(calibrated_siaf_params, output_prefix="roman_wfi_updates"):
    """
    Exports the calibrated SIAF parameters to a YAML file for PRD XML conversion.

    Parses the calibrated parameter dictionary and generates a flat, precisely
    formatted YAML file containing V2Ref, V3Ref, and V3IdlYAngle updates. It
    automatically appends a YYYYMMDD date stamp to both the file name and the
    internal version metadata.

    Parameters
    ----------
    calibrated_siaf_params : dict
        Dictionary mapping SCA names to their updated geometric parameters.
    output_prefix : str, optional
        Prefix for the output filename. Default is "roman_wfi_updates".

    Returns
    -------
    output_filename : str
        The full name of the generated file (e.g., 'roman_wfi_updates_20260526.yml').
    """
    current_date = datetime.now().strftime("%Y%m%d")
    output_filename = f"{output_prefix}_{current_date}.yml"
    yaml_lines = [f"version: '{current_date}'"]

    # Sort the keys to ensure WFI01 through WFI18 are printed in order
    for sca_name in sorted(calibrated_siaf_params.keys()):
        formatted_name = sca_name if "_FULL" in sca_name else f"{sca_name}_FULL"
        yaml_lines.append(f"{formatted_name}:")

        params = calibrated_siaf_params[sca_name]
        yaml_lines.append(f"  V2Ref: {params['V2Ref']:.3f}")
        yaml_lines.append(f"  V3Ref: {params['V3Ref']:.3f}")
        yaml_lines.append(f"  V3IdlYAngle: {params['V3IdlYAngle']:.5f}")

        # --- Export Linear Distortion terms (Forward and Inverse) ---
        if "Sci2IdlX10" in params:
            yaml_lines.append(f"  Sci2IdlX10: {params['Sci2IdlX10']:.8e}")
            yaml_lines.append(f"  Sci2IdlX01: {params['Sci2IdlX01']:.8e}")
            yaml_lines.append(f"  Sci2IdlY10: {params['Sci2IdlY10']:.8e}")
            yaml_lines.append(f"  Sci2IdlY01: {params['Sci2IdlY01']:.8e}")

            yaml_lines.append(f"  Idl2SciX10: {params['Idl2SciX10']:.8e}")
            yaml_lines.append(f"  Idl2SciX01: {params['Idl2SciX01']:.8e}")
            yaml_lines.append(f"  Idl2SciY10: {params['Idl2SciY10']:.8e}")
            yaml_lines.append(f"  Idl2SciY01: {params['Idl2SciY01']:.8e}")

    with open(output_filename, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(
        f"[{os.path.basename(output_filename)}] Successfully exported {len(calibrated_siaf_params)} SCA updates."
    )
    return output_filename
