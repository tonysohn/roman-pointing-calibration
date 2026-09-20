import glob
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysiaf
from astropy.table import Table
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.transform import SimilarityTransform
from tqdm import tqdm

from .diagnostics import generate_alignment_diagnostics


class TerminalLogger(object):
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        # Generates a stamp like: alignment_20260917_211516.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filepath = os.path.join(log_dir, f"alignment_{timestamp}.log")

        self.terminal = sys.stdout
        self.log_file = open(log_filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Forces write to disk immediately so crashes don't lose data

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# Override standard output globally
sys.stdout = TerminalLogger()


def get_3d_vector(v2_arcsec, v3_arcsec):
    """Converts V2/V3 arcsec to a 3D Cartesian unit vector."""
    v2_rad = np.deg2rad(v2_arcsec / 3600.0)
    v3_rad = np.deg2rad(v3_arcsec / 3600.0)
    v1 = np.cos(v3_rad) * np.cos(v2_rad)
    v2 = np.cos(v3_rad) * np.sin(v2_rad)
    v3 = np.sin(v3_rad)
    return np.array([v1, v2, v3])


def get_v2v3_from_3d(vec):
    """Converts a 3D Cartesian unit vector back to V2/V3 arcsec."""
    v2_rad = np.arctan2(vec[1], vec[0])
    v3_rad = np.arcsin(vec[2])
    return np.rad2deg(v2_rad) * 3600.0, np.rad2deg(v3_rad) * 3600.0


def fit_full_distortion(x_pix, y_pix, x_idl, y_idl, x_sci_ref, y_sci_ref, degree=5):
    """Fits an N-th order 2D polynomial from Science Pixels to Ideal Arcsec with numerical scaling."""
    norm = 2048.0
    dx = (x_pix - x_sci_ref) / norm
    dy = (y_pix - y_sci_ref) / norm

    terms = []
    for d in range(degree + 1):
        for y_deg in range(d + 1):
            x_deg = d - y_deg
            terms.append((dx**x_deg) * (dy**y_deg))

    design_matrix = np.column_stack(terms)

    # Iterative Sigma Clipping
    valid = np.ones(len(x_pix), dtype=bool)
    for _ in range(5):
        X_valid = design_matrix[valid]
        cx, _, _, _ = np.linalg.lstsq(X_valid, x_idl[valid], rcond=None)
        cy, _, _, _ = np.linalg.lstsq(X_valid, y_idl[valid], rcond=None)

        calc_x = design_matrix @ cx
        calc_y = design_matrix @ cy
        dist = np.hypot(x_idl - calc_x, y_idl - calc_y)

        med_dist = np.median(dist[valid])
        mad_dist = np.median(np.abs(dist[valid] - med_dist))
        keep = dist < (med_dist + 4.0 * max(mad_dist, 1e-4))

        if np.all(keep[valid]) or np.sum(keep) < len(cx):
            break
        valid = keep

    cx, _, _, _ = np.linalg.lstsq(design_matrix[valid], x_idl[valid], rcond=None)
    cy, _, _, _ = np.linalg.lstsq(design_matrix[valid], y_idl[valid], rcond=None)

    # Rescale coefficients back to physical pixel units
    idx = 0
    for d in range(degree + 1):
        for y_deg in range(d + 1):
            x_deg = d - y_deg
            scale_factor = (norm**x_deg) * (norm**y_deg)
            cx[idx] /= scale_factor
            cy[idx] /= scale_factor
            idx += 1

    return cx, cy, valid


def fit_inverse_distortion(
    x_idl, y_idl, x_pix, y_pix, x_sci_ref, y_sci_ref, valid_mask, degree=5
):
    """Fits the inverse mapping from Ideal Arcsec back to Science Pixels with numerical scaling."""
    dx = x_pix - x_sci_ref
    dy = y_pix - y_sci_ref

    norm = 200.0
    x_n = x_idl / norm
    y_n = y_idl / norm

    terms = []
    for d in range(degree + 1):
        for y_deg in range(d + 1):
            x_deg = d - y_deg
            terms.append((x_n**x_deg) * (y_n**y_deg))

    design_matrix = np.column_stack(terms)
    X_valid = design_matrix[valid_mask]

    cx_inv, _, _, _ = np.linalg.lstsq(X_valid, dx[valid_mask], rcond=None)
    cy_inv, _, _, _ = np.linalg.lstsq(X_valid, dy[valid_mask], rcond=None)

    idx = 0
    for d in range(degree + 1):
        for y_deg in range(d + 1):
            x_deg = d - y_deg
            scale_factor = (norm**x_deg) * (norm**y_deg)
            cx_inv[idx] /= scale_factor
            cy_inv[idx] /= scale_factor
            idx += 1

    return cx_inv, cy_inv


def load_standalone_matches_as_seed(output_dir="roman_gaia_match"):
    out_path = Path(output_dir)
    summary_files = list(out_path.glob("*_summary.json"))
    if not summary_files:
        return None
    precomputed_seeds = {}
    for sf in summary_files:
        with open(sf, "r") as f:
            data = json.load(f)
            det = data.get("detector")
            sca_key = f"{det}_FULL" if not det.endswith("_FULL") else det
            precomputed_seeds[sca_key] = {
                "dx": data.get("dx_center", 0.0),
                "dy": data.get("dy_center", 0.0),
                "affine_coeffs": data.get("affine_offset_coefficients"),
            }
    return precomputed_seeds


def load_standalone_matches(output_dir="roman_gaia_match"):
    matched_data = {}
    ecsv_files = glob.glob(f"{output_dir}/*_matches.ecsv")
    if not ecsv_files:
        return None

    for f in ecsv_files:
        det = os.path.basename(f).split("_")[0]
        sca_key = f"{det}_FULL" if not det.endswith("_FULL") else det
        t = Table.read(f, format="ascii.ecsv")
        matched_data[sca_key] = {
            "x_obs": np.array(t["x"]),
            "y_obs": np.array(t["y"]),
            "ra_cat": np.array(t["ra_epoch"]),
            "dec_cat": np.array(t["dec_epoch"]),
            "mag_cat": np.array(t["phot_g_mean_mag"])
            if "phot_g_mean_mag" in t.colnames
            else np.full(len(t), 99.0),
        }
    return matched_data


def _robust_cross_match(
    v2_obs,
    v3_obs,
    v2_cat,
    v3_cat,
    flux_obs=None,
    mag_cat=None,
    broad_tol=10.0,
    strict_tol=1.0,
    is_wfi01=False,
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
    v2_c_filt, v3_c_filt = v2_cat[in_bounds], v3_cat[in_bounds]
    orig_indices = np.where(in_bounds)[0]

    if len(v2_c_filt) == 0:
        return (
            np.zeros_like(v2_obs, dtype=bool),
            np.zeros_like(v2_obs, dtype=int),
            0.0,
            0.0,
        )

    v2_obs_arr, v3_obs_arr = np.asarray(v2_obs), np.asarray(v3_obs)
    mag_c_filt = (
        np.asarray(mag_cat)[in_bounds]
        if mag_cat is not None
        else np.ones(len(v2_c_filt)) * 99.0
    )
    mag_c_filt = np.where(np.isnan(mag_c_filt) | (mag_c_filt == 0.0), 99.0, mag_c_filt)

    top_n_obs, top_n_cat = min(30, len(v2_obs)), min(150, len(v2_c_filt))

    if flux_obs is not None:
        obs_sort = np.argsort(np.asarray(flux_obs))[::-1]
        v2_o_sub, v3_o_sub = (
            v2_obs_arr[obs_sort][:top_n_obs],
            v3_obs_arr[obs_sort][:top_n_obs],
        )
    else:
        v2_o_sub, v3_o_sub = v2_obs_arr[:top_n_obs], v3_obs_arr[:top_n_obs]

    cat_sort = np.argsort(mag_c_filt)
    v2_r_sub, v3_r_sub = (
        v2_c_filt[cat_sort][:top_n_cat],
        v3_c_filt[cat_sort][:top_n_cat],
    )

    effective_sub_tol = max(broad_tol, 15.0) if broad_tol >= 5.0 else broad_tol

    tree_sub = cKDTree(np.column_stack([v2_r_sub, v3_r_sub]))
    sub_dists, sub_idxs = tree_sub.query(
        np.column_stack([v2_o_sub, v3_o_sub]), distance_upper_bound=effective_sub_tol
    )
    sub_valid = sub_dists < effective_sub_tol

    dv2_bulk, dv3_bulk = 0.0, 0.0
    if np.sum(sub_valid) >= 3:
        matched_obs = np.column_stack([v2_o_sub, v3_o_sub])[sub_valid]
        matched_ref = np.column_stack([v2_r_sub, v3_r_sub])[sub_idxs[sub_valid]]
        dv2_bulk = np.median(matched_ref[:, 0] - matched_obs[:, 0])
        dv3_bulk = np.median(matched_ref[:, 1] - matched_obs[:, 1])

    ref_coords = np.column_stack([v2_c_filt, v3_c_filt])
    tree = cKDTree(ref_coords)
    obs_coords_shifted = np.column_stack([v2_obs_arr + dv2_bulk, v3_obs_arr + dv3_bulk])
    dist_strict, idx_strict_filt = tree.query(
        obs_coords_shifted, distance_upper_bound=strict_tol
    )
    valid_strict = dist_strict < strict_tol

    if np.sum(valid_strict) > 5:
        res_mag = np.hypot(
            ref_coords[idx_strict_filt[valid_strict], 0]
            - obs_coords_shifted[valid_strict, 0],
            ref_coords[idx_strict_filt[valid_strict], 1]
            - obs_coords_shifted[valid_strict, 1],
        )
        med_res = np.median(res_mag)
        mad_res = np.median(np.abs(res_mag - med_res))
        clip_threshold = med_res + 5.0 * max(mad_res, 0.05)
        final_mask = res_mag < clip_threshold
        valid_indices = np.where(valid_strict)[0]
        valid_strict[valid_indices[~final_mask]] = False

    if np.sum(valid_strict) >= 10:
        src_pts = ref_coords[idx_strict_filt[valid_strict]]
        dst_pts = np.column_stack([v2_obs_arr[valid_strict], v3_obs_arr[valid_strict]])

        model = SimilarityTransform()
        success = model.estimate(src_pts, dst_pts)

        if success:
            predicted_obs_coords = model(ref_coords)
            deep_tree = cKDTree(predicted_obs_coords)
            obs_all_coords = np.column_stack([v2_obs_arr, v3_obs_arr])
            deep_dists, deep_idxs = deep_tree.query(
                obs_all_coords, distance_upper_bound=strict_tol
            )

            rev_tree = cKDTree(obs_all_coords)
            rev_dists, rev_idxs = rev_tree.query(
                predicted_obs_coords, distance_upper_bound=strict_tol
            )

            valid_deep = np.zeros(len(v2_obs_arr), dtype=bool)
            final_catalog_indices = np.zeros(len(v2_obs_arr), dtype=int)

            for o_idx in range(len(v2_obs_arr)):
                c_idx = deep_idxs[o_idx]
                if deep_dists[o_idx] < strict_tol:
                    if rev_idxs[c_idx] == o_idx:
                        valid_deep[o_idx] = True
                        final_catalog_indices[o_idx] = c_idx

            valid_strict = valid_deep
            idx_strict_filt = final_catalog_indices

    idx_strict_full = np.zeros_like(idx_strict_filt)
    idx_strict_full[valid_strict] = orig_indices[idx_strict_filt[valid_strict]]

    return valid_strict, idx_strict_full, dv2_bulk, dv3_bulk


def _attitude_residuals(
    delta_params, base_ra, base_dec, base_pa, ra_cat, dec_cat, v2_obs, v3_obs
):
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

    v3_rad = np.deg2rad(v3_calc / 3600.0)
    dv2 = (v2_calc - v2_obs) * np.cos(v3_rad)
    dv3 = v3_calc - v3_obs

    return np.concatenate([dv2, dv3])


def _fit_sca_alignment(
    v2_obs, v3_obs, v2_cat, v3_cat, v2_fiducial, v3_fiducial, sigma_clip=3.0, max_iter=5
):
    x_obs = np.asarray(v2_obs) - v2_fiducial
    y_obs = np.asarray(v3_obs) - v3_fiducial
    x_ref = np.asarray(v2_cat) - v2_fiducial
    y_ref = np.asarray(v3_cat) - v3_fiducial

    valid = np.ones(len(x_obs), dtype=bool)

    for _ in range(max_iter):
        design_matrix = np.column_stack(
            (x_obs[valid], y_obs[valid], np.ones(np.sum(valid)))
        )
        coeffs_x, _, _, _ = np.linalg.lstsq(design_matrix, x_ref[valid], rcond=None)
        coeffs_y, _, _, _ = np.linalg.lstsq(design_matrix, y_ref[valid], rcond=None)

        x_calc = design_matrix @ coeffs_x
        y_calc = design_matrix @ coeffs_y
        dist = np.hypot(x_ref[valid] - x_calc, y_ref[valid] - y_calc)

        med_dist = np.median(dist)
        mad_dist = np.median(np.abs(dist - med_dist))

        if mad_dist == 0:
            break

        keep = dist < (med_dist + 4.0 * max(mad_dist, 1e-4))
        if np.all(keep) or np.sum(keep) < 3:
            break

        valid_indices = np.where(valid)[0]
        valid[valid_indices[~keep]] = False

    design_matrix = np.column_stack(
        (x_obs[valid], y_obs[valid], np.ones(np.sum(valid)))
    )
    coeffs_x, _, _, _ = np.linalg.lstsq(design_matrix, x_ref[valid], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(design_matrix, y_ref[valid], rcond=None)

    x_calc = design_matrix @ coeffs_x
    y_calc = design_matrix @ coeffs_y

    dof = np.sum(valid) - 3
    if dof > 0:
        MSE_x = np.sum((x_ref[valid] - x_calc) ** 2) / dof
        MSE_y = np.sum((y_ref[valid] - y_calc) ** 2) / dof
    else:
        MSE_x, MSE_y = 0.0, 0.0

    try:
        cov_matrix = np.linalg.inv(design_matrix.T @ design_matrix)
        err_x = np.sqrt(np.diag(cov_matrix) * MSE_x)
        err_y = np.sqrt(np.diag(cov_matrix) * MSE_y)
    except np.linalg.LinAlgError:
        err_x = np.array([np.nan, np.nan, np.nan])
        err_y = np.array([np.nan, np.nan, np.nan])

    dv2, dv2_err = coeffs_x[2], err_x[2]
    dv3, dv3_err = coeffs_y[2], err_y[2]

    sin_theta = (coeffs_y[0] - coeffs_x[1]) / 2.0
    d_theta_deg = np.degrees(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))

    scale_x = coeffs_x[0]
    scale_y = coeffs_y[1]
    skew = (coeffs_x[1] + coeffs_y[0]) / 2.0

    dx = x_ref[valid] - x_calc
    dy = y_ref[valid] - y_calc
    final_rms_mas = np.sqrt(np.mean(dx**2 + dy**2)) * 1000.0
    return dv2, dv3, d_theta_deg, scale_x, scale_y, skew, final_rms_mas


def align_wfi(
    phot_catalogs,
    ref_catalog,
    pointing_info,
    max_iterations=5,
    debug=False,
    precomputed_matches_dir="roman_gaia_match",
    target_wfi_cen=None,
    roman_siaf=None,
    fit_degree=4,  # Default to Full Polynomial
):
    warnings.filterwarnings("ignore", message=".*Gaia archive is in evolution.*")

    if roman_siaf is None:
        roman_siaf = pysiaf.Siaf("Roman")

    cos_dec = np.cos(np.deg2rad(pointing_info["DEC_V1"]))

    # Directly initialize from telemetry without manual offsets
    current_ra = pointing_info["RA_V1"]
    current_dec = pointing_info["DEC_V1"]
    current_pa = pointing_info["PA_V3"]

    precomputed_matches = load_standalone_matches(precomputed_matches_dir)
    iteration_history = []

    print("\nSolving Global Attitude...")
    # -------------------------------------------------------------------------
    # ITERATIVE GLOBAL ATTITUDE SOLVER
    # -------------------------------------------------------------------------
    for i in range(max_iterations):
        att_matrix = pysiaf.utils.rotations.attitude(
            0, 0, current_ra, current_dec, current_pa
        )
        global_v2_obs, global_v3_obs, global_ra_ref, global_dec_ref = [], [], [], []
        valid_scas_found = 0

        for aper_name, catalog in phot_catalogs.items():
            aper = roman_siaf[aper_name]
            if not aper:
                continue

            # --- DIRECT INJECTION OF EXACT MATCHES ---
            if precomputed_matches and aper_name in precomputed_matches:
                x_obs_1based = precomputed_matches[aper_name]["x_obs"]
                y_obs_1based = precomputed_matches[aper_name]["y_obs"]

                v2_obs, v3_obs = aper.sci_to_tel(x_obs_1based, y_obs_1based)

                valid_scas_found += 1
                global_v2_obs.extend(v2_obs)
                global_v3_obs.extend(v3_obs)
                global_ra_ref.extend(precomputed_matches[aper_name]["ra_cat"])
                global_dec_ref.extend(precomputed_matches[aper_name]["dec_cat"])

            else:
                # Fallback to KD-tree guessing
                v2_obs, v3_obs = aper.sci_to_tel(catalog["x"] + 1, catalog["y"] + 1)
                flux_obs = catalog["flux"] if "flux" in catalog.colnames else None
                tol = 300.0 if i == 0 else 20.0 if i == 1 else 10.0
                strict_tol = 5.0 if i == 0 else 3.0 if i < 3 else 1.0

                valid, idx, _, _ = _robust_cross_match(
                    v2_obs,
                    v3_obs,
                    v2_gaia,
                    v3_gaia,
                    flux_obs=flux_obs,
                    mag_cat=mag_gaia,
                    broad_tol=tol,
                    strict_tol=strict_tol,
                    is_wfi01=("WFI01" in aper_name),
                )

                if np.sum(valid) > 5:
                    valid_scas_found += 1
                    valid_indices = np.where(valid)[0]
                    global_v2_obs.extend(v2_obs[valid_indices])
                    global_v3_obs.extend(v3_obs[valid_indices])
                    global_ra_ref.extend(ref_catalog["ra_epoch"][idx[valid_indices]])
                    global_dec_ref.extend(ref_catalog["dec_epoch"][idx[valid_indices]])

        bounds = ([-300.0, -300.0, -1800.0], [300.0, 300.0, 1800.0])
        global_result = least_squares(
            _attitude_residuals,
            [0.0, 0.0, 0.0],
            args=(
                current_ra,
                current_dec,
                current_pa,
                np.asarray(global_ra_ref),
                np.asarray(global_dec_ref),
                np.asarray(global_v2_obs),
                np.asarray(global_v3_obs),
            ),
            method="trf",
            loss="linear" if i == 0 else "soft_l1",
            f_scale=1.0,
            bounds=bounds,
            ftol=1e-5,
            xtol=1e-5,
        )

        d_ra, d_dec, d_pa = global_result.x
        learning_rate = 1.0 if i < 2 else (1.0 / (i + 1))
        current_ra += learning_rate * ((d_ra / 3600.0) / cos_dec)
        current_dec += learning_rate * (d_dec / 3600.0)
        current_pa = (current_pa + learning_rate * (d_pa / 3600.0)) % 360.0

        step_mag_arcsec = np.sqrt(d_ra**2 + d_dec**2)
        iteration_history.append(step_mag_arcsec)

        progress = int((i + 1) / max_iterations * 100)
        bar_str = "█" * int(progress / 2) + " " * (50 - int(progress / 2))
        print(
            f"  {progress:3d}%|{bar_str}| {i + 1}/{max_iterations} [Res_Mag_arcsec={step_mag_arcsec:.3f}]"
        )

        if np.sqrt(d_ra**2 + d_dec**2 + d_pa**2) < 0.001:
            break

    diag_dir = "diagnostics"
    os.makedirs(diag_dir, exist_ok=True)

    J = global_result.jac
    cov_global = np.linalg.inv(J.T @ J)
    MSE_global = (global_result.fun**2).mean()
    att_err_arcsec = np.sqrt(np.diagonal(cov_global) * MSE_global)

    locked_att_matrix = pysiaf.utils.rotations.attitude(
        0, 0, current_ra, current_dec, current_pa
    )

    # -------------------------------------------------------------------------
    # LOCAL SCA FITTING & DIAGNOSTIC LOGGING
    # -------------------------------------------------------------------------
    calibrated_siaf_params = {}
    summary_log_data = []
    matched_pairs_log_data = []
    successfully_fitted_scas = []

    for aper_name, catalog in phot_catalogs.items():
        aper = roman_siaf[aper_name]

        if precomputed_matches and aper_name in precomputed_matches:
            x_obs_1based = precomputed_matches[aper_name]["x_obs"]
            y_obs_1based = precomputed_matches[aper_name]["y_obs"]

            v2_obs, v3_obs = aper.sci_to_tel(x_obs_1based, y_obs_1based)
            v2_ref_fit, v3_ref_fit = pysiaf.utils.rotations.getv2v3(
                locked_att_matrix,
                precomputed_matches[aper_name]["ra_cat"],
                precomputed_matches[aper_name]["dec_cat"],
            )

            log_x_fit, log_y_fit = x_obs_1based - 1, y_obs_1based - 1
            log_ra_fit, log_dec_fit = (
                precomputed_matches[aper_name]["ra_cat"],
                precomputed_matches[aper_name]["dec_cat"],
            )
            log_f_fit, log_m_fit = (
                np.full(len(x_obs_1based), np.nan),
                precomputed_matches[aper_name]["mag_cat"],
            )

            num_matched = len(v2_ref_fit)
            dv2_bulk = np.median(v2_ref_fit - v2_obs) if num_matched > 0 else 0.0
            dv3_bulk = np.median(v3_ref_fit - v3_obs) if num_matched > 0 else 0.0
            summary_log_data.append(
                [
                    aper_name,
                    len(v2_obs),
                    num_matched,
                    round(dv2_bulk, 3),
                    round(dv3_bulk, 3),
                ]
            )

            if num_matched < 15:
                calibrated_siaf_params[aper_name] = {
                    "V2Ref": aper.V2Ref,
                    "V3Ref": aper.V3Ref,
                    "V3IdlYAngle": aper.V3IdlYAngle,
                }
                continue

            # Lock the local reference frame origins using the affine fit
            dv2, dv3, d_theta, scale_x, scale_y, skew, sca_rms = _fit_sca_alignment(
                v2_obs, v3_obs, v2_ref_fit, v3_ref_fit, aper.V2Ref, aper.V3Ref
            )
            aper.V2Ref += dv2
            aper.V3Ref += dv3
            aper.V3IdlYAngle -= d_theta

            calibrated_siaf_params[aper_name] = {
                "V2Ref": aper.V2Ref,
                "V3Ref": aper.V3Ref,
                "V3IdlYAngle": aper.V3IdlYAngle,
            }
            poly_coeffs = aper.get_polynomial_coefficients()

            if fit_degree == 5:
                # Extract target Ideal coordinates using the newly locked frame
                x_idl_ref, y_idl_ref = aper.tel_to_idl(v2_ref_fit, v3_ref_fit)

                # Fit 5th order polynomials
                cx, cy, valid_mask = fit_full_distortion(
                    x_obs_1based,
                    y_obs_1based,
                    x_idl_ref,
                    y_idl_ref,
                    aper.XSciRef,
                    aper.YSciRef,
                    degree=5,
                )
                cx_inv, cy_inv = fit_inverse_distortion(
                    x_idl_ref,
                    y_idl_ref,
                    x_obs_1based,
                    y_obs_1based,
                    aper.XSciRef,
                    aper.YSciRef,
                    valid_mask,
                    degree=5,
                )

                num_fitted_terms = len(cx)  # Will be 21 for degree 5
                for i in range(num_fitted_terms):
                    if i == 0:  # Force affine zero-points strictly to 0.0
                        poly_coeffs["Sci2IdlX"][i] = 0.0
                        poly_coeffs["Sci2IdlY"][i] = 0.0
                        poly_coeffs["Idl2SciX"][i] = 0.0
                        poly_coeffs["Idl2SciY"][i] = 0.0
                    else:
                        poly_coeffs["Sci2IdlX"][i] = cx[i]
                        poly_coeffs["Sci2IdlY"][i] = cy[i]
                        poly_coeffs["Idl2SciX"][i] = cx_inv[i]
                        poly_coeffs["Idl2SciY"][i] = cy_inv[i]

                for i in range(num_fitted_terms, len(poly_coeffs["Sci2IdlX"])):
                    poly_coeffs["Sci2IdlX"][i] = 0.0
                    poly_coeffs["Sci2IdlY"][i] = 0.0
                    poly_coeffs["Idl2SciX"][i] = 0.0
                    poly_coeffs["Idl2SciY"][i] = 0.0

            elif fit_degree == 1:
                # Standard Affine Mode (Scale & Skew updates)
                scale_error = max(abs(scale_x - 1.0), abs(scale_y - 1.0))
                skew_error = abs(skew)

                if scale_error > 5e-5 or skew_error > 5e-5:
                    poly_coeffs["Sci2IdlX"][1] *= scale_x
                    poly_coeffs["Sci2IdlX"][2] += skew
                    poly_coeffs["Sci2IdlY"][1] += skew
                    poly_coeffs["Sci2IdlY"][2] *= scale_y

                    M_forward = np.array(
                        [
                            [poly_coeffs["Sci2IdlX"][1], poly_coeffs["Sci2IdlX"][2]],
                            [poly_coeffs["Sci2IdlY"][1], poly_coeffs["Sci2IdlY"][2]],
                        ]
                    )
                    M_inverse = np.linalg.inv(M_forward)
                    poly_coeffs["Idl2SciX"][1] = M_inverse[0, 0]
                    poly_coeffs["Idl2SciX"][2] = M_inverse[0, 1]
                    poly_coeffs["Idl2SciY"][1] = M_inverse[1, 0]
                    poly_coeffs["Idl2SciY"][2] = M_inverse[1, 1]

            # Update Aperture and store populated terms for YAML Export
            lower_poly_coeffs = {k.lower(): v for k, v in poly_coeffs.items()}
            aper.set_polynomial_coefficients(**lower_poly_coeffs)

            mapping = {}
            idx = 0

            # Dynamically bound the loop instead of hardcoding 6
            for d in range(fit_degree + 1):
                for y_deg in range(d + 1):
                    mapping[idx] = f"{d}{y_deg}"
                    idx += 1

            for i in range(len(poly_coeffs["Sci2IdlX"])):
                suffix = mapping[i]
                calibrated_siaf_params[aper_name][f"Sci2IdlX{suffix}"] = poly_coeffs[
                    "Sci2IdlX"
                ][i]
                calibrated_siaf_params[aper_name][f"Sci2IdlY{suffix}"] = poly_coeffs[
                    "Sci2IdlY"
                ][i]
                calibrated_siaf_params[aper_name][f"Idl2SciX{suffix}"] = poly_coeffs[
                    "Idl2SciX"
                ][i]
                calibrated_siaf_params[aper_name][f"Idl2SciY{suffix}"] = poly_coeffs[
                    "Idl2SciY"
                ][i]

            # --- CALCULATE TRUE RESIDUALS IMMEDIATELY (VECTORIZED) ---
            # Evaluate final pre-shift coordinates for accurate physical residual math
            v2_cal_final, v3_cal_final = aper.sci_to_tel(
                log_x_fit + 1.0, log_y_fit + 1.0
            )
            res_v2_mas = (v2_cal_final - v2_ref_fit) * 1000.0
            res_v3_mas = (v3_cal_final - v3_ref_fit) * 1000.0

            for j in range(num_matched):
                matched_pairs_log_data.append(
                    [
                        aper_name,
                        round(log_x_fit[j], 2),
                        round(log_y_fit[j], 2),
                        round(log_ra_fit[j], 6),
                        round(log_dec_fit[j], 6),
                        log_f_fit[j],
                        log_m_fit[j],
                        round(float(v2_cal_final[j]), 4),
                        round(float(v3_cal_final[j]), 4),
                        round(float(res_v2_mas[j]), 2),
                        round(float(res_v3_mas[j]), 2),
                    ]
                )

            successfully_fitted_scas.append(aper_name)

    attitude_results = {
        "RA_V1": current_ra,
        "DEC_V1": current_dec,
        "PA_V3": current_pa,
        "RA_V1_err_arcsec": att_err_arcsec[0],
        "DEC_V1_err_arcsec": att_err_arcsec[1],
        "PA_V3_err_arcsec": att_err_arcsec[2],
    }

    if len(successfully_fitted_scas) > 0:
        mean_dv2 = np.mean(
            [
                calibrated_siaf_params[s]["V2Ref"] - roman_siaf[s].V2Ref
                for s in successfully_fitted_scas
            ]
        )
        mean_dv3 = np.mean(
            [
                calibrated_siaf_params[s]["V3Ref"] - roman_siaf[s].V3Ref
                for s in successfully_fitted_scas
            ]
        )
        mean_dtheta = np.mean(
            [
                calibrated_siaf_params[s]["V3IdlYAngle"] - roman_siaf[s].V3IdlYAngle
                for s in successfully_fitted_scas
            ]
        )

        # --- RIGID BODY ENSEMBLE UPDATE FOR WFI_CEN ---
        # Update WFI_CEN as the master reference aperture tracking the bulk movement of the 18 SCAs
        cen = roman_siaf["WFI_CEN"]
        calibrated_siaf_params["WFI_CEN"] = {
            "V2Ref": cen.V2Ref + mean_dv2,
            "V3Ref": cen.V3Ref + mean_dv3,
            "V3IdlYAngle": cen.V3IdlYAngle + mean_dtheta,
        }
        # ----------------------------------------------

        # NOTE: We intentionally do NOT subtract the mean from the individual SCAs here.
        # Each SCA retains its true absolute solved position so the exported YAML
        # accurately captures the floating mosaic geometry.

        attitude_results["Residual_Mean_V2_mas"] = mean_dv2 * 1000.0
        attitude_results["Residual_Mean_V3_mas"] = mean_dv3 * 1000.0
        attitude_results["RA_V1"] += (mean_dv2 / 3600.0) / cos_dec
        attitude_results["DEC_V1"] += mean_dv3 / 3600.0
        attitude_results["PA_V3"] += mean_dtheta / 3600.0

    if target_wfi_cen is not None:
        bam_v2_cen = target_wfi_cen["V2"]
        bam_v3_cen = target_wfi_cen["V3"]
        bam_angle_cen = target_wfi_cen["Angle"]

        nom_v2_cen, nom_v3_cen, nom_angle_cen = (
            roman_siaf["WFI_CEN"].V2Ref,
            roman_siaf["WFI_CEN"].V3Ref,
            roman_siaf["WFI_CEN"].V3IdlYAngle,
        )

        dtheta_rad = np.deg2rad(bam_angle_cen - nom_angle_cen)
        cos_t, sin_t = np.cos(dtheta_rad), np.sin(dtheta_rad)

        for sca in successfully_fitted_scas:
            dx, dy = (
                calibrated_siaf_params[sca]["V2Ref"] - nom_v2_cen,
                calibrated_siaf_params[sca]["V3Ref"] - nom_v3_cen,
            )
            # Corrected V2/V3 Counter-Clockwise Rotation Matrix
            calibrated_siaf_params[sca]["V2Ref"] = bam_v2_cen + (
                dx * cos_t + dy * sin_t
            )
            calibrated_siaf_params[sca]["V3Ref"] = bam_v3_cen + (
                -dx * sin_t + dy * cos_t
            )
            calibrated_siaf_params[sca]["V3IdlYAngle"] += bam_angle_cen - nom_angle_cen

        calibrated_siaf_params["WFI_CEN"] = {
            "V2Ref": bam_v2_cen,
            "V3Ref": bam_v3_cen,
            "V3IdlYAngle": bam_angle_cen,
        }

    # Vectorized post-calibration logging & diagnostic evaluation
    diagnostic_log = []
    rows_by_sca = {}
    for row in matched_pairs_log_data:
        sca_name = row[0]
        if sca_name not in rows_by_sca:
            rows_by_sca[sca_name] = []
        rows_by_sca[sca_name].append(row)

    for sca_name, rows in rows_by_sca.items():
        aper = roman_siaf[sca_name]
        aper.V2Ref = calibrated_siaf_params[sca_name]["V2Ref"]
        aper.V3Ref = calibrated_siaf_params[sca_name]["V3Ref"]
        aper.V3IdlYAngle = calibrated_siaf_params[sca_name]["V3IdlYAngle"]

        x_pix = np.array([r[1] for r in rows]) + 1.0
        y_pix = np.array([r[2] for r in rows]) + 1.0

        final_v2, final_v3 = aper.sci_to_tel(x_pix, y_pix)

        for i, r in enumerate(rows):
            new_row = list(r)
            # Update ONLY the plotting coordinates (indices 7 and 8) so they track with WFI_CEN
            new_row[7] = round(float(final_v2[i]), 4)
            new_row[8] = round(float(final_v3[i]), 4)
            # Do NOT recalculate residuals here; the true physics were already locked above!
            diagnostic_log.append(new_row)

    generate_alignment_diagnostics(
        matched_pairs_log=diagnostic_log,
        iteration_history=iteration_history,
        output_dir="./diagnostics",
        calibrated_siaf_params=calibrated_siaf_params,
        nominal_siaf=roman_siaf,
    )

    # -------------------------------------------------------------------------
    # SPHERICAL GEOMETRY UPDATE FOR CGI_CEN
    # -------------------------------------------------------------------------
    # CAUTION: We are explicitly loading the PRD preflight SIAF here as the
    # baseline rather than relying on the `roman_siaf` function argument.
    # This ensures CGI_CEN is transformed against the pristine preflight
    # hardware geometry before applying the newly calibrated WFI_CEN shift.

    prd_siaf = pysiaf.Siaf("Roman")

    if "CGI_CEN" in prd_siaf.apertures:
        cgi_old = prd_siaf["CGI_CEN"]
        wfi_old = prd_siaf["WFI_CEN"]

        v2_wfi_old, v3_wfi_old = wfi_old.V2Ref, wfi_old.V3Ref
        v2_wfi_new, v3_wfi_new = (
            calibrated_siaf_params["WFI_CEN"]["V2Ref"],
            calibrated_siaf_params["WFI_CEN"]["V3Ref"],
        )
        dtheta_deg = (
            calibrated_siaf_params["WFI_CEN"]["V3IdlYAngle"] - wfi_old.V3IdlYAngle
        )

        # M_old maps Body to Local Sky. M_new maps Calibrated Body to the same Local Sky.
        M_old = pysiaf.utils.rotations.attitude(v2_wfi_old, v3_wfi_old, 0.0, 0.0, 0.0)
        M_new = pysiaf.utils.rotations.attitude(
            v2_wfi_new, v3_wfi_new, 0.0, 0.0, dtheta_deg
        )

        cgi_vec_old = get_3d_vector(cgi_old.V2Ref, cgi_old.V3Ref)

        # Body-to-Sky transform, then Sky-to-Body transform
        cgi_vec_sky = np.dot(M_old, cgi_vec_old)
        cgi_vec_new = np.dot(M_new.T, cgi_vec_sky)

        v2_cgi_new, v3_cgi_new = get_v2v3_from_3d(cgi_vec_new)
        angle_cgi_new = cgi_old.V3IdlYAngle + dtheta_deg

        calibrated_siaf_params["CGI_CEN"] = {
            "V2Ref": v2_cgi_new,
            "V3Ref": v3_cgi_new,
            "V3IdlYAngle": angle_cgi_new,
        }

    return calibrated_siaf_params, attitude_results, diagnostic_log


def export_alignment_to_yaml(calibrated_siaf_params, output_prefix="roman_wfi_updates"):
    current_date = datetime.now().strftime("%Y%m%d")
    output_filename = f"{output_prefix}_{current_date}.yml"
    yaml_lines = [f"version: '{current_date}'"]

    mapping = {}
    idx = 0

    # Dynamically bound the loop instead of hardcoding 6
    for d in range(fit_degree + 1):
        for y_deg in range(d + 1):
            mapping[idx] = f"{d - y_deg}{y_deg}"
            idx += 1

    for sca_name in sorted(calibrated_siaf_params.keys()):
        formatted_name = sca_name if "_FULL" in sca_name else f"{sca_name}_FULL"
        yaml_lines.append(f"{formatted_name}:")
        params = calibrated_siaf_params[sca_name]
        yaml_lines.append(f"  V2Ref: {params['V2Ref']:.3f}")
        yaml_lines.append(f"  V3Ref: {params['V3Ref']:.3f}")
        yaml_lines.append(f"  V3IdlYAngle: {params['V3IdlYAngle']:.5f}")

        if "Sci2IdlX10" in params:
            for i in range(21):
                term_key = f"Sci2IdlX{mapping[i]}"
                if (
                    term_key in params and params[term_key] != 0.0
                ):  # Export non-zero dynamic terms
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
