import os
import warnings
from dataclasses import dataclass

import asdf
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RobustMatchConfig:
    """Configuration parameters for robust image-to-catalog matching."""

    search_radius_pix: float = 1.5
    max_translation_pix: float = 2500.0
    primary_bin_pix: float = 4.0
    fallback_bin_pix: float = 20.0
    window_padding_pix: float = 4000.0
    top_catalog_sources: int = 8000
    top_image_sources: int = 8000
    max_candidate_peaks: int = 30
    min_inliers_required: int = 15
    min_total_matches: int = 50
    robust_loss_scale: float = 0.3
    normalization_scale: float = 2000.0
    max_matrix_deviation: float = 0.02


def _compute_mutual_neighbors(predicted_coords, observed_coords, radius_limit):
    """Isolates mutual nearest-neighbor pairs within a specified pixel threshold."""
    tree_obs = cKDTree(observed_coords)
    tree_pred = cKDTree(predicted_coords)

    dists, idxs = tree_obs.query(predicted_coords)
    rev_idxs = tree_pred.query(observed_coords)[1]

    valid_mask = (dists < radius_limit) & (
        rev_idxs[idxs] == np.arange(len(predicted_coords))
    )
    pred_indices = np.where(valid_mask)[0]
    obs_indices = idxs[valid_mask]

    return pred_indices, obs_indices, dists[valid_mask]


def optimize_affine_mapping(
    pred_pts, obs_pts, center_xy=(2044.5, 2044.5), config=RobustMatchConfig()
):
    """Solves for a robust affine transformation matrix using soft-L1 loss optimization."""
    norm = config.normalization_scale
    design_matrix = np.column_stack(
        [(pred_pts - center_xy) / norm, np.ones(len(pred_pts))]
    )
    initial_guess = np.linalg.lstsq(design_matrix, obs_pts - pred_pts, rcond=None)[0]

    optimization_result = least_squares(
        lambda params: (
            design_matrix @ params.reshape(3, 2) - (obs_pts - pred_pts)
        ).ravel(),
        initial_guess.ravel(),
        loss="soft_l1",
        f_scale=config.robust_loss_scale,
    )
    return optimization_result.x.reshape(3, 2)


def apply_affine_transformation(
    xy_points, affine_coeffs, center_xy=(2044.5, 2044.5), config=RobustMatchConfig()
):
    """Applies solved affine parameters to convert nominal coordinates to observed space."""
    norm = config.normalization_scale
    transform_design = np.column_stack(
        [(xy_points - center_xy) / norm, np.ones(len(xy_points))]
    )
    return xy_points + transform_design @ affine_coeffs


def find_detector_consensus_matches(
    pred_coords,
    obs_coords,
    bin_width,
    seed=None,
    shape=(4088, 4088),
    config=RobustMatchConfig(),
):
    """Performs 2D histogram voting (with optional seed offset) followed by iterative affine refinement."""
    detector_center = np.array([(shape[1] + 1) / 2, (shape[0] + 1) / 2])
    size = np.array(shape[::-1])

    if seed is None:
        voted_cat = pred_coords[: config.top_catalog_sources]
        voted_obs = obs_coords[: config.top_image_sources]
        edgesx = edgesy = np.arange(
            -config.max_translation_pix,
            config.max_translation_pix + bin_width,
            bin_width,
        )
    else:
        expected = pred_coords + seed
        voted_cat = pred_coords[
            np.all(
                (expected > -config.window_padding_pix)
                & (expected < size + config.window_padding_pix),
                axis=1,
            )
        ][: config.top_catalog_sources]
        voted_obs = obs_coords[: config.top_image_sources]
        edgesx = np.arange(
            seed[0] - config.window_padding_pix,
            seed[0] + config.window_padding_pix + bin_width,
            bin_width,
        )
        edgesy = np.arange(
            seed[1] - config.window_padding_pix,
            seed[1] + config.window_padding_pix + bin_width,
            bin_width,
        )

    offsets = (voted_obs[:, None, :] - voted_cat[None, :, :]).reshape(-1, 2)
    histogram, _, _ = np.histogram2d(
        offsets[:, 0], offsets[:, 1], bins=(edgesx, edgesy)
    )
    del offsets

    background = gaussian_filter(histogram, 100 / bin_width)
    filtered_hist = (gaussian_filter(histogram, 0.8) - background) / np.sqrt(
        background + 0.2
    )

    peak_coords = np.argwhere(filtered_hist == maximum_filter(filtered_hist, size=9))
    peak_coords = sorted(
        peak_coords, key=lambda pt: filtered_hist[tuple(pt)], reverse=True
    )[: config.max_candidate_peaks]

    valid_solutions = []
    for ix, iy in peak_coords:
        co = np.zeros((3, 2))
        co[2] = [edgesx[ix] + bin_width / 2, edgesy[iy] + bin_width / 2]

        radius_schedule = [80, 40, 20, 10, 5, config.search_radius_pix]

        for lim in radius_schedule:
            p_idx, o_idx, _ = _compute_mutual_neighbors(
                apply_affine_transformation(pred_coords, co, detector_center, config),
                obs_coords,
                lim,
            )
            if len(p_idx) < config.min_inliers_required:
                break
            co = optimize_affine_mapping(
                pred_coords[p_idx], obs_coords[o_idx], detector_center, config
            )

        p_idx, o_idx, dists = _compute_mutual_neighbors(
            apply_affine_transformation(pred_coords, co, detector_center, config),
            obs_coords,
            config.search_radius_pix,
        )

        if (
            len(p_idx) >= config.min_inliers_required
            and np.max(np.abs(co[:2])) / config.normalization_scale
            < config.max_matrix_deviation
        ):
            valid_solutions.append((len(p_idx), float(np.median(dists)), co))

    valid_solutions.sort(key=lambda item: (-item[0], item[1]))
    if not valid_solutions or valid_solutions[0][0] < config.min_total_matches:
        raise RuntimeError("Histogram voting failed to establish a secure consensus.")

    best_affine = valid_solutions[0][2]
    p_idx, o_idx, _ = _compute_mutual_neighbors(
        apply_affine_transformation(pred_coords, best_affine, detector_center, config),
        obs_coords,
        config.search_radius_pix,
    )

    return p_idx, o_idx, best_affine


def project_catalog(
    asdf_filepath,
    reference_catalog,
    config=RobustMatchConfig(),
    custom_siaf_filepath=None,
    dV2=0.0,
    dV3=0.0,
):
    """Projects Gaia sources entirely using PySIAF, accepting XML or YAML overrides, falling back to PRD."""
    import os
    import warnings

    import asdf
    import astropy.units as u
    import numpy as np
    import pysiaf
    import yaml
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with asdf.open(asdf_filepath, lazy_load=True) as f:
            exposure_start = f["roman"]["meta"]["exposure"]["start_time"].utc.isot
            shape = f["roman"]["data"].shape
            ra_v1 = f["roman"]["meta"]["pointing"]["ra_v1"]
            dec_v1 = f["roman"]["meta"]["pointing"]["dec_v1"]
            pa_v3 = f["roman"]["meta"]["pointing"]["pa_v3"]

    basename = os.path.basename(asdf_filepath)
    det = next(
        (p for p in basename.upper().split("_") if p.startswith("WFI") and len(p) == 5),
        None,
    )
    sca_key = f"{det}_FULL"

    # Start with the baseline PRD SIAF
    rsiaf = pysiaf.Siaf("Roman")
    aper = rsiaf[sca_key]

    if custom_siaf_filepath and os.path.exists(custom_siaf_filepath):
        if custom_siaf_filepath.lower().endswith(".xml"):
            base_dir = os.path.dirname(os.path.abspath(custom_siaf_filepath))
            file_name = os.path.basename(custom_siaf_filepath)
            custom_siaf = pysiaf.Siaf("Roman", basepath=base_dir, filename=file_name)
            if sca_key in custom_siaf.apertures:
                aper = custom_siaf[sca_key]

        elif custom_siaf_filepath.lower().endswith((".yml", ".yaml")):
            with open(custom_siaf_filepath, "r") as yf:
                cal_data = yaml.safe_load(yf)

            if sca_key in cal_data:
                params = cal_data[sca_key]
                aper.V2Ref = params["V2Ref"]
                aper.V3Ref = params["V3Ref"]
                aper.V3IdlYAngle = params["V3IdlYAngle"]

                poly_coeffs = aper.get_polynomial_coefficients()
                mapping = {}
                idx = 0
                for d in range(6):
                    for y_deg in range(d + 1):
                        mapping[idx] = f"{d - y_deg}{y_deg}"
                        idx += 1

                for i in range(len(poly_coeffs["Sci2IdlX"])):
                    suffix = mapping[i]
                    if f"Sci2IdlX{suffix}" in params:
                        poly_coeffs["Sci2IdlX"][i] = params[f"Sci2IdlX{suffix}"]
                        poly_coeffs["Sci2IdlY"][i] = params[f"Sci2IdlY{suffix}"]
                        poly_coeffs["Idl2SciX"][i] = params[f"Idl2SciX{suffix}"]
                        poly_coeffs["Idl2SciY"][i] = params[f"Idl2SciY{suffix}"]

                lower_poly_coeffs = {k.lower(): v for k, v in poly_coeffs.items()}
                aper.set_polynomial_coefficients(**lower_poly_coeffs)
        else:
            print(
                f"\n  [WARNING] Unsupported SIAF format: {custom_siaf_filepath}. Using default PRD SIAF."
            )
    else:
        if custom_siaf_filepath:
            print(
                f"\n  [WARNING] Provided SIAF file not found: {custom_siaf_filepath}. Using default PRD."
            )
        else:
            print(
                f"\n  [WARNING] No calibrated SIAF provided for {det}. Using default PRD SIAF."
            )
            print(
                f"            If the cross-match fails due to large offsets, consider reverting to gWCS."
            )

    ra_col = "ra_epoch" if "ra_epoch" in reference_catalog.colnames else "ra"
    dec_col = "dec_epoch" if "dec_epoch" in reference_catalog.colnames else "dec"

    sky_coords = SkyCoord(
        ra=np.asarray(reference_catalog[ra_col]) * u.deg,
        dec=np.asarray(reference_catalog[dec_col]) * u.deg,
        pm_ra_cosdec=np.asarray(reference_catalog["pmra"]) * u.mas / u.yr,
        pm_dec=np.asarray(reference_catalog["pmdec"]) * u.mas / u.yr,
        obstime=Time(np.asarray(reference_catalog["ref_epoch"]), format="jyear"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        propagated_sky = sky_coords.apply_space_motion(new_obstime=Time(exposure_start))

    prop_ra = propagated_sky.ra.deg
    prop_dec = propagated_sky.dec.deg

    att = pysiaf.utils.rotations.attitude(0, 0, ra_v1, dec_v1, pa_v3)
    v2, v3 = pysiaf.utils.rotations.getv2v3(att, prop_ra, prop_dec)

    v2 += dV2
    v3 += dV3

    pixel_x, pixel_y = aper.tel_to_sci(v2, v3)

    valid_pixels = np.isfinite(pixel_x) & np.isfinite(pixel_y)
    pixel_x = pixel_x[valid_pixels]
    pixel_y = pixel_y[valid_pixels]
    filtered_catalog = reference_catalog[valid_pixels]
    prop_ra = prop_ra[valid_pixels]
    prop_dec = prop_dec[valid_pixels]

    ny, nx = shape
    pad = config.window_padding_pix + 15000
    in_bounds = (
        (pixel_x >= -pad)
        & (pixel_x <= nx + pad)
        & (pixel_y >= -pad)
        & (pixel_y <= ny + pad)
    )

    final_catalog = filtered_catalog[in_bounds]

    # PySIAF tel_to_sci outputs 1-based pixels natively. Do not add + 1.
    predicted_pixels = np.column_stack([pixel_x[in_bounds], pixel_y[in_bounds]])

    final_catalog["ra_epoch"] = prop_ra[in_bounds]
    final_catalog["dec_epoch"] = prop_dec[in_bounds]

    brightness_rank = np.argsort(np.asarray(final_catalog["phot_g_mean_mag"]))
    return final_catalog[brightness_rank], predicted_pixels[brightness_rank]
