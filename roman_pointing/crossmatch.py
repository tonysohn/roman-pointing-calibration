import numpy as np
import pysiaf
from scipy.spatial import cKDTree


def generate_cross_matches(
    phot_catalogs, ref_catalog, pointing_info, roman_siaf=None, match_radius_arcsec=1.5
):
    if roman_siaf is None:
        roman_siaf = pysiaf.Siaf("Roman")
    # Remove the old roman_siaf = pysiaf.Siaf("Roman") line below this!    matched_data = {}

    # --- NEW: Dynamically determine the correct RA/Dec column names ---
    ra_col = "ra_epoch" if "ra_epoch" in ref_catalog.colnames else "ra"
    dec_col = "dec_epoch" if "dec_epoch" in ref_catalog.colnames else "dec"

    # 1. Establish the locked base attitude matrix from telemetry
    base_att_matrix = pysiaf.utils.rotations.attitude(
        0, 0, pointing_info["RA_V1"], pointing_info["DEC_V1"], pointing_info["PA_V3"]
    )

    # 2. Spherical Pre-Shift: Project the entire Gaia catalog into ideal V2/V3 space
    v2_gaia, v3_gaia = pysiaf.utils.rotations.getv2v3(
        base_att_matrix,
        np.asarray(ref_catalog[ra_col]),
        np.asarray(ref_catalog[dec_col]),
    )
    gaia_tree = cKDTree(np.column_stack([v2_gaia, v3_gaia]))
    mag_gaia = (
        np.asarray(ref_catalog["phot_g_mean_mag"])
        if "phot_g_mean_mag" in ref_catalog.colnames
        else np.full(len(ref_catalog), 99.0)
    )

    # 3. Cross-match each active SCA
    for sca_key, catalog in phot_catalogs.items():
        aper = roman_siaf[sca_key]
        if not aper:
            continue

        # Extract 1-based coordinates
        x_obs_1based = catalog["x"] + 1
        y_obs_1based = catalog["y"] + 1

        # Map observed pixels to V2/V3
        v2_obs, v3_obs = aper.sci_to_tel(x_obs_1based, y_obs_1based)

        # Nearest-neighbor query
        dists, idxs = gaia_tree.query(
            np.column_stack([v2_obs, v3_obs]), distance_upper_bound=match_radius_arcsec
        )
        valid = dists < match_radius_arcsec

        if np.sum(valid) < 5:
            continue

        valid_obs_indices = np.where(valid)[0]
        valid_cat_indices = idxs[valid]

        # 4. Pack perfectly matched pairs into the standalone format
        matched_data[sca_key] = {
            "x_obs": x_obs_1based[valid_obs_indices],
            "y_obs": y_obs_1based[valid_obs_indices],
            "ra_cat": np.asarray(ref_catalog[ra_col])[valid_cat_indices],
            "dec_cat": np.asarray(ref_catalog[dec_col])[valid_cat_indices],
            "mag_cat": mag_gaia[valid_cat_indices],
        }

    print(
        f"In-memory cross-match complete: successfully paired stars across {len(matched_data)} SCAs."
    )
    return matched_data
