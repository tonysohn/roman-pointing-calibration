import warnings

import astropy.units as u
import erfa
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

# Suppress the ERFA distance override warnings caused by missing parallaxes
warnings.filterwarnings("ignore", category=erfa.core.ErfaWarning, append=True)
warnings.filterwarnings("ignore", message=".*Gaia archive is in evolution.*")


def fetch_local_commissioning_gaia(
    obs_date_str, local_csv_path="gaia_dr3_commissioning_field.ecsv", apply_pm=True
):

    # Load the catalog
    ref_catalog = Table.read(local_csv_path, format="ascii.ecsv")

    # Clean out NaNs in proper motion (required for propagation)
    valid_pm = ~np.isnan(ref_catalog["pmra"]) & ~np.isnan(ref_catalog["pmdec"])
    ref_catalog = ref_catalog[valid_pm]

    if apply_pm:
        print(
            f"  -> Propagating Gaia proper motions to {obs_date_str} (Parallax forced to 0.0)"
        )

        # Create SkyCoord object at Gaia DR3 epoch (2016.0)
        # CRITICAL: Force parallax and RV to 0 to avoid LMC distance systematics!
        coords = SkyCoord(
            ra=ref_catalog["ra"] * u.deg,
            dec=ref_catalog["dec"] * u.deg,
            pm_ra_cosdec=ref_catalog["pmra"] * u.mas / u.yr,
            pm_dec=ref_catalog["pmdec"] * u.mas / u.yr,
            frame="icrs",
            obstime=Time(2016.0, format="jyear"),
        )

        # Propagate to the observation epoch
        obs_time = Time(obs_date_str)
        propagated_coords = coords.apply_space_motion(obs_time)

        ref_catalog["ra_epoch"] = propagated_coords.ra.deg
        ref_catalog["dec_epoch"] = propagated_coords.dec.deg
    else:
        print(
            "  -> Proper motion propagation disabled. Using native Gaia DR3 coordinates (epoch J2016.0)."
        )
        ref_catalog["ra_epoch"] = ref_catalog["ra"]
        ref_catalog["dec_epoch"] = ref_catalog["dec"]

    return ref_catalog


def apply_dva_scale_to_catalog(ra_true, dec_true, ra_ref, dec_ref, scale_factor):
    """
    Applies Differential Velocity Aberration (DVA) stretch to a catalog using
    exact spherical trigonometry.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    # 1. Define SkyCoord objects (handles both scalars and arrays natively)
    ref_coord = SkyCoord(ra=ra_ref, dec=dec_ref, unit=u.deg)
    true_coord = SkyCoord(ra=ra_true, dec=dec_true, unit=u.deg)

    # 2. Calculate exact great-circle separation and position angle (bearing)
    sep = ref_coord.separation(true_coord)
    pa = ref_coord.position_angle(true_coord)

    # 3. Apply the DVA stretch radially to the separation arc
    app_sep = sep / scale_factor

    # 4. Project the new coordinates along the exact spherical surface
    app_coord = ref_coord.directional_offset_by(pa, app_sep)

    return app_coord.ra.deg, app_coord.dec.deg
