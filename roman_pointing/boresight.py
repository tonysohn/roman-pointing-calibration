import numpy as np
from astropy.constants import c
from scipy.spatial.transform import Rotation as R
import pysiaf

def calibrate_roman_fgs_alignment(
    reference_stars_radec,  # (N, 2) array [deg]
    measured_v2_v3,  # (N, 2) array [arcsec] from WFI alignment
    q_eci2b,  # (4,) quaternion [x, y, z, w]
    v_sc_eci_kms,  # (3,) vector [km/s]
    wfi_cen_aper,  # The WFI_CEN PySIAF aperture object
    q_b2fgs_old=np.array(
        [
            -0.1859673417539929,
            +0.6837984564491885,
            -0.1800546332580956,
            +0.6822141509826322,
        ]
    ),  # (4,) quaternion [x, y, z, w]
):
    """
    Estimates an updated Body-to-FGS alignment quaternion for the Roman Space Telescope.

    This function implements a multi-star best-fit delta rotation workflow. It predicts
    the apparent Body-frame directions of astrometric reference stars by applying spacecraft
    velocity aberration to catalog ECI vectors. It then compares these predictions against
    the actual measured Body-frame directions, traced through the calibrated SIAF geometry.
    By solving Wahba's problem across multiple stars, it extracts a 3D correction matrix
    that constrains both translational offsets and rotational position-angle errors.

    Parameters
    ----------
    reference_stars_radec : ndarray
        (N, 2) array of reference star sky coordinates (RA, Dec) in degrees.
    gs_sci_pixels : ndarray
        (N, 2) array of measured star positions as (x_sci, y_sci) in detector pixels.
    q_eci2b : ndarray
        (4,) array representing the known ECI-to-Body telemetry attitude quaternion (scalar-last).
    q_b2fgs_old : ndarray
        (4,) array representing the nominal or previous Body-to-FGS alignment quaternion (scalar-last).
    v_sc_eci_kms : ndarray
        (3,) vector containing the spacecraft ECI velocity [km/s] for aberration correction.
    wfi_cen_aper : pysiaf.Aperture
        The PySIAF aperture object defining the fixed WFI_CEN focal plane origin.

    Returns
    -------
    q_b2fgs_new : ndarray
        (4,) array representing the updated, calibrated Body-to-FGS alignment quaternion (scalar-last).
    """
    reference_stars_radec = np.atleast_2d(reference_stars_radec)
    measured_v2_v3 = np.atleast_2d(measured_v2_v3)

    # --- EXPLICIT MATRICES ---
    # Due to Roman FSW telemetry conventions:
    # q_eci2b (or qbj) is actually Body-to-ECI, so we take the transpose to get ECI-to-Body.
    m_ECI_to_B = R.from_quat(q_eci2b).as_matrix().T

    # q_b2fgs_old (qb) natively generates M_FGStoB in Scipy
    m_FGS_to_B_old = R.from_quat(q_b2fgs_old).as_matrix()

    # --- 1. PREDICTED VECTORS (Reference in Body Frame) ---
    ra_rad = np.deg2rad(reference_stars_radec[:, 0])
    dec_rad = np.deg2rad(reference_stars_radec[:, 1])
    u_eci = np.stack(
        [
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ],
        axis=1,
    )

    beta = (v_sc_eci_kms / c.to("km/s").value).reshape(1, 3)
    u_dot_beta = np.sum(u_eci * beta, axis=1, keepdims=True)
    u_eci_apparent = u_eci + beta - u_dot_beta * u_eci
    u_eci_apparent /= np.linalg.norm(u_eci_apparent, axis=1, keepdims=True)

    # Rotate ECI vectors into the Body Frame
    u_body_ref = (m_ECI_to_B @ u_eci_apparent.T).T

    # --- 2. MEASURED VECTORS (Measured FGS in Body Frame) ---
    x_wc, y_wc = wfi_cen_aper.tel_to_idl(measured_v2_v3[:, 0], measured_v2_v3[:, 1])

    # FGS-specific axis flip from WFI_CEN
    x_fgs_ang = x_wc
    y_fgs_ang = -y_wc

    x_rad = np.deg2rad(x_fgs_ang / 3600.0)
    y_rad = np.deg2rad(y_fgs_ang / 3600.0)

    u_fgs_meas = np.stack([np.tan(x_rad), np.tan(y_rad), np.ones_like(x_rad)], axis=1)
    u_fgs_meas /= np.linalg.norm(u_fgs_meas, axis=1, keepdims=True)

    # Rotate measured FGS vectors into the Body Frame
    u_body_meas = (m_FGS_to_B_old @ u_fgs_meas.T).T

    # --- 3. SOLVE DELTA CORRECTION ---
    # R.align_vectors(A, B) finds rotation R such that R * B = A
    # Here, it finds the Delta that maps measured body vectors to reference body vectors
    wahba_res = R.align_vectors(u_body_ref, u_body_meas)
    delta_R = wahba_res[0].as_matrix()

    print(f"Wahba Solver Internal Delta: {np.rad2deg(wahba_res[1]) * 3600:.4f} arcsec")

    # --- 4. UPDATE ALIGNMENT MATRIX & CONVERT TO QUATERNION ---
    # M_{FGS -> Body_new} = Delta_R * M_{FGS -> Body_old}
    m_FGS_to_B_new = delta_R @ m_FGS_to_B_old
    q_b2fgs_new = R.from_matrix(m_FGS_to_B_new).as_quat()

    return q_b2fgs_new
