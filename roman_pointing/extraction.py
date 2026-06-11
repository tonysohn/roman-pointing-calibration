import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import roman_datamodels as rdm
import stpsf
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table
from astropy.visualization import simple_norm
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.background import MADStdBackgroundRMS, MMMBackground
from photutils.centroids import centroid_2dg, centroid_sources
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import GriddedPSFModel, IterativePSFPhotometry


def _extract_with_gaussian(data_es, bkg_val, std_val):
    """
    Helper function to extract sources using IRAFStarFinder and 2D Gaussian centroiding.
    """
    # 1. Initial coarse pass
    bright_stars = IRAFStarFinder(
        threshold=50.0 * std_val + bkg_val,
        fwhm=1.5,
        min_separation=7,
        roundness_range=(0.0, 0.6),
        sharpness_range=(0.6, 1.4),
    )

    sources = bright_stars(data_es)

    if sources is None or len(sources) == 0:
        return Table(names=("x", "y", "flux"), dtype=("f8", "f8", "f8"))

    # 2. Morphological cuts
    mask = (
        (sources["sharpness"] > 0.75)
        & (sources["sharpness"] < 1.2)
        & (sources["flux"] > 50)
    )
    sources_masked = sources[mask]

    xarr_raw = sources_masked["x_centroid"]
    yarr_raw = sources_masked["y_centroid"]
    fluxarr_raw = sources_masked["flux"]

    # 3. Precision centroiding
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The fit may not have converged.*",
            category=UserWarning,
            module="photutils.centroids.gaussian",
        )
        xarr_fit, yarr_fit = centroid_sources(
            data_es, xarr_raw, yarr_raw, box_size=5, centroid_func=centroid_2dg
        )

    # Return exactly matching the pipeline standard
    return Table([xarr_fit, yarr_fit, fluxarr_raw], names=("x", "y", "flux"))


def _extract_with_stpsf(data_es, bkg_val, std_val, sca_name):
    """
    Helper function to extract sources using STPSF and Photutils IterativePSFPhotometry.
    """

    print(f"  -> Generating STPSF optical model for {sca_name}...")

    # 1. Configure the Roman Simulator
    wfi = stpsf.WFI()
    wfi.filter = "F146"
    wfi.detector = sca_name.upper()

    # 2. Generate a 3x3 grid of PSFs to capture optical variations across the chip
    grid = wfi.psf_grid(num_psfs=9, oversample=4)
    psf_model = GriddedPSFModel(grid)

    # 3. Set up the Iterative PSF Fitter
    photometry = IterativePSFPhotometry(
        finder=DAOStarFinder(threshold=50.0 * std_val + bkg_val, fwhm=1.5),
        psf_model=psf_model,
        fitter=LevMarLSQFitter(),
        fit_shape=(11, 11),  # Fit box size
        aperture_radius=3.0,
    )

    # 4. Run the high-fidelity extraction
    print(f"  -> Fitting Effective PSFs to sources...")
    catalog = photometry(data_es - bkg_val)

    if catalog is None or len(catalog) == 0:
        return Table(names=("x", "y", "flux"), dtype=("f8", "f8", "f8"))

    # Rename columns to match the pipeline standard
    catalog.rename_column("x_fit", "x")
    catalog.rename_column("y_fit", "y")
    catalog.rename_column("flux_fit", "flux")

    return catalog


def extract_wfi_sources(
    asdf_filepath,
    centroid_method="gaussian",
    save_diagnostic_plot=True,
    plot_outdir=".",
):
    """
    Extracts high-fidelity star positions from a Roman Level 2 WFI image.
    """
    print(f"Extracting sources from: {asdf_filepath}")
    file = rdm.open(asdf_filepath)

    try:
        sca_name = file.meta.instrument.detector
    except AttributeError:
        sca_name = "UNKNOWN"

    data_es = file.data * 2

    bkgrms = MADStdBackgroundRMS()
    mmm_bkg = MMMBackground()
    std = bkgrms(data_es)
    bkg = mmm_bkg(data_es)

    # =========================================================================
    # STRATEGY ROUTING
    # =========================================================================
    if centroid_method == "gaussian":
        sources = _extract_with_gaussian(data_es, bkg, std)
    elif centroid_method == "epsf":
        sources = _extract_with_stpsf(data_es, bkg, std, sca_name)
    else:
        file.close()
        raise ValueError(f"Unknown centroid_method: '{centroid_method}'")

    if len(sources) == 0:
        print("  -> WARNING: No sources found.")
        file.close()
        return sources

    # =========================================================================
    # CONVERGENCE: EDGE FILTERING & DQ SCREENING
    # =========================================================================
    box_size = 5
    margin = (box_size // 2) + 1
    ny, nx = data_es.shape

    edge_mask = (
        (sources["x"] > margin)
        & (sources["x"] < nx - margin)
        & (sources["y"] > margin)
        & (sources["y"] < ny - margin)
    )

    sources_edge = sources[edge_mask]

    if len(sources_edge) == 0:
        file.close()
        return Table(names=("x", "y", "flux"), dtype=("f8", "f8", "f8"))

    coords = np.column_stack((sources_edge["x"], sources_edge["y"]))
    srcaper = CircularAnnulus(coords, r_in=1, r_out=3)
    srcaper_masks = srcaper.to_mask(method="center")

    satflag = np.zeros((len(sources_edge),), dtype=int)
    i = 0

    for mask_obj in srcaper_masks:
        srcaper_dq = mask_obj.multiply(file.dq)

        if srcaper_dq is None:
            satflag[i] = 1
            i += 1
            continue

        srcaper_dq_1d = srcaper_dq[mask_obj.data > 0]
        badpix = np.logical_and(srcaper_dq_1d > 2, srcaper_dq_1d < 7)
        reallybad = np.where(srcaper_dq_1d == 1)

        if (len(srcaper_dq_1d[badpix]) > 1) or (len(srcaper_dq_1d[reallybad]) > 0):
            satflag[i] = 1
        i += 1

    final_catalog = sources_edge[np.where(satflag == 0)]

    print(f"  -> Background std: {std:.2f}, bkg: {bkg:.2f}")
    print(f"  -> Sources pre-filtering: {len(sources)}")
    print(f"  -> Sources after edge cut: {len(sources_edge)}")
    print(f"  -> Sources after DQ screening (Final): {len(final_catalog)}")

    # ---------------------------------------------------------
    # DIAGNOSTIC PLOTTING
    # ---------------------------------------------------------
    if save_diagnostic_plot:
        print("  -> Generating diagnostic plot...")
        os.makedirs(plot_outdir, exist_ok=True)

        norm = simple_norm(data_es, "asinh", vmin=0.5, vmax=4)
        positions = np.column_stack((final_catalog["x"], final_catalog["y"]))

        apertures = CircularAperture(positions, r=10) if len(positions) > 0 else None

        plt.figure(figsize=(20, 20))
        ax = plt.subplot()
        ax.set_xlabel("X [pix]")
        ax.set_ylabel("Y [pix]")
        ax.imshow(data_es, norm=norm, cmap="Greys", origin="lower")

        if apertures is not None:
            apertures.plot(color="blue", lw=0.7, alpha=0.5)

        base_name = os.path.splitext(os.path.basename(asdf_filepath))[0]
        plot_path = os.path.join(plot_outdir, f"{base_name}_sources.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  -> Saved diagnostic plot to: {plot_path}\n")

    file.close()
    return final_catalog
