import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysiaf


def generate_alignment_diagnostics(
    matched_pairs_log,
    iteration_history,
    output_dir="./diagnostics",
    calibrated_siaf_params=None,
    nominal_siaf=None,
):
    os.makedirs(output_dir, exist_ok=True)

    if not matched_pairs_log:
        return

    columns = [
        "SCA",
        "X",
        "Y",
        "RA",
        "Dec",
        "Flux",
        "Mag",
        "V2",
        "V3",
        "ResV2_mas",
        "ResV3_mas",
    ]
    df = pd.DataFrame(matched_pairs_log, columns=columns)

    if nominal_siaf is None:
        nominal_siaf = pysiaf.Siaf("Roman")

    new_siaf = copy.deepcopy(nominal_siaf)
    if calibrated_siaf_params:
        for sca, params in calibrated_siaf_params.items():
            new_siaf[sca].V2Ref = params["V2Ref"]
            new_siaf[sca].V3Ref = params["V3Ref"]
            new_siaf[sca].V3IdlYAngle = params["V3IdlYAngle"]

            if "Sci2IdlX10" in params:
                poly_coeffs = new_siaf[sca].get_polynomial_coefficients()

                # Auto-detect the degree natively using PySIAF
                num_coeffs = len(poly_coeffs["Sci2IdlX"])
                fit_degree = pysiaf.utils.polynomial.polynomial_degree(num_coeffs)

                mapping = {
                    idx: f"{d}{y_deg}"
                    for idx, (d, y_deg) in enumerate(
                        [(d, y) for d in range(fit_degree + 1) for y in range(d + 1)]
                    )
                }

                for i in range(num_coeffs):
                    suffix = mapping.get(i, "00")
                    poly_coeffs["Sci2IdlX"][i] = (
                        0.0
                        if suffix == "00"
                        else params.get(f"Sci2IdlX{suffix}", poly_coeffs["Sci2IdlX"][i])
                    )
                    poly_coeffs["Sci2IdlY"][i] = (
                        0.0
                        if suffix == "00"
                        else params.get(f"Sci2IdlY{suffix}", poly_coeffs["Sci2IdlY"][i])
                    )

                lower_poly_coeffs = {k.lower(): v for k, v in poly_coeffs.items()}
                new_siaf[sca].set_polynomial_coefficients(**lower_poly_coeffs)

    bulk_dtheta = new_siaf["WFI_CEN"].V3IdlYAngle - nominal_siaf["WFI_CEN"].V3IdlYAngle
    sca_stats = []

    for sca, group in df.groupby("SCA"):
        sca_label = sca.replace("_FULL", "")

        dx_siaf = new_siaf[sca].V2Ref - nominal_siaf[sca].V2Ref
        dy_siaf = new_siaf[sca].V3Ref - nominal_siaf[sca].V3Ref
        dtheta_int_arcmin = (
            new_siaf[sca].V3IdlYAngle - nominal_siaf[sca].V3IdlYAngle - bulk_dtheta
        ) * 60.0

        old_scale_x = nominal_siaf[sca].Sci2IdlX10
        new_scale_x = new_siaf[sca].Sci2IdlX10
        dscale_x_ppm = (
            (new_scale_x / old_scale_x - 1.0) * 1e6 if old_scale_x != 0 else 0.0
        )

        old_scale_y = nominal_siaf[sca].Sci2IdlY11
        new_scale_y = new_siaf[sca].Sci2IdlY11
        dscale_y_ppm = (
            (new_scale_y / old_scale_y - 1.0) * 1e6 if old_scale_y != 0 else 0.0
        )

        rms = np.sqrt(np.mean(group["ResV2_mas"] ** 2 + group["ResV3_mas"] ** 2))

        sca_stats.append(
            [
                sca_label,
                len(group),
                f"{dx_siaf:.1f}",
                f"{dy_siaf:.1f}",
                f"{dtheta_int_arcmin:.3f}",
                f"{dscale_x_ppm:.1f}",
                f"{dscale_y_ppm:.1f}",
                f"{rms:.1f}",
            ]
        )

    fig, (ax_quiver, ax_table) = plt.subplots(1, 2, figsize=(16, 9))
    plt.subplots_adjust(wspace=0.05)
    ax_table.axis("off")

    for sca_name, group in df.groupby("SCA"):
        v2_pts, v3_pts = new_siaf[sca_name].sci_to_tel(group["X"], group["Y"])
        ax_quiver.scatter(
            v2_pts, v3_pts, s=0.3, color="gray", alpha=0.3, edgecolors="none"
        )

    all_v2, all_v3, all_dv2, all_dv3 = [], [], [], []

    for sca_name in nominal_siaf.apertures:
        if "FULL" not in sca_name:
            continue

        v2c, v3c = new_siaf[sca_name].corners("tel")
        v2c = np.append(v2c, v2c[0])
        v3c = np.append(v3c, v3c[0])
        ax_quiver.plot(v2c, v3c, color="black", linewidth=1.0, alpha=0.5)

        # 6x6 grid including edges
        x_pix = np.linspace(1, 4088, 10)
        y_pix = np.linspace(1, 4088, 10)
        xx, yy = np.meshgrid(x_pix, y_pix)

        x_idl_old, y_idl_old = nominal_siaf[sca_name].sci_to_idl(
            xx.flatten(), yy.flatten()
        )
        x_idl_new, y_idl_new = new_siaf[sca_name].sci_to_idl(xx.flatten(), yy.flatten())

        dx_idl = x_idl_new - x_idl_old
        dy_idl = y_idl_new - y_idl_old

        theta = np.deg2rad(new_siaf[sca_name].V3IdlYAngle)
        dv2 = dx_idl * np.cos(theta) - dy_idl * np.sin(theta)
        dv3 = dx_idl * np.sin(theta) + dy_idl * np.cos(theta)

        v2_plot, v3_plot = new_siaf[sca_name].idl_to_tel(x_idl_old, y_idl_old)

        all_v2.extend(v2_plot)
        all_v3.extend(v3_plot)
        all_dv2.extend(dv2)
        all_dv3.extend(dv3)

        ax_quiver.text(
            np.mean(v2_plot),
            np.mean(v3_plot),
            sca_name.replace("WFI", "").replace("_FULL", ""),
            color="black",
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
            alpha=0.2,
            zorder=5,
        )

    max_mag = np.max(np.hypot(all_dv2, all_dv3)) if len(all_dv2) > 0 else 1.0

    # Force a tiny non-zero floor to prevent log10(0) and Matplotlib scale=0 errors
    if max_mag <= 0.0:
        max_mag = 1e-6

    # Scale increased to make arrows 1/3 the length of the previous version
    target_plot_length = 400.0 / 3.0
    dynamic_scale = max_mag / target_plot_length

    Q = ax_quiver.quiver(
        all_v2,
        all_v3,
        all_dv2,
        all_dv3,
        color="red",
        angles="xy",
        scale_units="xy",
        scale=dynamic_scale,
        width=0.0015,
        headwidth=3,
        headlength=4,
        headaxislength=3,  # Thinner arrows, smaller heads
    )

    exponent = np.floor(np.log10(max_mag))
    mantissa = max_mag / 10**exponent
    ref_val = (
        1 * 10**exponent
        if mantissa < 2
        else 2 * 10**exponent
        if mantissa < 5
        else 5 * 10**exponent
    )

    # Clean label format
    ax_quiver.quiverkey(
        Q,
        0.80,
        0.94,
        ref_val,
        f"{ref_val:g} arcsec",
        labelpos="E",
        fontproperties={"size": 11},
        coordinates="axes",
    )

    ax_quiver.scatter(
        nominal_siaf["WFI_CEN"].V2Ref,
        nominal_siaf["WFI_CEN"].V3Ref,
        marker="x",
        color="gray",
        s=80,
        linewidths=2,
        label="WFI_CEN (Old)",
    )
    ax_quiver.scatter(
        new_siaf["WFI_CEN"].V2Ref,
        new_siaf["WFI_CEN"].V3Ref,
        marker="x",
        color="dodgerblue",
        s=80,
        linewidths=2,
        label="WFI_CEN (New)",
    )
    ax_quiver.legend(loc="lower left", fontsize=11)

    ax_quiver.invert_xaxis()
    ax_quiver.set_aspect("equal")
    ax_quiver.set_xlabel("V2 (arcsec)", fontsize=12)
    ax_quiver.set_ylabel("V3 (arcsec)", fontsize=12)
    ax_quiver.set_title(
        "SCA Internal Geometry Update (Bulk Shift/Rotation Removed)", fontsize=14
    )
    ax_quiver.grid(True, linestyle="--", alpha=0.5)

    # =========================================================================
    # PLOT 3: Table Construction
    # =========================================================================
    table_headers = [
        "SCA",
        "Stars",
        "V2 Shift\n(arcsec)",
        "V3 Shift\n(arcsec)",
        r"$\Delta\theta_{internal}$" + "\n(arcmin)",
        "ΔScale X\n(ppm)",
        "ΔScale Y\n(ppm)",
        "2D_RMS\n(mas)",
    ]

    table_data = [table_headers] + sca_stats

    # Steal width from the right 4 columns and give it to the left 4
    col_widths = [0.13, 0.10, 0.13, 0.13, 0.12, 0.12, 0.12, 0.11]

    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for j in range(len(table_headers)):
        table[(0, j)].set_text_props(weight="bold")
        table[(0, j)].set_facecolor("#e0e0e0")
    for i in range(1, len(table_data)):
        table[(i, 0)].set_text_props(weight="bold")
        table[(i, j)].set_facecolor("#f9f9f9" if i % 2 == 0 else "white")

    ax_table.set_title(
        "SIAF Update Summary: Shifts, Rotations, and Scales", fontsize=15, y=1.0, pad=6
    )

    plt.savefig(
        os.path.join(output_dir, "focal_plane_quiver_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
