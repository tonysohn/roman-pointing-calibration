import copy
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysiaf
from matplotlib.gridspec import GridSpec


def generate_alignment_diagnostics(
    matched_pairs_log,
    iteration_history,
    output_dir="./diagnostics",
    calibrated_siaf_params=None,
    nominal_siaf=None,
):
    os.makedirs(output_dir, exist_ok=True)

    if iteration_history:
        plt.figure(figsize=(8, 5))
        iterations = np.arange(1, len(iteration_history) + 1)
        plt.plot(iterations, iteration_history, marker="o", linestyle="-", linewidth=2)
        plt.yscale("log")
        plt.xlabel("Iteration Number", fontsize=12)
        plt.ylabel("RMS Residual (arcsec) [Log Scale]", fontsize=12)
        plt.title("Global Attitude Refinement Convergence", fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attitude_convergence.png"), dpi=300)
        plt.close()

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

    # Construct a temporary SIAF object to evaluate the new calibrated geometry
    new_siaf = copy.deepcopy(nominal_siaf)
    if calibrated_siaf_params:
        for sca, params in calibrated_siaf_params.items():
            new_siaf[sca].V2Ref = params["V2Ref"]
            new_siaf[sca].V3Ref = params["V3Ref"]
            new_siaf[sca].V3IdlYAngle = params["V3IdlYAngle"]

    sca_stats = []
    for sca, group in df.groupby("SCA"):
        sca_label = sca.replace("_FULL", "")
        mean_dv2 = group["ResV2_mas"].mean()
        mean_dv3 = group["ResV3_mas"].mean()
        rms = np.sqrt(np.mean(group["ResV2_mas"] ** 2 + group["ResV3_mas"] ** 2))

        v2_local = group["V2"] - group["V2"].mean()
        v3_local = group["V3"] - group["V3"].mean()
        res_v2_arcsec = group["ResV2_mas"] / 1000.0
        res_v3_arcsec = group["ResV3_mas"] / 1000.0

        numerator = np.sum(v2_local * res_v3_arcsec - v3_local * res_v2_arcsec)
        denominator = np.sum(v2_local**2 + v3_local**2)
        dtheta_arcsec = (numerator / denominator if denominator != 0 else 0) * 206265.0

        # Now properly calculated against the untouched nominal_siaf
        dx_siaf = calibrated_siaf_params[sca]["V2Ref"] - nominal_siaf[sca].V2Ref
        dy_siaf = calibrated_siaf_params[sca]["V3Ref"] - nominal_siaf[sca].V3Ref

        sca_stats.append(
            [
                sca_label,
                len(group),
                f"{dx_siaf:.1f}",
                f"{dy_siaf:.1f}",
                f"{mean_dv2:.1f}",
                f"{mean_dv3:.1f}",
                f"{dtheta_arcsec:.3f}",
                f"{rms:.1f}",
            ]
        )

    fig, (ax_quiver, ax_table) = plt.subplots(1, 2, figsize=(18, 9))
    ax_table.axis("off")

    # =========================================================================
    # NEW QUIVER PLOT: 3x3 Grid Old vs New SIAF
    # =========================================================================
    for sca_name in nominal_siaf.apertures:
        if "FULL" not in sca_name:
            continue

        # Create a 3x3 pixel grid across the 4088x4088 detector
        x_pix = np.linspace(4, 4084, 3)
        y_pix = np.linspace(4, 4084, 3)
        xx, yy = np.meshgrid(x_pix, y_pix)

        v2_old, v3_old = nominal_siaf[sca_name].sci_to_tel(xx.flatten(), yy.flatten())
        v2_new, v3_new = new_siaf[sca_name].sci_to_tel(xx.flatten(), yy.flatten())

        dv2 = v2_new - v2_old
        dv3 = v3_new - v3_old

        # Plot physical transformation vectors (scaled slightly so 20 arcsec is visible)
        ax_quiver.quiver(
            v2_old,
            v3_old,
            dv2,
            dv3,
            color="red",
            angles="xy",
            scale_units="xy",
            scale=0.1,
            width=0.003,
        )

        # Add background label
        ax_quiver.text(
            np.mean(v2_old),
            np.mean(v3_old),
            sca_name.replace("WFI", "").replace("_FULL", ""),
            color="black",
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
            alpha=0.2,
        )

    # Reference Arrow for the scaled Quiver
    ax_quiver.annotate(
        "",
        xy=(0.82, 0.92),
        xytext=(0.72, 0.92),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5),
    )
    ax_quiver.text(
        0.77,
        0.94,
        "10x Exaggerated Shift",
        color="black",
        ha="center",
        transform=ax_quiver.transAxes,
        fontsize=11,
    )

    # Mark Old vs New WFI_CEN
    ax_quiver.scatter(
        nominal_siaf["WFI_CEN"].V2Ref,
        nominal_siaf["WFI_CEN"].V3Ref,
        marker="X",
        color="gray",
        s=150,
        alpha=0.5,
        label="WFI_CEN (Old)",
    )
    ax_quiver.scatter(
        new_siaf["WFI_CEN"].V2Ref,
        new_siaf["WFI_CEN"].V3Ref,
        marker="X",
        color="dodgerblue",
        s=200,
        label="WFI_CEN (New)",
    )

    ax_quiver.invert_xaxis()
    ax_quiver.set_xlabel("V2 (arcsec)", fontsize=12)
    ax_quiver.set_ylabel("V3 (arcsec)", fontsize=12)
    ax_quiver.set_title("Focal Plane Alignment Updates (Old -> New)", fontsize=14)
    ax_quiver.grid(True, linestyle="--", alpha=0.5)
    ax_quiver.legend(loc="lower left", fontsize=11)

    # =========================================================================
    # TABLE: Renamed Headers
    # =========================================================================
    table_headers = [
        "SCA",
        "Stars",
        "V2 Shift\n(arcsec)",
        "V3 Shift\n(arcsec)",
        "Mean_ΔV2\n(mas)",
        "Mean_ΔV3\n(mas)",
        "Mean_Δθ\n(arcsec)",
        "2D_RMS\n(mas)",
    ]

    table_data = [table_headers] + sca_stats
    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.08, 0.08, 0.17, 0.17, 0.13, 0.13, 0.13, 0.11],
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for j in range(len(table_headers)):
        table[(0, j)].set_text_props(weight="bold")
        table[(0, j)].set_facecolor("#e0e0e0")

    for i in range(1, len(table_data)):
        table[(i, 0)].set_text_props(weight="bold")
        color = "#f9f9f9" if i % 2 == 0 else "white"
        for j in range(len(table_headers)):
            table[(i, j)].set_facecolor(color)

    ax_table.set_title("Per-SCA Geometric Bias Summary", fontsize=15, pad=20)
    plt.savefig(
        os.path.join(output_dir, "focal_plane_quiver_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def stash_generate_alignment_diagnostics(
    matched_pairs_log,
    iteration_history,
    output_dir="./diagnostics",
    calibrated_siaf_params=None,
    nominal_siaf=None,
):
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # PLOT 1: Global Attitude Convergence
    # =========================================================================
    if iteration_history:
        plt.figure(figsize=(8, 5))
        iterations = np.arange(1, len(iteration_history) + 1)
        plt.plot(
            iterations,
            iteration_history,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
        )
        plt.yscale("log")
        plt.xlabel("Iteration Number", fontsize=12)
        plt.ylabel("RMS Residual (arcsec) [Log Scale]", fontsize=12)
        plt.title("Global Attitude Refinement Convergence", fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xticks(iterations)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attitude_convergence.png"), dpi=300)
        plt.close()

    # =========================================================================
    # DATA PREP: Calculate Table Statistics
    # =========================================================================
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

    # Safely load a nominal SIAF for comparison
    if nominal_siaf is None:
        nominal_siaf = pysiaf.Siaf("Roman")

    sca_stats = []

    for sca, group in df.groupby("SCA"):
        sca_label = sca.replace("_FULL", "")
        mean_dv2 = group["ResV2_mas"].mean()
        mean_dv3 = group["ResV3_mas"].mean()
        rms = np.sqrt(np.mean(group["ResV2_mas"] ** 2 + group["ResV3_mas"] ** 2))

        v2_local = group["V2"] - group["V2"].mean()
        v3_local = group["V3"] - group["V3"].mean()
        res_v2_arcsec = group["ResV2_mas"] / 1000.0
        res_v3_arcsec = group["ResV3_mas"] / 1000.0

        numerator = np.sum(v2_local * res_v3_arcsec - v3_local * res_v2_arcsec)
        denominator = np.sum(v2_local**2 + v3_local**2)
        dtheta_arcsec = (numerator / denominator if denominator != 0 else 0) * 206265.0

        # Calculate physical shift from nominal SIAF
        if calibrated_siaf_params and sca in calibrated_siaf_params and nominal_siaf:
            dx_siaf = calibrated_siaf_params[sca]["V2Ref"] - nominal_siaf[sca].V2Ref
            dy_siaf = calibrated_siaf_params[sca]["V3Ref"] - nominal_siaf[sca].V3Ref
        else:
            dx_siaf, dy_siaf = 0.0, 0.0

        sca_stats.append(
            [
                sca_label,
                len(group),
                f"{dx_siaf:.1f}",
                f"{dy_siaf:.1f}",
                f"{mean_dv2:.1f}",
                f"{mean_dv3:.1f}",
                f"{dtheta_arcsec:.3f}",
                f"{rms:.1f}",
            ]
        )

    # =========================================================================
    # PLOT 2 & 3: Figure Layout Setup
    # =========================================================================
    fig, (ax_quiver, ax_table) = plt.subplots(1, 2, figsize=(18, 9))
    ax_table.axis("off")  # Hide axes for the table subplot

    # =========================================================================
    # PLOT 2: Quiver Plot Construction
    # =========================================================================
    # Scatter all stars lightly in the background
    ax_quiver.scatter(
        df["V2"], df["V3"], s=0.3, color="gray", alpha=0.3, edgecolors="none"
    )

    # Add large, semi-transparent SCA numbers over each chip
    for sca, group in df.groupby("SCA"):
        mean_v2 = group["V2"].mean()
        mean_v3 = group["V3"].mean()
        sca_num = sca.replace("WFI", "").replace("_FULL", "")

        ax_quiver.text(
            mean_v2,
            mean_v3,
            sca_num,
            color="black",
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
            alpha=0.2,
            zorder=5,
        )

    # Subsample quiver arrows so Matplotlib doesn't crash drawing 120,000 vectors
    step = max(1, len(df) // 5000)

    # --- DYNAMIC SCALING LOGIC ---
    # Calculate the 90th percentile of the residual magnitudes
    res_mag = np.hypot(df["ResV2_mas"], df["ResV3_mas"])
    p90 = np.percentile(res_mag, 90)

    # Determine a clean, human-readable reference arrow value based on the data
    if p90 < 10:
        ref_val = 5
    elif p90 < 40:
        ref_val = 20
    elif p90 < 100:
        ref_val = 50
    elif p90 < 300:
        ref_val = 200
    else:
        ref_val = 500

    # Dynamically adjust the Matplotlib scale (higher scale = smaller arrows)
    # This ratio ensures the reference arrow always draws at the exact same visual length on the PNG
    dynamic_scale = ref_val * 125

    # Plot residual vectors normally using the dynamic scale
    q = ax_quiver.quiver(
        df["V2"].iloc[::step],
        df["V3"].iloc[::step],
        df["ResV2_mas"].iloc[::step],
        df["ResV3_mas"].iloc[::step],
        color="red",
        scale=dynamic_scale,
        width=0.002,
    )

    # Draw a clean, explicitly scaled reference arrow and label
    ax_quiver.annotate(
        "",
        xy=(0.82, 0.92),
        xytext=(0.72, 0.92),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5),
    )
    ax_quiver.text(
        0.77,
        0.94,
        f"{ref_val} mas residual",
        color="black",
        ha="center",
        transform=ax_quiver.transAxes,
        fontsize=11,
    )

    # Formatting the Quiver plot axes
    ax_quiver.invert_xaxis()  # V2 is inverted (runs right-to-left)
    ax_quiver.set_xlabel("V2 (arcsec)", fontsize=12)
    ax_quiver.set_ylabel("V3 (arcsec)", fontsize=12)
    ax_quiver.set_title("Focal Plane Residual Vectors", fontsize=14)
    ax_quiver.grid(True, linestyle="--", alpha=0.5)

    # Add WFI_CEN marker
    if calibrated_siaf_params and "WFI_CEN" in calibrated_siaf_params:
        cen_v2 = calibrated_siaf_params["WFI_CEN"]["V2Ref"]
        cen_v3 = calibrated_siaf_params["WFI_CEN"]["V3Ref"]
        ax_quiver.scatter(
            cen_v2,
            cen_v3,
            marker="X",
            color="dodgerblue",
            s=200,
            edgecolor="white",
            linewidth=1.5,
            zorder=10,
            label="WFI_CEN",
        )
        ax_quiver.legend(loc="lower left", fontsize=11)

    # =========================================================================
    # PLOT 3: Table Construction
    # =========================================================================
    table_headers = [
        "SCA",
        "Stars",
        "SCA_Shift_V2\n(arcsec)",
        "SCA_Shift_V3\n(arcsec)",
        "Mean_ΔV2\n(mas)",
        "Mean_ΔV3\n(mas)",
        "Mean_Δθ\n(arcsec)",
        "2D_RMS\n(mas)",
    ]

    table_data = [table_headers] + sca_stats

    # Redistributed colWidths (summing to 1.0) to accommodate wider titles
    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.08, 0.08, 0.17, 0.17, 0.13, 0.13, 0.13, 0.11],
        bbox=[0.0, 0.0, 1.0, 1.0],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style headers and alternating rows
    for j in range(len(table_headers)):
        table[(0, j)].set_text_props(weight="bold")
        table[(0, j)].set_facecolor("#e0e0e0")

    for i in range(1, len(table_data)):
        table[(i, 0)].set_text_props(weight="bold")
        color = "#f9f9f9" if i % 2 == 0 else "white"
        for j in range(len(table_headers)):
            table[(i, j)].set_facecolor(color)

    ax_table.set_title("Per-SCA Geometric Bias Summary", fontsize=15, pad=20)

    # =========================================================================
    # SAVE & CLOSE
    # =========================================================================
    plt.savefig(
        os.path.join(output_dir, "focal_plane_quiver_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
