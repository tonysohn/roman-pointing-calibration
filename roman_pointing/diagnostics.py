import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pysiaf

def generate_alignment_diagnostics(
    matched_pairs_log, iteration_history, output_dir="./diagnostics"
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
    # PLOT 2 & 3: Quiver Plot and Per-SCA Statistics Table
    # =========================================================================
    if not matched_pairs_log:
        return

    # Convert the matched pairs list into a DataFrame
    # Expected format: [SCA, X, Y, RA, Dec, Flux, Mag, V2, V3, ResV2_mas, ResV3_mas]
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

    sca_stats = []

    # Calculate per-SCA statistics, including local rotation
    for sca, group in df.groupby("SCA"):
        mean_dv2 = group["ResV2_mas"].mean()
        mean_dv3 = group["ResV3_mas"].mean()
        rms = np.sqrt(np.mean(group["ResV2_mas"] ** 2 + group["ResV3_mas"] ** 2))

        # Estimate SCA rotation (dTheta) around its own center
        # Convert residuals back to arcsec for the math
        v2_local = group["V2"] - group["V2"].mean()
        v3_local = group["V3"] - group["V3"].mean()
        res_v2_arcsec = group["ResV2_mas"] / 1000.0
        res_v3_arcsec = group["ResV3_mas"] / 1000.0

        # Small angle rotation approximation: dTheta = Sum(r x dV) / Sum(r^2)
        numerator = np.sum(v2_local * res_v3_arcsec - v3_local * res_v2_arcsec)
        denominator = np.sum(v2_local**2 + v3_local**2)

        dtheta_rad = numerator / denominator if denominator != 0 else 0
        dtheta_arcsec = dtheta_rad * 206265.0  # Convert radians to arcsec

        sca_stats.append(
            [
                sca,
                len(group),
                f"{mean_dv2:.1f}",
                f"{mean_dv3:.1f}",
                f"{dtheta_arcsec:.3f}",
                f"{rms:.1f}",
            ]
        )

    # --- Setup the Figure Canvas ---
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.05)

    # --- LEFT: Quiver Plot ---
    ax_quiver = fig.add_subplot(gs[0])

    # NEW: Plot the original stars as a faint background to show SCA boundaries
    ax_quiver.scatter(df["V2"], df["V3"], s=1, color="gray", alpha=0.15, zorder=0)

    binned_v2, binned_v3 = [], []
    binned_dv2, binned_dv3 = [], []

    for sca, group in df.groupby("SCA"):
        vec_mag = np.sqrt(group["ResV2_mas"] ** 2 + group["ResV3_mas"] ** 2)
        p95 = np.nanpercentile(vec_mag, 95)
        clean_group = group[vec_mag <= p95].copy()

        if clean_group.empty:
            continue

        v2_bins = np.linspace(clean_group["V2"].min(), clean_group["V2"].max(), 3)
        v3_bins = np.linspace(clean_group["V3"].min(), clean_group["V3"].max(), 3)

        clean_group.loc[:, "V2_bin"] = pd.cut(
            clean_group["V2"], bins=v2_bins, include_lowest=True
        )
        clean_group.loc[:, "V3_bin"] = pd.cut(
            clean_group["V3"], bins=v3_bins, include_lowest=True
        )

        binned = (
            clean_group.groupby(["V2_bin", "V3_bin"], observed=False)
            .mean(numeric_only=True)
            .dropna()
        )

        binned_v2.extend(binned["V2"].values)
        binned_v3.extend(binned["V3"].values)
        binned_dv2.extend(binned["ResV2_mas"].values)
        binned_dv3.extend(binned["ResV3_mas"].values)

    # NEW: Thinner arrows and smaller heads
    q = ax_quiver.quiver(
        binned_v2,
        binned_v3,
        binned_dv2,
        binned_dv3,
        color="crimson",
        alpha=0.9,
        angles="xy",
        scale_units="xy",
        width=0.003,
        headwidth=3,
        headlength=4,
        zorder=5,
    )

    ref_length_mas = np.round(np.median([float(row[-1]) for row in sca_stats]), -1)
    if ref_length_mas <= 0:
        ref_length_mas = 10.0

    # NEW: Legend text moved below the arrow, anchor shifted left
    ax_quiver.quiverkey(
        q,
        X=0.85,
        Y=0.95,
        U=ref_length_mas,
        label=f"{ref_length_mas} mas residual",
        labelpos="S",
        fontproperties={"size": 11},
    )

    ax_quiver.set_xlabel("V2 (arcsec)", fontsize=12)
    ax_quiver.set_ylabel("V3 (arcsec)", fontsize=12)
    ax_quiver.set_title(
        "Binned Focal Plane Residual Vectors (4x4 per SCA)", fontsize=14, pad=15
    )
    ax_quiver.invert_xaxis()

    ax_quiver.set_aspect("equal", adjustable="box")
    ax_quiver.grid(True, linestyle="--", alpha=0.4)

    # --- RIGHT: Per-SCA Table ---
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    table_headers = [
        "SCA",
        "Stars",
        "ΔV2\n(mas)",
        "ΔV3\n(mas)",
        "Δθ\n(arcsec)",
        "RMS\n(mas)",
    ]
    table_data = [table_headers] + sca_stats

    # NEW: First column width increased for padding, bbox forced to full height [0,0,1,1]
    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.22, 0.12, 0.15, 0.15, 0.15, 0.15],
        bbox=[0.0, 0.0, 1.0, 1.0],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style the table headers and bold the SCA names
    for j in range(len(table_headers)):
        table[(0, j)].set_text_props(weight="bold")
        table[(0, j)].set_facecolor("#e0e0e0")

    for i in range(1, len(table_data)):
        # NEW: Bold the first column (SCA names)
        table[(i, 0)].set_text_props(weight="bold")

        color = "#f9f9f9" if i % 2 == 0 else "white"
        for j in range(len(table_headers)):
            table[(i, j)].set_facecolor(color)

    ax_table.set_title("Per-SCA Geometric Bias Summary", fontsize=14, pad=15)

    plt.savefig(
        os.path.join(output_dir, "focal_plane_quiver_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()