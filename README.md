# Roman Space Telescope Pointing Calibration

This repository contains the commissioning pipeline for the Nancy Grace Roman Space Telescope's pointing calibration. It provides a modular, flight-ready toolkit to align the 18 Wide Field Instrument (WFI) Sensor Chip Assemblies (SCAs) and calibrate the Fine Guidance Sensor (FGS) boresight relative to the spacecraft body frame.

## Flowchart

![Flowchart of roman-pointing-calibration](images/flowchart.png)

## Core Capabilities

* **High-Fidelity Astrometry:** Dual-path source extraction utilizing either rapid 2D Gaussian centroiding (`photutils`) or high-precision Effective PSFs (ePSF) generated via optical models from `stpsf`.
* **Robust Cross-Matching:** Affine-based, topology-driven cross-matching algorithm utilizing full PySIAF geometric projections to securely pair sources even under massive initial pointing errors, optical parity inversions, or severe edge distortion.
* **Kinematic Transformations:** Exact spherical Differential Velocity Aberration (DVA) scaling and proper motion propagation from Gaia DR3, optimized to handle LMC zero-parallax kinematics.
* **WFI Macroscopic Alignment:** Solves the global attitude and local focal plane geometry of the 18 SCAs, preserving true physical detector plate residuals without artificial zero-mean constraints.
* **FGS Boresight Calibration:** Solves Wahba's problem using WFI cross-matched star catalogs and flight ephemeris to derive the updated Body-to-FGS quaternion (`SCF_AC_FGS_TBL_Qb`).

## Installation

This package is designed to be installed in "editable" mode, allowing you to run the extraction and alignment scripts from any data directory while maintaining a centralized codebase.

```bash
git clone https://github.com/tonysohn/roman-pointing-calibration.git
cd roman-pointing-calibration
pip install -e .
```

## Usage

The pipeline is split into four primary runner scripts located in the `scripts/` directory.

1. Batch Source Extraction

Navigate to your directory containing the Level 2 ASDF files and execute the extraction script. This will generate `.ecsv` catalogs for each SCA.

```bash
cd /path/to/commissioning/data/
python /path/to/roman-pointing-calibration/scripts/batch_extraction.py
```

2. Fetch Reference Catalog

Download the Gaia DR3 reference catalog covering the observation field. This uses the pointing telemetry embedded in your input ASDF files to frame the correct geometry

```bash
python /path/to/roman-pointing/calibration/scripts/fetch_gaia_catalog.py r0102801002001003001_0002_wfi01_f146_cal.asdf -o local_gaia_catalog.ecsv
```
3. SCA-Level Cross-Matching

Run the robust cross-matching routine to pair the extracted SCA catalogs with the downloaded Gaia reference catalog. This step bypasses gWCS divergence limits by relying purely on PySIAF modeling to handle large initial pointing errors cleanly.

```bash
python /path/to/roman-pointing-calibration/scripts/run_crossmatch.py
```

4. Alignment & Calibration

Once the cross-matches are generated, run the master calibration solver. This executes the macro-alignment loop and calculates the boresight quaternions as well as the SIAF alignments+distortions for each SCA.

```bash
python /path/to/roman-pointing-calibration/scripts/run_calibration.py
```

## Outputs
* `calibrated_roman_siaf.yml`: The updated geometric definitions for the 18 SCAs.
* `crossmatch_results/`: Matched catalogs (.ecsv), DS9 regions (.reg), and coverage plots (.png).
* `diagnostics/`: Overlays of extracted sources, 2x2 grid quiver plots of focal plane residuals, and global attitude convergence plots.
* Standard out logging detailing the ΔRA, ΔDec, ΔPA, and the updated BAM telemetry quaternion.
