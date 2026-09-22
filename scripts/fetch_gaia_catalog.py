#!/usr/bin/env python3
"""
Downloads Gaia DR3 reference catalog for Roman pointing calibration.
Dynamically calculates the focal plane center using ASDF telemetry and SIAF.
"""

import argparse
import os
import asdf
import pysiaf
import yaml
from astroquery.gaia import Gaia

def main():
    parser = argparse.ArgumentParser(description="Download Gaia reference catalog for a Roman observation.")
    parser.add_argument("asdf_file", help="Path to a representative ASDF file from the observation.")
    parser.add_argument("--siaf", type=str, default=None, help="Path to calibrated SIAF (.xml or .yml)")
    parser.add_argument("--mag-limit", type=float, default=19.0, help="Faintest Gaia G-band magnitude.")
    parser.add_argument("--radius", type=float, default=0.8, help="Search radius in degrees to cover WFI FOV.")
    parser.add_argument("-o", "--output", default="local_gaia_catalog.ecsv", help="Output filename.")
    args = parser.parse_args()

    print(f"Extracting pointing telemetry from: {args.asdf_file}")
    with asdf.open(args.asdf_file, lazy_load=True) as f:
        ra_v1 = f["roman"]["meta"]["pointing"]["ra_v1"]
        dec_v1 = f["roman"]["meta"]["pointing"]["dec_v1"]
        pa_v3 = f["roman"]["meta"]["pointing"]["pa_v3"]

    print(f"Telemetry Pointing: RA_V1={ra_v1:.4f}, DEC_V1={dec_v1:.4f}, PA_V3={pa_v3:.4f}")

    # Load SIAF to dynamically calculate the WFI focal plane center
    if args.siaf and os.path.exists(args.siaf):
        if args.siaf.lower().endswith('.xml'):
            base_dir = os.path.dirname(os.path.abspath(args.siaf))
            file_name = os.path.basename(args.siaf)
            rsiaf = pysiaf.Siaf("Roman", basepath=base_dir, filename=file_name)
        else:
            rsiaf = pysiaf.Siaf("Roman")
            with open(args.siaf, "r") as yf:
                cal_data = yaml.safe_load(yf)
            if "WFI_CEN" in cal_data:
                rsiaf["WFI_CEN"].V2Ref = cal_data["WFI_CEN"]["V2Ref"]
                rsiaf["WFI_CEN"].V3Ref = cal_data["WFI_CEN"]["V3Ref"]
    else:
        rsiaf = pysiaf.Siaf("Roman")

    # Project WFI_CEN to sky coordinates
    att = pysiaf.utils.rotations.attitude(0, 0, ra_v1, dec_v1, pa_v3)
    wfi_cen = rsiaf["WFI_CEN"]
    ra_cen, dec_cen = pysiaf.utils.rotations.pointing(att, wfi_cen.V2Ref, wfi_cen.V3Ref)
    
    print(f"WFI Boresight (WFI_CEN) computed at: RA={ra_cen:.5f}, Dec={dec_cen:.5f}")
    print(f"Querying Gaia DR3 within {args.radius} degrees...")

    query = f"""
    SELECT source_id, ra, dec, pmra, pmdec, ref_epoch, phot_g_mean_mag, ruwe, phot_rp_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_cen}, {dec_cen}, {args.radius}))
    AND phot_g_mean_mag <= {args.mag_limit}
    """

    job = Gaia.launch_job_async(query)
    catalog = job.get_results()
    
    catalog.write(args.output, format='ascii.ecsv', overwrite=True)
    print(f"Successfully saved {len(catalog)} sources to {args.output}")

if __name__ == "__main__":
    main()