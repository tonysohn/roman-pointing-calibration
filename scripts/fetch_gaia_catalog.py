"""Roman WFI_CEN Gaia DR3 query: MAST or the attached FORTRAN's local dBJ format.
Local decoding follows query_GAIA_DR3.F; selection uses a spherical cone.
No epoch propagation is performed. Local packed values are not lossless originals.
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

NSTAR = 1811709771
DATA_OFFSET = 3052 * 8500
MAP_SHAPE = (1801, 3601)
COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "ref_epoch",
    "phot_g_mean_mag",
    "ruwe",
    "phot_rp_mean_mag",
    "phot_bp_mean_mag",
    "parallax",
    "ra_error",
    "dec_error",
    "pmra_error",
    "pmdec_error",
    "parallax_error",
    "astrometric_params_solved",
]


def numeric(col):
    return np.asarray(np.ma.filled(np.ma.asarray(col, dtype=float), np.nan))


def center_from_asdf(args):
    import asdf
    import pysiaf
    import yaml

    with asdf.open(args.asdf_file, lazy_load=True) as af:
        p = af["roman"]["meta"]["pointing"]
        ra, dec, pa = [float(p[k]) for k in ("ra_v1", "dec_v1", "pa_v3")]
    if not np.all(np.isfinite([ra, dec, pa])):
        raise ValueError("Nonfinite pointing metadata")
    print(f"Telemetry: RA_V1={ra:.8f}, DEC_V1={dec:.8f}, PA_V3={pa:.8f}", flush=True)
    if args.siaf and Path(args.siaf).suffix.lower() == ".xml":
        path = Path(args.siaf).resolve()
        siaf = pysiaf.Siaf("Roman", basepath=str(path.parent), filename=path.name)
    else:
        siaf = pysiaf.Siaf("Roman")
        if args.siaf:
            with open(args.siaf) as f:
                values = yaml.safe_load(f)["WFI_CEN"]
            for k in ("V2Ref", "V3Ref"):
                setattr(siaf["WFI_CEN"], k, float(values[k]))
    ap = siaf["WFI_CEN"]
    att = pysiaf.utils.rotations.attitude(0, 0, ra, dec, pa)
    r, d = pysiaf.utils.rotations.pointing(att, ap.V2Ref, ap.V3Ref)
    print(f"WFI_CEN: RA={r:.8f}, Dec={d:.8f} deg", flush=True)
    return SkyCoord(float(r) * u.deg, float(d) * u.deg)


def unpack_uint(b, start, count):
    """Fast little-endian unsigned integer unpacking from (N, 85) uint8 array."""
    if len(b) == 0:
        return np.empty(0, dtype=np.uint64)
    # Fast path for 4-byte and 8-byte fields using native NumPy view
    if count == 4:
        return np.ndarray(
            shape=(len(b),),
            dtype="<u4",
            buffer=b,
            offset=start,
            strides=(b.strides[0],),
        ).astype(np.uint64)
    if count == 8:
        return np.ndarray(
            shape=(len(b),),
            dtype="<u8",
            buffer=b,
            offset=start,
            strides=(b.strides[0],),
        )
    if count == 7 and start + 8 <= b.shape[1]:
        # Read 8 bytes as <u8 and mask off the 8th byte (0x00FFFFFFFFFFFFFF)
        val8 = np.ndarray(
            shape=(len(b),),
            dtype="<u8",
            buffer=b,
            offset=start,
            strides=(b.strides[0],),
        )
        return val8 & np.uint64(0x00FFFFFFFFFFFFFF)
    if count == 5 and start + 8 <= b.shape[1]:
        # Read 8 bytes as <u8 and mask off top 3 bytes (0x000000FFFFFFFFFF)
        val8 = np.ndarray(
            shape=(len(b),),
            dtype="<u8",
            buffer=b,
            offset=start,
            strides=(b.strides[0],),
        )
        return val8 & np.uint64(0x000000FFFFFFFFFF)

    value = np.zeros(len(b), dtype=np.uint64)
    for k in range(count):
        value |= b[:, start + k].astype(np.uint64) << (8 * k)
    return value


def decode(b):
    """Vectorized translation of b50_rdpmugbv (zero-based byte offsets)."""
    b = np.ascontiguousarray(b)
    t = Table()
    t["source_id"] = unpack_uint(b, 41, 8).astype(np.int64)
    specs = [
        ("ra", "ra_error", 0, 7, 360.0 / 2**44, 0.0),
        ("dec", "dec_error", 7, 7, 180.0 / 2**44, -90.0),
        ("parallax", "parallax_error", 14, 5, 1e-5, -134217728e-5),
        ("pmra", "pmra_error", 19, 5, 1e-4, -134217728e-4),
        ("pmdec", "pmdec_error", 24, 5, 1e-4, -134217728e-4),
    ]
    for name, err, start, count, scale, offset in specs:
        packed = unpack_uint(b, start, count)
        t[name] = (packed >> 12).astype(float) * scale + offset
        e = (packed & 4095).astype(float)
        t[err] = np.where(e > 0, 10.0 ** (e / 500.0) / 1000.0, 0.0)
    for name, start in [
        ("phot_g_mean_mag", 29),
        ("phot_bp_mean_mag", 33),
        ("phot_rp_mean_mag", 37),
    ]:
        t[name] = (unpack_uint(b, start, 4) >> 12).astype(float) / 40000.0
    t["ref_epoch"] = np.full(len(b), 2016.0)
    t["astrometric_params_solved"] = b[:, 49].astype(np.int16)
    return t


def cone_keep(center, ra, dec, radius):
    # Haversine formula avoids arccos precision loss near the center.
    r, d = np.deg2rad(ra), np.deg2rad(dec)
    r0, d0 = center.ra.rad, center.dec.rad
    h = np.sin((d - d0) / 2) ** 2 + np.cos(d) * np.cos(d0) * np.sin((r - r0) / 2) ** 2
    return h <= np.sin(np.deg2rad(radius) / 2) ** 2


def local_ranges(index, center, radius):
    """Conservative index bounds, padded as in the FORTRAN; merge overlap."""
    ra, dec = center.ra.deg, center.dec.deg
    dmin, dmax = max(-90.0, dec - radius), min(90.0, dec + radius)
    if abs(dec) + radius >= 90:
        rmin, rmax = 0.0, 360.0
    else:
        half = np.rad2deg(
            np.arcsin(np.sin(np.deg2rad(radius)) / np.cos(center.dec.rad))
        )
        rmin, rmax = ra - half, ra + half
        if rmin < 0 or rmax > 360:
            rmin, rmax = 0.0, 360.0
    # These are the original FORTRAN's one-based bounds.
    ilo = max(1, int(1 + rmin * 10 + 0.5 - 2))
    ihi = min(3601, int(1 + rmax * 10 + 0.5 + 2) + 1)
    jlo = max(1, int(1 + (dmin + 90) * 10 + 0.5 - 2))
    jhi = min(1801, int(1 + (dmax + 90) * 10 + 0.5 + 2))
    spans = []
    for j in range(jlo - 1, jhi):
        r1 = 1 + abs(int(index[j, ilo - 1])) // 100
        r2 = 1 + abs(int(index[j, ihi - 1])) // 100
        if r2 < r1:
            raise ValueError(
                "Nonmonotonic spatial index: wrong file format or byte order"
            )
        start, end = (r1 - 1) * 100, min(r2 * 100, NSTAR)
        if not 0 <= start <= end <= NSTAR:
            raise ValueError("Spatial index outside expected catalog bounds")
        if start < end:
            spans.append((start, end))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def attach_ruwe(t, filename):
    """Read sorted 10-byte source_id/int16 RUWE records; ignore trailing index."""
    if not filename:
        return
    t0 = time.perf_counter()
    dtype = np.dtype([("source_id", "<i8"), ("packed_ruwe", "<i2")])
    if Path(filename).stat().st_size < NSTAR * 10:
        raise ValueError("RUWE file is smaller than the format in the FORTRAN")
    db = np.memmap(filename, mode="r", dtype=dtype, shape=(NSTAR,))
    ids = np.asarray(t["source_id"], dtype=np.int64)
    print("Looking up RUWE in the optional sorted-ID file...", flush=True)
    pos = np.searchsorted(db["source_id"], ids)
    valid = pos < NSTAR
    safe = np.minimum(pos, NSTAR - 1)
    valid &= db["source_id"][safe] == ids
    values = np.full(len(t), np.nan)
    values[valid] = db["packed_ruwe"][safe[valid]].astype(float) / 100.0
    values[values < 0] = np.nan
    t["ruwe"] = values
    dt = time.perf_counter() - t0
    print(
        f"RUWE matched: {np.count_nonzero(valid):,}/{len(t):,} in {dt:.2f} s",
        flush=True,
    )


def query_local(args, center):
    t_start = time.perf_counter()
    path = Path(args.local_catalog)
    if path.stat().st_size < DATA_OFFSET + NSTAR * 85:
        raise ValueError(
            "Local file is smaller than the full dBJ layout in the supplied FORTRAN"
        )
    index = np.memmap(path, mode="r", dtype="<i4", shape=MAP_SHAPE)
    # Fast spot-check on index corners instead of scanning 6.5M ints
    if abs(int(index[0, 0])) > NSTAR + 100 or abs(int(index[-1, -1])) > NSTAR + 100:
        raise ValueError(
            "Invalid spatial index; expected little-endian int32 dBJ format"
        )
    spans = local_ranges(index, center, args.radius)
    db = np.memmap(
        path, mode="r", dtype=np.uint8, offset=DATA_OFFSET, shape=(NSTAR, 85)
    )
    total = sum(end - start for start, end in spans)
    t_index = time.perf_counter() - t_start
    print(
        f"Local query: {total:,} candidate records in {len(spans)} merged spans "
        f"(index lookup: {t_index:.3f} s)",
        flush=True,
    )

    t_scan_start = time.perf_counter()
    kept_blocks = []
    examined, retained = 0, 0
    # Precompute integer G-mag threshold so we don't convert rejected stars to float
    g_thresh_int = int(np.floor(args.mag_limit * 40000.0))
    ra_scale = 360.0 / 2**44
    dec_scale = 180.0 / 2**44

    for start, end in spans:
        b = db[start:end]
        # Fast uint32 view of packed G magnitude at byte offset 29
        g_raw = unpack_uint(b, 29, 4) >> 12
        b = b[g_raw <= g_thresh_int]
        if len(b):
            ra = (unpack_uint(b, 0, 7) >> 12).astype(float) * ra_scale
            dec = (unpack_uint(b, 7, 7) >> 12).astype(float) * dec_scale - 90.0
            keep = cone_keep(center, ra, dec, args.radius)
            if np.any(keep):
                b_kept = np.array(b[keep], copy=True)
                kept_blocks.append(b_kept)
                retained += len(b_kept)
        examined += end - start
        print(f"  Examined {examined:,}/{total:,}; retained {retained:,}", flush=True)

    t_scan = time.perf_counter() - t_scan_start

    t_decode_start = time.perf_counter()
    if kept_blocks:
        all_kept = np.vstack(kept_blocks)
        t = decode(all_kept)
    else:
        t = decode(np.empty((0, 85), dtype=np.uint8))
    t_decode = time.perf_counter() - t_decode_start

    print(
        f"[Timing] Binary scan & cone filter: {t_scan:.2f} s | "
        f"Single-pass decode ({len(t):,} stars): {t_decode:.2f} s",
        flush=True,
    )

    if len(t):
        attach_ruwe(t, args.local_ruwe)
    t.meta["query_complete"] = True
    t.meta["local_catalog"] = str(path.resolve())
    t.meta["local_format"] = "query_GAIA_DR3.F packed 85-byte little-endian records"
    return t


def query_mast(args, center):
    from astroquery.mast import Mast

    params = dict(
        ra=float(center.ra.deg), dec=float(center.dec.deg), radius=args.radius
    )
    # Per-page files survive errors. Automatic resume is deliberately not attempted.
    import tempfile

    checkpoint = Path(
        tempfile.mkdtemp(
            prefix=Path(args.output).name + ".pages-",
            dir=str(Path(args.output).resolve().parent),
        )
    )
    print(f"MAST page checkpoints: {checkpoint}", flush=True)
    chunks, count, page, complete = [], 0, 1, False
    while True:
        print(f"Fetching MAST page {page}...", flush=True)
        raw = Mast.service_request(
            "Mast.Catalogs.GaiaDR3.Cone", params, pagesize=50000, page=page
        )
        n = 0 if raw is None else len(raw)
        if not n:
            complete = True
            break
        g = numeric(raw["phot_g_mean_mag"])
        part = raw[np.isfinite(g) & (g <= args.mag_limit)]
        part.write(checkpoint / f"page_{page:06d}.ecsv", format="ascii.ecsv")
        chunks.append(part)
        count += len(part)
        print(f"  Returned {n:,}; total retained {count:,}", flush=True)
        if n < 50000:
            complete = True
            break
        if args.max_rows and count >= args.max_rows:
            warnings.warn("Soft row limit reached: saving an INCOMPLETE cone search")
            break
        page += 1
    t = vstack(chunks, metadata_conflicts="silent") if chunks else Table()
    t.meta["query_complete"] = complete
    t.meta["checkpoint_directory"] = str(checkpoint)
    return t


def main():
    t_total_start = time.perf_counter()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("asdf_file")
    p.add_argument("--siaf")
    p.add_argument("--radius", type=float, default=0.8)
    p.add_argument("--mag-limit", type=float, default=19.5)
    p.add_argument("-o", "--output", default="local_gaia_catalog.ecsv")
    p.add_argument("--backend", choices=["mast", "local"], default=None)
    p.add_argument(
        "--local-catalog",
        help="Path to GAIA_DR3.dBJ; selects local backend unless overridden",
    )
    p.add_argument("--local-ruwe", help="Optional GAIA_DR3_RUWE_ID.dBJ")
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="MAST-only soft retained-row limit; 0 means unlimited. Ignored for local queries.",
    )
    args = p.parse_args()
    args.backend = args.backend or ("local" if args.local_catalog else "mast")
    if not np.isfinite(args.radius) or not 0 < args.radius < 90:
        p.error("--radius must be finite and in (0, 90) degrees")
    if not np.isfinite(args.mag_limit) or args.max_rows < 0:
        p.error("Magnitude must be finite; max-rows must be nonnegative")
    if args.backend == "local" and not args.local_catalog:
        p.error("--backend local requires --local-catalog")
    if args.local_ruwe and args.backend != "local":
        p.error("--local-ruwe is only valid for local queries")
    paths = [args.asdf_file, args.siaf]
    if args.backend == "local":
        paths += [args.local_catalog, args.local_ruwe]
    for path in paths:
        if path and not Path(path).is_file():
            p.error(f"File does not exist: {path}")
    if not Path(args.output).resolve().parent.is_dir():
        p.error("Output directory does not exist")

    center = center_from_asdf(args)
    t_query_start = time.perf_counter()
    t = (
        query_local(args, center)
        if args.backend == "local"
        else query_mast(args, center)
    )
    t_query = time.perf_counter() - t_query_start

    t = t[[name for name in COLUMNS if name in t.colnames]]
    t.meta.update(
        search_center_ra_deg=float(center.ra.deg),
        search_center_dec_deg=float(center.dec.deg),
        search_radius_deg=args.radius,
        gaia_g_mag_limit=args.mag_limit,
        backend=args.backend,
    )
    if len(t):
        positions = SkyCoord(numeric(t["ra"]) * u.deg, numeric(t["dec"]) * u.deg)
        print(f"Maximum separation: {np.max(center.separation(positions).deg):.8f} deg")

    # Write adjacent temporary file, then atomically replace final output.
    import tempfile

    t_write_start = time.perf_counter()
    fd, temporary = tempfile.mkstemp(
        prefix=Path(args.output).name + ".",
        suffix=".tmp",
        dir=str(Path(args.output).resolve().parent),
    )
    os.close(fd)
    try:
        t.write(temporary, format="ascii.ecsv", overwrite=True)
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    t_write = time.perf_counter() - t_write_start
    t_total = time.perf_counter() - t_total_start

    print(
        f"[Timing] Query + Decode: {t_query:.2f} s | ECSV Write: {t_write:.2f} s | "
        f"Total Elapsed: {t_total:.2f} s",
        flush=True,
    )
    print(
        f"Saved {len(t):,} sources to {args.output}; complete={t.meta['query_complete']}"
    )


if __name__ == "__main__":
    main()
