"""Data access: resolve targets, query IRSA (SIA2), download cutouts, demo data."""
import io, os, re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

# Check available names with: Irsa.list_collections()
COLLECTION = os.getenv("SPHEREX_COLLECTION", "spherex_qr2")
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "40"))


@dataclass
class Frame:
    data: np.ndarray
    flags: np.ndarray
    wcs: WCS
    mjd: float
    det: int


def resolve(text: str) -> SkyCoord:
    """Accepts a name, 'ra dec' in degrees, or sexagesimal like '08h55m10.8s -07d14m42.5s'."""
    t = text.strip()
    parts = t.replace(",", " ").split()
    if len(parts) == 2:
        try:
            return SkyCoord(float(parts[0]) * u.deg, float(parts[1]) * u.deg)
        except ValueError:
            pass
    if re.fullmatch(r"[\d\s+\-.:hmsd]+", t):
        return SkyCoord(t, unit=(u.hourangle, u.deg))
    return SkyCoord.from_name(t)  # CDS Sesame name lookup


def _download(url, ra, dec, size_deg, mjd):
    """IRSA cutout: append center and size (degrees) to the image URL."""
    r = httpx.get(f"{url}?center={ra:.7f},{dec:.7f}&size={size_deg:.5f}",
                  timeout=180, follow_redirects=True)
    r.raise_for_status()
    with fits.open(io.BytesIO(r.content)) as h:
        img, hdr = h["IMAGE"].data.astype(float), h["IMAGE"].header
        flags = h["FLAGS"].data.copy() if "FLAGS" in h else np.zeros(img.shape, int)
    m = re.search(r"2D(\d)", url)
    return Frame(img, flags, WCS(hdr), mjd, int(m.group(1)) if m else 0)


def real_frames(c: SkyCoord, size_deg: float):
    from astroquery.ipac.irsa import Irsa
    tab = Irsa.query_sia(pos=(c, 10 * u.arcsec), collection=COLLECTION)
    if len(tab) == 0:
        return []
    tab = tab[np.argsort(tab["t_min"])][:MAX_IMAGES]
    jobs = [(str(r["access_url"]), c.ra.deg, c.dec.deg, size_deg, float(r["t_min"])) for r in tab]

    def safe(j):
        try:
            return _download(*j)
        except Exception:  # skip a bad image rather than failing the whole request
            return None

    with ThreadPoolExecutor(8) as ex:
        out = [f for f in ex.map(safe, jobs) if f is not None]
    return sorted(out, key=lambda f: f.mjd)


def demo_frames(c: SkyCoord, size_deg: float, n: int = 14):
    """Synthetic frames (static stars + one mover, jittered pointing/rotation) for offline testing."""
    rng = np.random.default_rng(1)
    npix, scale = max(8, int(size_deg * 3600 / 6.2)), 6.2 / 3600
    stars = rng.uniform(-size_deg / 2.2, size_deg / 2.2, (12, 2))
    amps = rng.uniform(30, 300, 12)
    cosd = np.cos(np.deg2rad(c.dec.deg))
    yy, xx = np.mgrid[:npix, :npix]
    frames = []
    for i in range(n):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crpix = [npix / 2 + 1, npix / 2 + 1]
        w.wcs.crval = [c.ra.deg + rng.normal(0, 1e-3), c.dec.deg + rng.normal(0, 1e-3)]
        th = np.deg2rad(rng.uniform(-30, 30))
        w.wcs.cd = scale * np.array([[-np.cos(th), np.sin(th)], [np.sin(th), np.cos(th)]])
        img = rng.normal(100, 5, (npix, npix))
        src = [(dx, dy, a) for (dx, dy), a in zip(stars, amps)]
        src.append(((i - n / 2) * size_deg / (1.6 * n), 0.1 * size_deg, 250.0))  # the mover
        for dx, dy, a in src:
            x, y = w.world_to_pixel_values(c.ra.deg + dx / cosd, c.dec.deg + dy)
            img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2)
        frames.append(Frame(img, np.zeros(img.shape, int), w, 60800 + 3.0 * i, i % 6 + 1))
    return frames
