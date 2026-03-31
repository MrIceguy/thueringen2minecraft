"""
tile_loader.py – Mehrere Kacheln laden & zusammensetzen
========================================================
Unterstützt .tif (bevorzugt, schnell) und .xyz (Fallback, langsam).

Kachelschema Thüringen (1km × 1km, UTM32N / EPSG:25832):
  dgm1_32_634_5615_1_th_2020-2025.tif
           ^^^  ^^^^
           |    Northing in km → UTM-N = 5615 * 1000
           Easting in km       → UTM-E =  634 * 1000
"""

import numpy as np
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_TILE_RE = re.compile(r"_32_(\d+)_(\d+)_", re.IGNORECASE)
TILE_SIZE_M = 1000


# ─────────────────────────────────────────────
# Dateiname → UTM-Koordinaten
# ─────────────────────────────────────────────

def parse_tile_coords(filepath):
    m = _TILE_RE.search(Path(filepath).stem)
    if not m:
        return None
    return int(m.group(1)) * 1000, int(m.group(2)) * 1000


# ─────────────────────────────────────────────
# Einzelne Kachel laden (.tif bevorzugt, .xyz Fallback)
# ─────────────────────────────────────────────

def load_tile(filepath):
    """
    Lädt eine Kachel. Gibt (array_2d, origin_x, origin_y, res) zurück.
    origin = UTM-Koordinate der SW-Ecke (links unten).
    """
    p = Path(filepath)
    if p.suffix.lower() in (".tif", ".tiff"):
        return _load_tif(p)
    elif p.suffix.lower() == ".xyz":
        return _load_xyz(p)
    else:
        raise ValueError(f"Unbekanntes Format: {p.suffix}")


def _load_tif(filepath):
    import rasterio
    with rasterio.open(filepath) as src:
        data      = src.read(1).astype(np.float32)
        nodata    = src.nodata
        transform = src.transform
        res       = float(src.res[0])

        if nodata is not None:
            data[data == nodata] = np.nan

        # origin = SW-Ecke
        origin_x = transform.c                          # West-Kante
        origin_y = transform.f + transform.e * src.height  # Süd-Kante
        return data, origin_x, origin_y, res


def _load_xyz(filepath):
    data      = np.loadtxt(filepath, dtype=np.float32)
    xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]
    unique_x  = np.unique(xs)
    unique_y  = np.unique(ys)
    res       = float(unique_x[1] - unique_x[0]) if len(unique_x) > 1 else 1.0
    nx, ny    = len(unique_x), len(unique_y)

    x_idx = np.searchsorted(unique_x, xs)
    y_idx = ny - 1 - np.searchsorted(unique_y, ys)

    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    grid[y_idx, x_idx] = zs

    return grid, float(unique_x.min()), float(unique_y.min()), res


# ─────────────────────────────────────────────
# Alle Kacheln eines Ordners laden & mosaikieren
# ─────────────────────────────────────────────

def load_all_tiles(folder, bbox=None, workers=4, label="tiles"):
    """
    Lädt alle .tif/.xyz-Dateien in `folder`, optional gefiltert auf `bbox`.

    Gibt zurück: (merged_array, origin_x, origin_y, resolution)
    """
    folder = Path(folder)

    # .tif bevorzugen, .xyz als Fallback
    tif_files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    xyz_files = sorted(folder.glob("*.xyz"))
    all_files = tif_files if tif_files else xyz_files

    if not all_files:
        raise FileNotFoundError(
            f"Keine .tif/.xyz-Dateien in '{folder}/'\n"
            f"Bitte Kacheln vom Geoportal Thüringen herunterladen."
        )

    fmt = "TIF" if tif_files else "XYZ"

    # Kacheln mit Koordinaten sammeln & auf BBox filtern
    tiles_to_load = []
    for f in all_files:
        coords = parse_tile_coords(f)
        if coords is None:
            print(f"  ⚠  Dateinamen-Schema unbekannt, übersprungen: {f.name}")
            continue
        east, north = coords
        if bbox:
            if (east + TILE_SIZE_M <= bbox["west"] or east  >= bbox["east"] or
                north + TILE_SIZE_M <= bbox["south"] or north >= bbox["north"]):
                continue
        tiles_to_load.append((f, east, north))

    if not tiles_to_load:
        avail = [parse_tile_coords(f) for f in all_files]
        avail = [c for c in avail if c]
        raise ValueError(
            f"Keine {label}-Kacheln in '{folder}/' überlappen die BBox.\n"
            f"Vorhandene Kacheln: {avail}"
        )

    print(f"  {len(tiles_to_load)} {label}-Kacheln ({fmt}):")
    for f, e, n in sorted(tiles_to_load, key=lambda x: (x[2], x[1])):
        print(f"    E{e//1000} N{n//1000}  {f.name}")

    # Ausdehnung des Mosaiks
    all_e = [e for _, e, _ in tiles_to_load]
    all_n = [n for _, _, n in tiles_to_load]
    min_e, max_e = min(all_e), max(all_e) + TILE_SIZE_M
    min_n, max_n = min(all_n), max(all_n) + TILE_SIZE_M

    # Auflösung aus erster Kachel
    _, _, _, res = load_tile(tiles_to_load[0][0])
    total_w = int(round((max_e - min_e) / res))
    total_h = int(round((max_n - min_n) / res))
    merged  = np.full((total_h, total_w), np.nan, dtype=np.float32)

    print(f"  Mosaik: {total_w} × {total_h} px  "
          f"({total_w * res / 1000:.1f} × {total_h * res / 1000:.1f} km)  "
          f"Auflösung: {res:.1f}m")

    def _place(args):
        f, east, north = args
        try:
            grid, _, _, tile_res = load_tile(f)
        except Exception as ex:
            print(f"  ⚠  {f.name}: {ex}")
            return
        col0 = int(round((east        - min_e) / res))
        row0 = int(round((max_n - north - TILE_SIZE_M) / res))
        r1   = min(total_h, row0 + grid.shape[0])
        c1   = min(total_w, col0 + grid.shape[1])
        gr   = r1 - row0
        gc   = c1 - col0
        merged[row0:r1, col0:c1] = grid[:gr, :gc]

    print(f"  Lade ({workers} parallel)...", end=" ")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_place, t) for t in tiles_to_load]
        for i, fut in enumerate(as_completed(futs), 1):
            print(f"{i}", end=" ", flush=True)
    print()

    # BBox-Crop
    if bbox:
        merged, min_e, min_n = _crop_to_bbox(merged, min_e, min_n, res, max_n, bbox)

    valid = merged[~np.isnan(merged)]
    if len(valid):
        print(f"  Höhe: {valid.min():.1f} – {valid.max():.1f} m  (∅ {valid.mean():.1f} m)")

    return merged, min_e, min_n, res


def _crop_to_bbox(arr, ox, oy, res, max_n_before_crop, bbox):
    h, w   = arr.shape
    col0   = max(0, int((bbox["west"]  - ox)              / res))
    col1   = min(w, int((bbox["east"]  - ox)              / res))
    row0   = max(0, int((max_n_before_crop - bbox["north"]) / res))
    row1   = min(h, int((max_n_before_crop - bbox["south"]) / res))
    cropped = arr[row0:row1, col0:col1]
    new_ox  = ox + col0 * res
    new_oy  = oy + (h - row1) * res
    print(f"  → BBox-Crop: {cropped.shape[1]} × {cropped.shape[0]} px")
    return cropped, new_ox, new_oy


# ─────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────

def make_rasterio_transform(origin_x, origin_y, res, height):
    from rasterio.transform import from_origin
    north = origin_y + height * res
    return from_origin(origin_x, north, res, res)


def list_available_tiles(folder):
    folder = Path(folder)
    files  = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.xyz"))
    if not files:
        print(f"  Keine Kacheln in {folder}/")
        return
    print(f"\n  {folder}/")
    print(f"  {'Datei':<55} {'E(km)':>6} {'N(km)':>6}")
    print(f"  {'-'*55} {'-'*6} {'-'*6}")
    for f in files:
        c = parse_tile_coords(f)
        if c:
            print(f"  {f.name:<55} {c[0]//1000:>6} {c[1]//1000:>6}")
        else:
            print(f"  {f.name:<55} {'?':>6} {'?':>6}")


def get_bbox_from_folder(folder):
    folder = Path(folder)
    files  = list(folder.glob("*.tif")) + list(folder.glob("*.xyz"))
    coords = [c for c in (parse_tile_coords(f) for f in files) if c]
    if not coords:
        return None
    return {
        "west":  min(c[0] for c in coords),
        "east":  max(c[0] for c in coords) + TILE_SIZE_M,
        "south": min(c[1] for c in coords),
        "north": max(c[1] for c in coords) + TILE_SIZE_M,
    }
