"""
download_geodata.py
===================
Prueft und laedt Geodaten aus dem Geoportal Thueringen fuer eine BBox.

Die Kachel-URLs werden direkt konstruiert (kein Feed-Parsing noetig).
URL-Muster aus dem Geoportal Thueringen:
  DGM:  https://...hoehendaten/DGM/dgm_2020-2025/dgm1_32_{E}_{N}_1_th_2020-2025.zip
  DOM:  https://...hoehendaten/DOM/dom_2020-2025/dom1_32_{E}_{N}_1_th_2020-2025.zip
  LoD2: https://...LoD2/LoD2_32_{E}_{N}_2_TH.zip

Verwendung:
    python download_geodata.py --bbox 641000 642000 5648000 5649000
    python download_geodata.py --bbox 641000 642000 5648000 5649000 --download
    python download_geodata.py --bbox 634000 637000 5615000 5618000 --download --no-atkis
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ─────────────────────────────────────────────
# URL-Muster (direkte Konstruktion ohne Feed)
# ─────────────────────────────────────────────

BASE_GEO = "https://geoportal.geoportal-th.de"

def url_dgm(e_km, n_km):
    return f"{BASE_GEO}/hoehendaten/DGM/dgm_2020-2025/dgm1_32_{e_km}_{n_km}_1_th_2020-2025.zip"

def url_dom(e_km, n_km):
    return f"{BASE_GEO}/hoehendaten/DOM/dom_2020-2025/dom1_32_{e_km}_{n_km}_1_th_2020-2025.zip"

def url_lod2(e_km, n_km):
    return f"{BASE_GEO}/3dgebaeude/LoD2/LoD2_32_{e_km}_{n_km}_2_TH.zip"

# ATKIS: Thueringen komplett (nicht kachelweise)
DL_BASE   = f"{BASE_GEO}/gaialight-th/_apps/dladownload/index.php"
ATKIS_ZIPS = {
    "ver": f"{DL_BASE}?type=1&service=dlm&thema=ver",
    "gew": f"{DL_BASE}?type=1&service=dlm&thema=gew",
    "sie": f"{DL_BASE}?type=1&service=dlm&thema=sie",
    "veg": f"{DL_BASE}?type=1&service=dlm&thema=veg",
}

TILE_SIZE_DGM  = 1000   # 1x1 km
TILE_SIZE_LOD2 = 2000   # 2x2 km


# ─────────────────────────────────────────────
# Kachel-Koordinaten berechnen
# ─────────────────────────────────────────────

def required_tiles(west, ost, sued, nord, tile_size):
    """Gibt alle Kachel-SW-Ecken (e, n) in Metern zurueck."""
    e_start = (west       // tile_size) * tile_size
    n_start = (sued       // tile_size) * tile_size
    e_end   = ((ost  - 1) // tile_size) * tile_size
    n_end   = ((nord - 1) // tile_size) * tile_size
    tiles = []
    e = e_start
    while e <= e_end:
        n = n_start
        while n <= n_end:
            tiles.append((e, n))
            n += tile_size
        e += tile_size
    return tiles


# ─────────────────────────────────────────────
# Lokale Kacheln pruefen
# ─────────────────────────────────────────────

def find_local_tile(folder, e_km, n_km, extensions=(".tif", ".tiff", ".gml", ".GML")):
    """Sucht lokal nach Kachel mit _32_{e_km}_{n_km}_ im Dateinamen."""
    folder = Path(folder)
    if not folder.exists():
        return None
    pattern = f"_32_{e_km}_{n_km}_"
    for ext in extensions:
        for f in folder.glob(f"*{ext}"):
            if pattern in f.name:
                return f
    return None


def check_tiles_local(folder, tiles, label):
    vorhanden, fehlend = [], []
    for e, n in tiles:
        e_km, n_km = e // 1000, n // 1000
        f = find_local_tile(folder, e_km, n_km)
        if f:
            vorhanden.append((e_km, n_km, f))
        else:
            fehlend.append((e_km, n_km))
    return vorhanden, fehlend


# ─────────────────────────────────────────────
# Download (ZIP entpacken)
# ─────────────────────────────────────────────

def download_and_extract(url, dest_dir, label=""):
    """Laedt ZIP herunter, entpackt relevante Dateien, loescht ZIP."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname    = url.split("/")[-1].split("?")[0]
    zip_path = dest_dir / fname

    if zip_path.exists():
        zip_path.unlink()

    print(f"  ↓ {label or fname}", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=300, stream=True)
        if resp.status_code == 404:
            print(f"[404] nicht verfügbar")
            return None  # None = nicht vorhanden, kein Fehler
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_mb = zip_path.stat().st_size / 1024 / 1024

        KEEP_EXTS = {".tif", ".tiff", ".gml", ".GML", ".shp", ".dbf",
                     ".shx", ".prj", ".cpg", ".qpj"}
        extracted = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                fname_m = Path(member).name
                if not fname_m or fname_m.startswith("."):
                    continue
                if Path(fname_m).suffix.lower() in KEEP_EXTS:
                    target = dest_dir / fname_m
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted.append(fname_m)
        zip_path.unlink()

        if extracted:
            print(f"OK ({size_mb:.1f} MB, {len(extracted)} Datei(en))")
            return True
        else:
            print(f"FEHLER: Keine bekannten Dateien im ZIP")
            return False

    except Exception as ex:
        print(f"FEHLER: {ex}")
        if zip_path.exists(): zip_path.unlink()
        return False


def download_tiles(missing_km, url_func, dest_dir, label, workers=4):
    """Laedt fehlende Kacheln parallel herunter. None = nicht vorhanden (404), kein Fehler."""
    tasks = [(url_func(e_km, n_km), dest_dir, f"E{e_km} N{n_km}")
             for e_km, n_km in missing_km]

    ok, not_available = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_and_extract, url, dest, lbl): lbl
                for url, dest, lbl in tasks}
        for fut in as_completed(futs):
            result = fut.result()
            if result is True:
                ok += 1
            elif result is None:
                not_available += 1
    if not_available:
        print(f"  ℹ  {not_available} Kachel(n) nicht im Geoportal (außerhalb Thüringens)")
        print(f"     → Welt wird ohne LoD2-Gebäude für diese Kacheln generiert")
    return ok


# ─────────────────────────────────────────────
# ATKIS (Thueringen komplett)
# ─────────────────────────────────────────────

def check_atkis(atkis_dir):
    fehlend = []
    for thema in ATKIS_ZIPS:
        d = Path(atkis_dir) / thema
        if d.exists() and list(d.glob("*.shp")):
            n = len(list(d.glob("*.shp")))
            print(f"  [OK]    atkis/{thema}/  ({n} SHPs)")
        else:
            print(f"  [FEHLT] atkis/{thema}/")
            fehlend.append(thema)
    return fehlend


def download_atkis(atkis_dir, themen=None):
    atkis_dir = Path(atkis_dir)
    for thema in (themen or list(ATKIS_ZIPS.keys())):
        thema_dir = atkis_dir / thema
        if thema_dir.exists() and list(thema_dir.glob("*.shp")):
            print(f"  [vorhanden] atkis/{thema}/")
            continue
        url      = ATKIS_ZIPS[thema]
        zip_path = atkis_dir / f"atkis_{thema}.zip"
        print(f"  ↓ atkis/{thema}...", end=" ", flush=True)
        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            atkis_dir.mkdir(parents=True, exist_ok=True)
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_mb = zip_path.stat().st_size / 1024 / 1024
            print(f"({size_mb:.1f} MB) entpacke...", end=" ", flush=True)
            thema_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    fname = Path(member).name
                    if fname and not fname.startswith("."):
                        with zf.open(member) as src, open(thema_dir / fname, "wb") as dst:
                            dst.write(src.read())
            zip_path.unlink()
            n_shp = len(list(thema_dir.glob("*.shp")))
            print(f"OK ({n_shp} SHPs)")
        except Exception as ex:
            print(f"FEHLER: {ex}")
            if zip_path.exists(): zip_path.unlink()


# ─────────────────────────────────────────────
# Hauptfunktion
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Geodaten-Check und Download fuer thueringen2minecraft",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python download_geodata.py --bbox 641000 642000 5648000 5649000
  python download_geodata.py --bbox 641000 642000 5648000 5649000 --download
  python download_geodata.py --bbox 634000 637000 5615000 5618000 --download --no-atkis
        """
    )
    p.add_argument("--bbox", nargs=4, type=int, metavar=("WEST","OST","SUED","NORD"),
                   required=True)
    p.add_argument("--download",  action="store_true")
    p.add_argument("--dgm-dir",   default="dgm")
    p.add_argument("--dom-dir",   default="dom")
    p.add_argument("--lod2-dir",  default="LoD2")
    p.add_argument("--atkis-dir", default="atkis")
    p.add_argument("--no-dom",    action="store_true")
    p.add_argument("--no-atkis",  action="store_true")
    p.add_argument("--no-lod2",   action="store_true")
    p.add_argument("--workers",   type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    west, ost, sued, nord = args.bbox
    all_ok = True

    print("=" * 60)
    print("  thueringen2minecraft  –  Daten-Check & Download")
    print("=" * 60)
    print(f"  BBox:    W{west} E{ost}  S{sued} N{nord}")
    print(f"  Groesse: {(ost-west)/1000:.1f} × {(nord-sued)/1000:.1f} km")
    print(f"  Modus:   {'DOWNLOAD' if args.download else 'NUR PRUEFEN'}")
    print()

    # ── DGM ──────────────────────────────────────────────
    print("── DGM (1m Raster, 1×1 km) " + "─" * 33)
    dgm_tiles = required_tiles(west, ost, sued, nord, TILE_SIZE_DGM)
    vorhanden, fehlend = check_tiles_local(args.dgm_dir, dgm_tiles, "DGM")
    for e_km, n_km, f in vorhanden:
        print(f"  [OK]    E{e_km} N{n_km}  →  {f.name}")
    for e_km, n_km in fehlend:
        print(f"  [FEHLT] E{e_km} N{n_km}  →  {url_dgm(e_km, n_km)}")
        all_ok = False
    if fehlend:
        if args.download:
            n_ok = download_tiles(fehlend, url_dgm, args.dgm_dir, "DGM", args.workers)
            print(f"  → {n_ok}/{len(fehlend)} DGM-Kacheln heruntergeladen")
        else:
            print(f"  → {len(fehlend)} fehlend. Starte mit --download")
    print()

    # ── DOM ──────────────────────────────────────────────
    if not args.no_dom:
        print("── DOM (1m Raster, 1×1 km, optional) " + "─" * 23)
        dom_tiles = required_tiles(west, ost, sued, nord, TILE_SIZE_DGM)
        vorhanden, fehlend = check_tiles_local(args.dom_dir, dom_tiles, "DOM")
        for e_km, n_km, f in vorhanden:
            print(f"  [OK]    E{e_km} N{n_km}  →  {f.name}")
        for e_km, n_km in fehlend:
            print(f"  [FEHLT] E{e_km} N{n_km}")
        if fehlend:
            if args.download:
                n_ok = download_tiles(fehlend, url_dom, args.dom_dir, "DOM", args.workers)
                print(f"  → {n_ok}/{len(fehlend)} DOM-Kacheln heruntergeladen")
            else:
                print(f"  → {len(fehlend)} fehlend (optional, --download zum Laden)")
        print()

    # ── LoD2 ─────────────────────────────────────────────
    if not args.no_lod2:
        print("── LoD2 (CityGML 3D-Gebaeude, 2×2 km) " + "─" * 21)
        lod2_tiles = required_tiles(west, ost, sued, nord, TILE_SIZE_LOD2)
        vorhanden, fehlend = check_tiles_local(args.lod2_dir, lod2_tiles, "LoD2")
        for e_km, n_km, f in vorhanden:
            print(f"  [OK]    E{e_km} N{n_km}  →  {f.name}")
        for e_km, n_km in fehlend:
            print(f"  [FEHLT] E{e_km} N{n_km}  →  {url_lod2(e_km, n_km)}")
        if fehlend:
            if args.download:
                n_ok = download_tiles(fehlend, url_lod2, args.lod2_dir, "LoD2", args.workers)
                if n_ok > 0:
                    print(f"  → {n_ok}/{len(fehlend)} LoD2-Kacheln heruntergeladen")
                # 404 = außerhalb Thüringen → kein Fehler, Welt ohne Gebäude generierbar
            else:
                print(f"  → {len(fehlend)} fehlend. Starte mit --download")
                all_ok = False  # nur wenn noch nicht versucht zu laden
        print()

    # ── ATKIS ────────────────────────────────────────────
    if not args.no_atkis:
        print("── ATKIS Basis-DLM (Thueringen komplett) " + "─" * 19)
        fehlend_atkis = check_atkis(args.atkis_dir)
        if fehlend_atkis:
            all_ok = False
            if args.download:
                download_atkis(args.atkis_dir, fehlend_atkis)
            else:
                print(f"  → {len(fehlend_atkis)} fehlend. Starte mit --download")
        print()

    # ── Ergebnis ─────────────────────────────────────────
    print("=" * 60)
    if all_ok:
        print("  ✓ Alle Pflichtdaten vorhanden!")
        print(f"\n  Welt generieren:")
        print(f"  python thueringen2minecraft.py --bbox {west} {ost} {sued} {nord}")
    else:
        if args.download:
            print("  ✓ Download abgeschlossen.")
            print(f"\n  Welt generieren (LoD2 evtl. nicht verfügbar für diesen Bereich):")
            print(f"  python thueringen2minecraft.py --bbox {west} {ost} {sued} {nord}")
        else:
            print("  ✗ Fehlende Daten!")
            print(f"  python download_geodata.py --bbox {west} {ost} {sued} {nord} --download")
    print("=" * 60)


if __name__ == "__main__":
    main()