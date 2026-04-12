"""
atkis_layer.py – ATKIS Basis-DLM Thüringen → Minecraft-Blöcke
==============================================================
Ersetzt osm_layer.py. Liest amtliche Vektordaten direkt vom
Geoportal Thüringen statt Overpass/OSM.

Ordnerstruktur erwartet:
    atkis/
        ver/    ← atkis.basis-dlm.ver.zip entpackt  (Verkehr)
        gew/    ← atkis.basis-dlm.gew.zip entpackt  (Gewässer)
        sie/    ← atkis.basis-dlm.sie.zip entpackt  (Siedlung)
        veg/    ← atkis.basis-dlm.veg.zip entpackt  (Vegetation)

Format: Shapefile (.shp) oder NAS/GML (.xml/.gml) – beides wird erkannt.

Installation:
    pip install geopandas shapely rasterio pyproj
    (kein osmnx, kein osmium nötig)
"""

import numpy as np
from pathlib import Path
import warnings

CHURCH_POINTS = []  # [(utm_x, utm_y)] — Kirchturm-Standorte aus ATKIS AX_Turm
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Block-Definitionen  (gleiche Werte wie vorher → Rest des Skripts unverändert)
# ─────────────────────────────────────────────

OSM_CLASSES = {
    1:  ("minecraft:water",               4,  10, "Fluss / Bach"),
    2:  ("minecraft:water",               2,   9, "See / Teich"),
    10: ("minecraft:gray_concrete",        8,   8, "Autobahn / Bundesstraße"),
    11: ("minecraft:light_gray_concrete",  6,   7, "Landstraße"),
    12: ("minecraft:smooth_stone",         5,   6, "Kreisstraße"),
    13: ("minecraft:stone_slab",           4,   5, "Gemeindestraße"),
    14: ("minecraft:stone_bricks",         3,   4, "Wohnstraße"),
    15: ("minecraft:cobblestone",          2,   3, "Feldweg / Wirtschaftsweg"),
    16: ("minecraft:dirt_path",            2,   2, "Fußweg / Pfad"),
    17: ("minecraft:oak_planks",           3,   3, "Radweg"),
    20: ("minecraft:rail",                 2,   6, "Bahnstrecke"),
    30: ("minecraft:grass_block",          0,   1, "Grünland / Wiese"),
    31: ("minecraft:moss_block",           0,   1, "Wald"),
    32: ("minecraft:sand",                 0,   2, "Sandfläche"),
}

def get_osm_block(osm_value):
    entry = OSM_CLASSES.get(int(osm_value))
    return entry[0] if entry else None

def is_water(osm_value):
    return int(osm_value) in (1, 2)

def print_osm_stats(raster):
    print("\n── ATKIS-Layer Statistik ───────────────────")
    for cls, (block, _, _, desc) in OSM_CLASSES.items():
        n = int((raster == cls).sum())
        if n > 0:
            print(f"  {desc:<35} {n:>8,} px  → {block}")
    print("────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# ATKIS-Objektarten → Block-Klassen
# Quellen: ATKIS-Objektartenkatalog Basis-DLM
# ─────────────────────────────────────────────

# Verkehr (ver): Attribut "OAK" oder "objart" oder "BEZ"
VERKEHR_MAP = {
    # Straßen nach Widmung / Kennung
    "AX_Autobahn":                  10,
    "AX_Bundesstrasse":             10,
    "AX_Landesstrasse":             11,
    "AX_Kreisstrasse":              12,
    "AX_Gemeindestrasse":           13,
    "AX_Strassenverkehr":           14,   # allg. Straßenfläche
    "AX_Weg":                       15,
    "AX_FussWandRadweg":            16,
    "AX_Bahnstrecke":               20,
    "AX_Seilbahn":                  20,
    # numerische OAK-Codes (kommen je nach Exportformat vor)
    "42001": 10,  # Autobahn
    "42003": 10,  # Bundesstraße
    "42005": 11,  # Landesstraße / Staatsstraße
    "42006": 12,  # Kreisstraße
    "42007": 13,  # Gemeindestraße
    "42008": 14,  # Wirtschaftsweg klassifiziert
    "42009": 15,  # Wirtschaftsweg
    "42010": 15,  # Weg
    "42015": 16,  # Fußweg
    "42016": 17,  # Radweg
    "42014": 20,  # Gleis
    "44001": 20,  # Bahnstrecke
    "44004": 20,  # Seilbahn
}

# Breitenangaben in Metern (für Linien-Puffer)
VERKEHR_BREITE = {
    10: 8, 11: 6, 12: 5, 13: 4, 14: 3, 15: 2, 16: 2, 17: 3, 20: 2,
}

# Gewässer (gew)
GEWAESSER_MAP = {
    "AX_Fliessgewaesser":    1,
    "AX_Gewaesserachse":     1,
    "AX_Wasserlauf":         1,
    "AX_Stehendesgewaesser": 2,
    "AX_Hafenbecken":        2,
    "AX_Meer":               2,
    "44300": 2,  # stehendes Gewässer
    "44001": 1,  # Fließgewässer (Achse)
    "44002": 1,  # Fließgewässer (Fläche)
    "44006": 1,  # Kanal
    "44007": 2,  # See / Teich
    "44008": 2,  # Hafenbecken
}

GEWAESSER_BREITE = {
    "AX_Fliessgewaesser": 8,
    "AX_Gewaesserachse":  4,
    "default":            4,
}

# Vegetation (veg)
VEGETATION_MAP = {
    "AX_Wald":              31,
    "AX_Gehoelz":          31,
    "AX_Heide":             30,
    "AX_Moor":              30,
    "AX_Sumpf":             30,
    "AX_Landwirtschaft":    30,
    "AX_Grünland":          30,
    "AX_Ackerland":         30,
    "43001": 31,  # Wald
    "43002": 31,  # Gehölz
    "43003": 30,  # Heide
    "43004": 30,  # Moor
    "43005": 30,  # Sumpf
    "43006": 30,  # Grünland
    "41001": 30,  # Ackerland
    "41002": 30,  # Gartenland
    "41003": 30,  # Obstplantage
}

# Siedlung (sie) – nur Grünflächen, Friedhöfe etc. (Gebäude kommen aus LoD2)
SIEDLUNG_MAP = {
    "AX_SportFreizeitUndErholungsflaeche": 30,
    "AX_Friedhof":      30,
    "AX_Grünanlage":    30,
    "41008": 30,  # Sport-/Freizeitanlage
    "41009": 30,  # Campingplatz
    "41010": 30,  # Friedhof
    "41006": 30,  # Grünanlage / Park
}


# ─────────────────────────────────────────────
# Format-Erkennung & Laden
# ─────────────────────────────────────────────

def _find_files(folder, extensions):
    """Sucht rekursiv nach Dateien mit bestimmten Endungen."""
    folder = Path(folder)
    if not folder.exists():
        return []
    found = []
    for ext in extensions:
        found.extend(folder.rglob(f"*{ext}"))
    return sorted(found)


def _load_geodataframe(folder, bbox_utm):
    """
    Lädt alle Vektor-Dateien aus einem Ordner als GeoDataFrame.
    Unterstützt: .shp, .gpkg, .geojson, .xml (NAS/GML), .gml

    Gibt ein einzelnes GeoDataFrame zurück (alle Dateien zusammengeführt).
    """
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    shp_files  = _find_files(folder, [".shp"])
    gpkg_files = _find_files(folder, [".gpkg"])
    gml_files  = _find_files(folder, [".xml", ".gml"])

    gdfs = []

    # Shapefile (bevorzugt)
    for f in shp_files:
        try:
            gdf = gpd.read_file(f)
            if len(gdf) == 0:
                continue
            # CRS prüfen / setzen
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:25832")
            elif gdf.crs.to_epsg() != 25832:
                gdf = gdf.to_crs("EPSG:25832")
            gdfs.append(gdf)
        except Exception as ex:
            print(f"    ⚠  {f.name}: {ex}")

    # GeoPackage
    for f in gpkg_files:
        try:
            gdf = gpd.read_file(f)
            if gdf.crs and gdf.crs.to_epsg() != 25832:
                gdf = gdf.to_crs("EPSG:25832")
            gdfs.append(gdf)
        except Exception as ex:
            print(f"    ⚠  {f.name}: {ex}")

    # NAS / GML (falls kein Shapefile)
    if not gdfs:
        for f in gml_files:
            try:
                gdf = gpd.read_file(f, driver="GML")
                if gdf.crs and gdf.crs.to_epsg() != 25832:
                    gdf = gdf.to_crs("EPSG:25832")
                if len(gdf):
                    gdfs.append(gdf)
            except Exception as ex:
                print(f"    ⚠  {f.name}: {ex}")

    if not gdfs:
        return None

    import pandas as pd
    merged = pd.concat(gdfs, ignore_index=True)
    gdf_all = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:25832")

    # BBox-Filter
    bbox_poly = shapely_box(bbox_utm["west"], bbox_utm["south"],
                            bbox_utm["east"],  bbox_utm["north"])
    gdf_clipped = gdf_all[gdf_all.geometry.intersects(bbox_poly)].copy()

    return gdf_clipped if len(gdf_clipped) else None


def _detect_type_column(gdf):
    """Findet die Spalte mit dem Objektart-Schlüssel."""
    candidates = ["objart", "OAK", "OBJART", "objart_txt", "OAT",
                  "TYPENAME", "typename", "type", "TYPE",
                  "BEZ", "bez", "GFK", "gfk", "FKT", "fkt"]
    for col in candidates:
        if col in gdf.columns:
            return col
    return None


def _get_class(val, mapping):
    """Mapped einen Rohwert auf eine Block-Klasse."""
    if val is None:
        return None
    s = str(val).strip()
    if s in mapping:
        return mapping[s]
    # Numerischen Code extrahieren (z.B. "42001" aus "AX_Strassenverkehr_42001")
    import re
    m = re.search(r'\b(\d{5})\b', s)
    if m and m.group(1) in mapping:
        return mapping[m.group(1)]
    return None


# ─────────────────────────────────────────────
# Hauptfunktion: ATKIS → Features
# ─────────────────────────────────────────────

def load_osm_layers(bbox, osm_file=None, cache_dir=None):
    """
    Kompatible Schnittstelle zu osm_layer.py.
    Lädt ATKIS-Vektordaten aus atkis/{ver,gew,sie,veg}/.

    Gibt Liste von (shapely_geom, klassen_wert, priorität) zurück.
    """
    atkis_root = Path("atkis")
    if not atkis_root.exists():
        print("  ⚠  Ordner 'atkis/' nicht gefunden.")
        print("     Bitte Struktur anlegen:")
        print("       atkis/ver/  ← atkis.basis-dlm.ver.zip entpackt")
        print("       atkis/gew/  ← atkis.basis-dlm.gew.zip entpackt")
        print("       atkis/sie/  ← atkis.basis-dlm.sie.zip entpackt")
        print("       atkis/veg/  ← atkis.basis-dlm.veg.zip entpackt")
        return []

    features = []

    # ── Verkehr ──────────────────────────────
    ver_dir = atkis_root / "ver"
    if ver_dir.exists():
        print("  Lade Verkehr (Straßen, Bahn)...")
        gdf = _load_geodataframe(ver_dir, bbox)
        if gdf is not None:
            col = _detect_type_column(gdf)
            n = 0
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                cls = _get_class(row.get(col) if col else None, VERKEHR_MAP)
                if cls is None:
                    cls = 14   # Fallback: Wohnstraße
                width = VERKEHR_BREITE.get(cls, 3)
                prio  = OSM_CLASSES[cls][2]
                try:
                    if geom.geom_type in ("LineString", "MultiLineString"):
                        buffered = geom.buffer(width / 2)
                    else:
                        buffered = geom   # Fläche direkt
                    if buffered.is_valid and not buffered.is_empty:
                        features.append((buffered, cls, prio))
                        n += 1
                except Exception:
                    pass
            print(f"    {n:,} Verkehrs-Features")
        else:
            print("    (keine Dateien gefunden)")
    else:
        print(f"  ⚠  {ver_dir}/ fehlt – Verkehr übersprungen")

    # ── Gewässer ─────────────────────────────
    gew_dir = atkis_root / "gew"
    if gew_dir.exists():
        print("  Lade Gewässer (Flüsse, Seen)...")
        gdf = _load_geodataframe(gew_dir, bbox)
        if gdf is not None:
            col = _detect_type_column(gdf)
            n = 0
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                raw = row.get(col) if col else None
                cls = _get_class(raw, GEWAESSER_MAP) or 1
                prio = OSM_CLASSES[cls][2]
                try:
                    if geom.geom_type in ("LineString", "MultiLineString"):
                        raw_str = str(raw) if raw else ""
                        width = GEWAESSER_BREITE.get(
                            raw_str, GEWAESSER_BREITE["default"])
                        buffered = geom.buffer(width / 2)
                    else:
                        buffered = geom
                    if buffered.is_valid and not buffered.is_empty:
                        features.append((buffered, cls, prio))
                        n += 1
                except Exception:
                    pass
            print(f"    {n:,} Gewässer-Features")
        else:
            print("    (keine Dateien gefunden)")
    else:
        print(f"  ⚠  {gew_dir}/ fehlt – Gewässer übersprungen")

    # ── Vegetation ───────────────────────────
    veg_dir = atkis_root / "veg"
    if veg_dir.exists():
        print("  Lade Vegetation (Wald, Grünland)...")
        gdf = _load_geodataframe(veg_dir, bbox)
        if gdf is not None:
            col = _detect_type_column(gdf)
            n = 0
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                cls = _get_class(row.get(col) if col else None, VEGETATION_MAP)
                if cls is None:
                    continue   # unbekannte Vegetationsart ignorieren
                try:
                    if geom.is_valid:
                        features.append((geom, cls, OSM_CLASSES[cls][2]))
                        n += 1
                except Exception:
                    pass
            print(f"    {n:,} Vegetations-Features")
        else:
            print("    (keine Dateien gefunden)")
    else:
        print(f"  ⚠  {veg_dir}/ fehlt – Vegetation übersprungen")


    # ── Siedlung (Grünflächen + Türme/Kirchen) ──
    sie_dir = atkis_root / "sie"
    if sie_dir.exists():
        print("  Lade Siedlungs-Grünflächen, Türme und Kirchen...")
        gdf = _load_geodataframe(sie_dir, bbox)
        if gdf is not None:
            col = _detect_type_column(gdf)
            n = 0
            n_turm = 0
            CHURCH_POINTS.clear()
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                objart = str(row.get("OBJART", "")).strip()
                # AX_Turm (51001) → Kirchturm-Koordinate speichern
                if objart == "51001":
                    bwf = str(row.get("BWF", "")).strip()
                    nam = str(row.get("NAM", "")).strip()
                    pt  = geom.centroid
                    print(f"      AX_Turm BWF={bwf} NAM='{nam}' @ E{pt.x:.0f} N{pt.y:.0f}")
                    CHURCH_POINTS.append((pt.x, pt.y))
                    n_turm += 1
                    continue
                cls = _get_class(row.get(col) if col else None, SIEDLUNG_MAP)
                if cls is None:
                    continue
                try:
                    if geom.is_valid:
                        features.append((geom, cls, OSM_CLASSES[cls][2]))
                        n += 1
                except Exception:
                    pass
            print(f"    {n:,} Siedlungs-Features  |  {n_turm} Tuerme/Kirchen (AX_Turm)")
            for cx, cy in CHURCH_POINTS:
                print(f"      Kirchturm @ E{cx:.0f} N{cy:.0f}")

    print(f"\n  → {len(features):,} ATKIS-Features gesamt")
    return features


# ─────────────────────────────────────────────
# Rasterisieren (identisch zu osm_layer.py)
# ─────────────────────────────────────────────

def burn_osm_to_raster(features, bbox, shape):
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds as tfb

    h, w      = shape
    transform = tfb(bbox["west"], bbox["south"],
                    bbox["east"], bbox["north"], w, h)
    raster    = np.zeros(shape, dtype=np.uint8)

    # Prioritäts-Raster einmalig aufbauen statt pro Feature neu berechnen
    prio_raster = np.zeros(shape, dtype=np.uint8)

    sorted_features = sorted(features, key=lambda x: x[2])
    total = len(sorted_features)
    step  = max(1, total // 20)   # alle 5% ein Update

    print(f"  Rasterisiere {total:,} Features auf {shape[1]}×{shape[0]} px...", flush=True)

    for i, (geom, cls, prio) in enumerate(sorted_features):
        if i % step == 0:
            pct = i / total * 100
            n_done = int((raster > 0).sum())
            desc = OSM_CLASSES.get(cls, ("?",))[3] if cls in OSM_CLASSES else "?"
            print(f"  {pct:5.1f}%  [{i:>5}/{total}]  {n_done:,} px belegt  (aktuell: {desc})",
                  end="\r", flush=True)

        if geom is None or geom.is_empty:
            continue
        try:
            burned = rasterize(
                [(geom, cls)], out_shape=shape,
                transform=transform, fill=0,
                dtype=np.uint8, all_touched=True,
            )
            mask = (burned > 0) & (prio >= prio_raster)
            raster[mask]      = cls
            prio_raster[mask] = prio
        except Exception:
            pass

    n = int((raster > 0).sum())
    print(f"\n  Raster fertig: {n:,} px belegt ({n / raster.size * 100:.1f}%)")
    return raster


# ─────────────────────────────────────────────
# Diagnose: Spalten einer ATKIS-Datei anzeigen
# ─────────────────────────────────────────────

def inspect_atkis(folder):
    """
    Zeigt Spalten und Beispielwerte aller Shapefiles in einem Ordner.
    Aufruf: python -c "from atkis_layer import inspect_atkis; inspect_atkis('atkis/ver')"
    """
    import geopandas as gpd
    folder = Path(folder)
    for f in sorted(folder.rglob("*.shp")):
        print(f"\n{'─'*60}")
        print(f"  {f.name}")
        try:
            gdf = gpd.read_file(f)
            print(f"  {len(gdf):,} Features  |  Spalten: {list(gdf.columns)}")
            for col in gdf.columns:
                if col == "geometry":
                    continue
                vals = gdf[col].dropna().unique()[:8]
                print(f"    {col}: {list(vals)}")
        except Exception as ex:
            print(f"  ⚠  {ex}")