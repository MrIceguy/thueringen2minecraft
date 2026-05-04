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
    1:  ("minecraft:water",               4,  15, "Fluss / Bach"),
    2:  ("minecraft:water",               2,  15, "See / Teich"),
    10: ("minecraft:gray_concrete",        8,  20, "Autobahn / Bundesstraße"),
    11: ("minecraft:light_gray_concrete",  6,  19, "Landstraße"),
    12: ("minecraft:smooth_stone",         5,  18, "Kreisstraße"),
    13: ("minecraft:stone_slab",           4,  17, "Gemeindestraße"),
    14: ("minecraft:stone_bricks",         3,  16, "Wohnstraße"),
    15: ("minecraft:cobblestone",          2,  15, "Feldweg / Wirtschaftsweg"),
    16: ("minecraft:cobblestone",          2,  15, "Fußweg / Pfad"),
    17: ("minecraft:oak_planks",           3,  15, "Radweg"),
    20: ("minecraft:rail",                 2,  18, "Bahnstrecke"),
    30: ("minecraft:grass_block",          0,   1, "Grünland / Wiese"),
    31: ("minecraft:moss_block",           0,   1, "Wald"),
    32: ("minecraft:sand",                 0,   2, "Sandfläche"),
    33: ("minecraft:grass_block",          0,   2, "Ackerland"),
    34: ("minecraft:grass_block",          0,   2, "Landwirtschaft"),
    35: ("minecraft:stone_bricks",         0,  14, "Schotter / Verkehrsfläche"),
    36: ("minecraft:gravel",               0,  14, "Platz / Vorplatz"),
    37: ("minecraft:gray_concrete",        0,  14, "Gewerbe / Industrie"),
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
    "AX_Strassenverkehr":           35,   # Straßenverkehrsfläche → Schotter
    "AX_Platz":                     36,   # Platz → Kies
    "AX_Weg":                       15,
    "AX_FussWandRadweg":            16,
    "AX_Bahnstrecke":               20,
    "AX_Seilbahn":                  20,
    # numerische OAK-Codes (kommen je nach Exportformat vor)
    "42001": 35,  # AX_Strassenverkehr (Fläche) → Schotter
    "42003": 10,  # Bundesstraße (Linie)
    "42005": 11,  # Landesstraße
    "42006": 12,  # Kreisstraße
    "42007": 13,  # Gemeindestraße
    "42008": 14,  # Wirtschaftsweg
    "42009": 36,  # AX_Platz → Kies
    "42010": 15,  # Weg
    "42015": 16,  # Fußweg
    "42016": 17,  # Radweg
    "42014": 20,  # Gleis
    "44001": 20,  # Bahnstrecke
    "44004": 20,  # Seilbahn
}

# Breitenangaben in Metern (für Linien-Puffer)
VERKEHR_BREITE = {
    10: 6, 11: 5, 12: 4, 13: 3, 14: 2, 15: 2, 16: 2, 17: 2, 20: 2,
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
    "44006": 2,  # AX_StehendesGewaesser (Teich/See)
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
    # OBJART_TXT (Fallback)
    "AX_Wald":              31,   # Wald → moss_block
    "AX_Gehoelz":          30,   # Gehölz/Gebüsch → grass_block
    "AX_Heide":             30,   # Heide → grass_block
    "AX_Moor":              30,   # Moor → grass_block
    "AX_Sumpf":             30,   # Sumpf → grass_block
    "AX_Landwirtschaft":    34,   # Landwirtschaft → grass_block, keine Bäume
    "AX_Grünland":          30,   # Grünland → grass_block
    "AX_Ackerland":         33,   # Ackerland → dirt
    "AX_UnlandVegetationsloseFlaeche": 32,  # Ödland → sand
    # Echte OBJART-Codes aus veg-Shapefiles
    "43001": 34,  # AX_Landwirtschaft → grass_block, keine Bäume
    "43002": 31,  # AX_Wald → moss_block, Bäume
    "43003": 30,  # AX_Gehoelz/Gebüsch → grass_block
    "43004": 30,  # AX_Heide → grass_block
    "43005": 30,  # AX_Moor → grass_block
    "43006": 30,  # AX_Sumpf → grass_block
    "43007": 32,  # AX_UnlandVegetationsloseFlaeche → sand
    # alte ATKIS-Codes (Fallback)
    "41001": 33,  # Ackerland → dirt
    "41002": 30,  # Gartenland → grass_block
    "41003": 30,  # Obstplantage → grass_block
}

# Siedlung (sie) – nur Grünflächen, Friedhöfe etc. (Gebäude kommen aus LoD2)
SIEDLUNG_MAP = {
    "AX_SportFreizeitUndErholungsflaeche": 30,
    "AX_Friedhof":                         30,
    "AX_Grünanlage":                       30,
    "AX_FlaecheGemischterNutzung":         30,  # Mischnutzung → Wiese
    "AX_Wohnbauflaeche":                   30,  # Wohnbaufläche → Wiese
    "AX_IndustrieUndGewerbeflaeche":       37,  # Gewerbe → gray_concrete
    "AX_FlaecheBesondererFunktionalerPraegung": 30,  # Schule/Krankenhaus → Wiese
    "41008": 30,  # Sport-/Freizeitanlage
    "41009": 30,  # Campingplatz
    "41010": 30,  # Friedhof
    "41006": 30,  # Gemischte Fläche → Wiese
    "41007": 30,  # Besondere Prägung → Wiese
    "41001": 30,  # Wohnbau → Wiese
    "41002": 35,  # Gewerbe → Schotter
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


def _load_geodataframe(folder, bbox_utm, exclude_stems=None):
    """
    Lädt alle Vektor-Dateien aus einem Ordner als GeoDataFrame.
    Unterstützt: .shp, .gpkg, .geojson, .xml (NAS/GML), .gml

    Gibt ein einzelnes GeoDataFrame zurück (alle Dateien zusammengeführt).
    """
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    def _excluded(f):
        return exclude_stems and any(ex in f.stem.lower() for ex in exclude_stems)

    shp_files  = [f for f in _find_files(folder, [".shp"])          if not _excluded(f)]
    gpkg_files = [f for f in _find_files(folder, [".gpkg"])         if not _excluded(f)]
    gml_files  = [f for f in _find_files(folder, [".xml", ".gml"])  if not _excluded(f)]

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

def load_ortslage_polygons(bbox):
    """Gibt Shapely-Polygone der AX_Ortslage zurück (für Laternen-Filter)."""
    import geopandas as gpd
    from shapely.geometry import box as shapely_box
    sie_dir = Path("atkis") / "sie"
    if not sie_dir.exists():
        return []
    gdf = _load_geodataframe(sie_dir, bbox)
    if gdf is None:
        return []
    col = _detect_type_column(gdf)
    polys = []
    for _, row in gdf.iterrows():
        objart = str(row.get("OBJART", "")).strip()
        if objart == "52001":  # AX_Ortslage
            geom = row.geometry
            if geom and not geom.is_empty:
                polys.append(geom)
    return polys


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
        # ver02_l: Fußwege explizit laden (OBJART unbekannt → würde sonst übersprungen)
        _ver02 = ver_dir / "ver02_l.shp"
        if _ver02.exists():
            try:
                import geopandas as gpd
                from shapely.geometry import box as _sbx2
                _gdf2 = gpd.read_file(str(_ver02))
                if _gdf2.crs and _gdf2.crs.to_epsg() != 25832:
                    _gdf2 = _gdf2.to_crs("EPSG:25832")
                _bb2 = _sbx2(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
                _gdf2 = _gdf2[_gdf2.geometry.intersects(_bb2)]
                _fw_w = VERKEHR_BREITE.get(16, 2)
                _n2 = 0
                for _, _r2 in _gdf2.iterrows():
                    _g2 = _r2.geometry
                    if _g2 is None or _g2.is_empty:
                        continue
                    try:
                        _buf2 = _g2.buffer(_fw_w / 2) if _g2.geom_type in ("LineString", "MultiLineString") else _g2
                        if _buf2.is_valid and not _buf2.is_empty:
                            features.append((_buf2, 16, 5))
                            _n2 += 1
                    except Exception:
                        pass
                if _n2:
                    print(f"    {_n2} Fußweg-Features (ver02_l, cls 16 prio 5)")
            except Exception as _e2:
                print(f"    ⚠ ver02_l: {_e2}")

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
                    continue   # unbekannte OBJART überspringen
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
            _has_nam = 'NAM' in gdf.columns
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                # Namenlose Linien-Gewässer = Drainage/Verrohrung → überspringen
                if (_has_nam
                        and geom.geom_type in ("LineString", "MultiLineString")
                        and not row.get('NAM')):
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
                objart = str(row.get("OBJART", "")).strip()
                # AX_Landwirtschaft: VEG-Feld für AL/GL-Unterscheidung nutzen
                if objart == "43001":
                    veg = str(row.get("VEG", "")).strip()
                    if veg in ("1020", "1021", "1022"):  # Ackerland
                        cls = 33  # dirt, keine Bäume
                    else:  # 1010=Grünland, 1030=Garten, 1050=Streuobst, sonstige
                        cls = 34  # grass_block, keine Bäume
                else:
                    cls = _get_class(row.get(col) if col else None, VEGETATION_MAP)
                if cls is None:
                    continue
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