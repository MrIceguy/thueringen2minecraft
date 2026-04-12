"""
lod2_loader.py – Thüringen LoD2 CityGML → Minecraft-Gebäude
=============================================================
Liest alle .gml-Dateien aus dem LoD2-Ordner, extrahiert:
  - Gebäudegrundrisse (Polygone in UTM32N)
  - Traufhöhe / Firsthöhe / mittlere Dachhöhe

Aus diesen Daten werden zwei Raster erzeugt:
  building_mask  – uint8,  1 = Gebäudefläche, 0 = kein Gebäude
  building_height – float32, Gebäudehöhe in Metern über Gelände

Abhängigkeiten:
    pip install lxml shapely numpy rasterio
"""

import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.transform import from_bounds as transform_from_bounds
import re

try:
    from lxml import etree as ET
    LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    LXML = False

# CityGML / GML Namespaces (Thüringen LoD2 verwendet diese)
NS = {
    "core":  "http://www.opengis.net/citygml/2.0",
    "bldg":  "http://www.opengis.net/citygml/building/2.0",
    "gml":   "http://www.opengis.net/gml",
    "gen":   "http://www.opengis.net/citygml/generics/2.0",
}

# Koordinaten-Reihenfolge im Thüringen LoD2: X Y Z (= Easting Northing Höhe)
COORD_ORDER = "XYZ"


# ─────────────────────────────────────────────
# GML parsen
# ─────────────────────────────────────────────

def _parse_pos_list(text):
    """Parst einen GML posList-String → Liste von (x, y, z)-Tupeln."""
    nums = [float(v) for v in text.split()]
    # Dreiergruppen: X Y Z
    return [(nums[i], nums[i+1], nums[i+2]) for i in range(0, len(nums)-2, 3)]


def _polygon_from_surface(surface_el):
    """
    Extrahiert das äußere Polygon einer gml:Polygon / gml:Surface.
    Gibt Shapely-Polygon in 2D (XY, UTM32N) zurück oder None.
    """
    # Exterior ring suchen
    for tag in ("gml:exterior", "gml:outerBoundaryIs"):
        ext = surface_el.find(f".//{tag}/gml:LinearRing/gml:posList", NS)
        if ext is not None and ext.text:
            pts = _parse_pos_list(ext.text)
            if len(pts) >= 3:
                coords_2d = [(p[0], p[1]) for p in pts]
                try:
                    poly = Polygon(coords_2d)
                    if poly.is_valid and not poly.is_empty:
                        return poly, [p[2] for p in pts]
                except Exception:
                    pass
    return None, None


def _mean_z(surface_el):
    """Mittlere Z-Höhe einer Fläche."""
    for tag in ("gml:exterior", "gml:outerBoundaryIs"):
        el = surface_el.find(f".//{tag}/gml:LinearRing/gml:posList", NS)
        if el is not None and el.text:
            pts = _parse_pos_list(el.text)
            if pts:
                return np.mean([p[2] for p in pts])
    return None


def _get_measured_height(bldg_el):
    """
    Liest explizite Höhenattribute aus dem Building-Element:
      bldg:measuredHeight, bldg:storeysAboveGround
    """
    mh = bldg_el.find("bldg:measuredHeight", NS)
    if mh is not None and mh.text:
        try:
            return float(mh.text)
        except ValueError:
            pass
    return None


# ─────────────────────────────────────────────
# Gebäude aus einer GML-Datei extrahieren
# ─────────────────────────────────────────────

def parse_gml_file(filepath):
    """
    Liest CityGML. Footprint aus GroundSurface, Dach aus roof_z absolut.
    Fallback: tiefstes Polygon als Grundriss wenn kein GroundSurface vorhanden.
    """
    buildings = []
    tree = ET.parse(str(filepath))
    root = tree.getroot()

    ns_map = {}
    for _, elem in ET.iterparse(str(filepath), events=["start-ns"]):
        ns_map[elem[0]] = elem[1]

    bldg_ns = next((v for k, v in ns_map.items() if "citygml/building" in v.lower()),
                   "http://www.opengis.net/citygml/building/1.0")
    gml_ns  = next((v for k, v in ns_map.items() if "opengis.net/gml" in v),
                   "http://www.opengis.net/gml")

    def findall(el, tag, ns):
        return el.findall(f".//{{{ns}}}{tag}")

    # Strategie: Wenn ein Building nested BuildingParts hat, rendere die Parts direkt.
    # Wenn ein Building keine nested BuildingParts hat, rendere es direkt.
    # Top-level BuildingParts (direkte cityObjectMember) werden immer direkt gerendert.
    # So werden Kaufland/Eishalle vollständig erfasst ohne Phantom-Duplikate.
    bldg_elements = []
    for child in root:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        el = None
        if tag in ("Building", "BuildingPart"):
            el = child
        else:
            for gc in child:
                gc_tag = gc.tag.split("}")[-1] if "}" in gc.tag else gc.tag
                if gc_tag in ("Building", "BuildingPart"):
                    el = gc
                    break
        if el is None:
            continue

        el_tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if el_tag == "Building":
            # Prüfe ob das Building nested BuildingParts enthält
            nested_parts = el.findall(f".//{{{bldg_ns}}}BuildingPart")
            if nested_parts:
                # Building ist nur Container → nested Parts direkt rendern
                bldg_elements.extend(nested_parts)
            else:
                # Building ohne Parts → direkt rendern
                bldg_elements.append(el)
        else:
            # Top-level BuildingPart → direkt rendern
            bldg_elements.append(el)

    n_buildings      = sum(1 for e in bldg_elements if e.tag.split("}")[-1] == "Building")
    n_building_parts = sum(1 for e in bldg_elements if e.tag.split("}")[-1] == "BuildingPart")
    if n_building_parts > 0:
        print(f"      {Path(filepath).name}: {n_buildings} Building + {n_building_parts} BuildingPart")

    for bldg in bldg_elements:
        gml_id = bldg.get(f"{{{gml_ns}}}id", "?")

        # roofType auslesen
        roof_type = "1000"   # Fallback: Flachdach
        rt_el = bldg.find(f"{{{bldg_ns}}}roofType")
        if rt_el is not None and rt_el.text:
            roof_type = rt_el.text.strip()
        # Auch als stringAttribute suchen (manche GML-Varianten)
        if roof_type == "1000":
            for sa in findall(bldg, "stringAttribute", 
                             next((v for k,v in ns_map.items() if "generics" in v),
                                  "http://www.opengis.net/citygml/generics/1.0")):
                name_el = sa.find(".//{*}name")
                val_el  = sa.find(".//{*}value")
                if name_el is not None and val_el is not None:
                    if "roof" in (name_el.text or "").lower():
                        roof_type = val_el.text.strip()
                        break

        # measuredHeight
        meas_h = None
        mh_el  = bldg.find(f"{{{bldg_ns}}}measuredHeight")
        if mh_el is not None and mh_el.text:
            try:
                meas_h = float(mh_el.text)
            except ValueError:
                pass

        # Jedes Element (Building oder BuildingPart) wird für sich allein geparst.
        # Da Buildings mit nested Parts bereits durch deren Parts ersetzt wurden,
        # enthalten die bldg_elements jetzt nur noch Elemente mit eigener Geometrie.
        # Normales findall ist hier korrekt.

        # Alle Polygone sammeln mit ihren mittleren Z-Werten
        all_polys = []   # (mean_z, polygon)
        roof_zs   = []

        for pl in findall(bldg, "posList", gml_ns):
            if not pl.text:
                continue
            pts = _parse_pos_list(pl.text)
            if len(pts) < 3:
                continue
            mean_z = float(np.mean([p[2] for p in pts]))
            try:
                poly = Polygon([(p[0], p[1]) for p in pts])
                if poly.is_valid and not poly.is_empty and poly.area >= 4:
                    all_polys.append((mean_z, poly))
            except Exception:
                pass

        if not all_polys:
            continue

        # RoofSurface: höchste Z-Werte
        for surf in findall(bldg, "RoofSurface", bldg_ns):
            for pl in findall(surf, "posList", gml_ns):
                if pl.text:
                    pts = _parse_pos_list(pl.text)
                    roof_zs.extend(p[2] for p in pts)

        # GroundSurface: direkt nehmen
        footprint_poly = None
        ground_z = None
        for surf in findall(bldg, "GroundSurface", bldg_ns):
            for pl in findall(surf, "posList", gml_ns):
                if not pl.text: continue
                pts = _parse_pos_list(pl.text)
                if len(pts) < 3: continue
                try:
                    poly = Polygon([(p[0], p[1]) for p in pts])
                    if poly.is_valid and not poly.is_empty and poly.area >= 4:
                        footprint_poly = poly
                        ground_z = float(np.mean([p[2] for p in pts]))
                        break
                except Exception:
                    pass
            if footprint_poly:
                break

        # Fallback: Polygon mit niedrigstem mittlerem Z
        if footprint_poly is None and all_polys:
            all_polys.sort(key=lambda x: x[0])
            ground_z, footprint_poly = all_polys[0]

        if footprint_poly is None:
            continue

        # Dachhöhe absolut (Meter ü.NN)
        roof_z = float(np.mean(roof_zs)) if roof_zs else None

        # Falls kein RoofSurface: höchstes Polygon nehmen
        if roof_z is None and all_polys:
            all_polys.sort(key=lambda x: x[0], reverse=True)
            roof_z = all_polys[0][0]

        # Gebäudehöhe
        if meas_h is not None:
            height_m = max(2.5, meas_h)
        elif ground_z is not None and roof_z is not None:
            height_m = max(2.5, roof_z - ground_z)
        else:
            height_m = 6.0

        buildings.append({
            "footprint": footprint_poly,
            "height_m":  float(height_m),
            "ground_z":  float(ground_z) if ground_z else 0.0,
            "roof_z":    float(roof_z)   if roof_z   else 0.0,
            "roof_type": roof_type,
            "has_ground": ground_z is not None,
            "gml_id":    gml_id,
        })

    return buildings

# ─────────────────────────────────────────────
# Alle GML-Dateien eines Ordners laden
# ─────────────────────────────────────────────

def load_all_lod2(folder, bbox=None):
    """
    Lädt alle .gml-Dateien aus `folder`.

    Returns:
        Liste aller Gebäude (dicts, siehe parse_gml_file)
    """
    folder    = Path(folder)
    gml_files = sorted(set(folder.glob("*.gml")) | set(folder.glob("*.GML")))

    if not gml_files:
        raise FileNotFoundError(f"Keine .gml-Dateien in '{folder}/'")

    print(f"  {len(gml_files)} GML-Datei(en):")
    for f in gml_files:
        print(f"    {f.name}")

    all_buildings = []
    for f in gml_files:
        try:
            bldgs = parse_gml_file(f)
            n_with_ground = sum(1 for b in bldgs if b.get("has_ground"))
            print(f"    {f.name}: {len(bldgs)} Gebäude")
            all_buildings.extend(bldgs)
        except Exception as ex:
            print(f"  ⚠  {f.name}: {ex}")

    # BBox-Filter
    if bbox:
        from shapely.geometry import box as shapely_box
        bbox_poly = shapely_box(bbox["west"], bbox["south"],
                                bbox["east"], bbox["north"])
        before = len(all_buildings)
        all_buildings = [b for b in all_buildings
                         if b["footprint"].intersects(bbox_poly)]
        print(f"  BBox-Filter: {before} → {len(all_buildings)} Gebäude")

    # Statistik
    if all_buildings:
        heights = [b["height_m"] for b in all_buildings]
        n_with_ground = sum(1 for b in all_buildings if b.get("has_ground"))
        print(f"  Gebäude gesamt: {len(all_buildings):,}  "
              f"(davon mit GroundSurface: {n_with_ground}, "
              f"Fallback: {len(all_buildings)-n_with_ground})")
        print(f"  Höhe: {min(heights):.1f} – {max(heights):.1f} m  "
              f"(∅ {np.mean(heights):.1f} m)")
        under5 = sum(1 for h in heights if h < 5)
        mid    = sum(1 for h in heights if 5 <= h < 15)
        tall   = sum(1 for h in heights if h >= 15)
        print(f"  < 5m: {under5}  |  5–15m: {mid}  |  >15m: {tall}")

    return all_buildings


# ─────────────────────────────────────────────
# Gebäude auf Raster brennen
# ─────────────────────────────────────────────

def rasterize_buildings(buildings, bbox, shape, real_min_h=450.0):
    """
    real_min_h: minimale Gelaendehoehe der BBox (aus DGM).
                Wird fuer roof_mc Berechnung verwendet.
                Muss korrekt uebergeben werden damit Dachoehen stimmen!
    """
    H, W      = shape
    transform = transform_from_bounds(
        bbox["west"], bbox["south"], bbox["east"], bbox["north"], W, H
    )
    px_m      = (bbox["east"] - bbox["west"]) / W
    mask      = np.zeros(shape, dtype=np.uint8)
    heights   = np.zeros(shape, dtype=np.float32)
    roof_mc   = np.zeros(shape, dtype=np.int32)
    roof_type = np.zeros(shape, dtype=np.uint16)

    ROOF_CODES = {"1000":1000,"2100":2100,"2200":2200,
                  "3100":3100,"3200":3200,"4000":4000,"5000":5000}

    n_ok = 0
    for bldg in sorted(buildings, key=lambda b: b["height_m"]):
        poly   = bldg["footprint"]
        h_m    = bldg["height_m"]
        r_z    = bldg.get("roof_z", 0.0)
        r_type = ROOF_CODES.get(str(bldg.get("roof_type","1000")), 1000)

        if poly is None or poly.is_empty or h_m < 1.5:
            continue
        if poly.area < 9:
            continue

        try:
            burned = rasterize(
                [(poly, 1)], out_shape=shape,
                transform=transform, fill=0,
                dtype=np.uint8, all_touched=True,
            )
            where = burned == 1
            if not where.any(): continue
            mask[where]      = 1
            heights[where]   = h_m
            roof_type[where] = r_type
            if r_z > 0:
                # Korrekter Offset: r_z (absolut ü.NN) → MC-Y relativ zu real_min_h
                roof_mc[where] = max(int(round(r_z - real_min_h + 10)), 3)
            n_ok += 1
        except Exception:
            pass

    print(f"  {n_ok:,} Gebäude rasterisiert | {int(mask.sum()):,} px")
    for code, name in [(1000,"Flach"),(2100,"Sattel"),(3100,"Zelt"),(3200,"Walm"),(4000,"Pult")]:
        n = int((roof_type == code).sum())
        if n: print(f"    {name}: {n:,} px")
    return mask, heights, roof_mc, roof_type


# ─────────────────────────────────────────────
# Minecraft-Gebäude setzen (aus echten LoD2-Daten)
# ─────────────────────────────────────────────

def place_lod2_building(region, bx, bz, surface_y, height_m):
    """
    Setzt ein Gebäude mit echter Höhe aus LoD2-Daten.
    Erdgeschoss → Obergeschosse → Flachdach.

    Materialwahl nach Höhe:
        < 5m  → Steinziegel   (Schuppen, Garagen)
        5–12m → Weißer Beton  (Wohnhäuser)
        > 12m → Quartz-Block  (Büro, Hochhäuser)
    """
    try:
        import anvil
    except ImportError:
        return

    h = max(2, min(int(round(height_m)), 50))

    if height_m < 5:
        wall_name = "stone_bricks"
        roof_name = "stone_brick_slab"
    elif height_m < 12:
        wall_name = "white_concrete"
        roof_name = "gray_concrete"
    else:
        wall_name = "quartz_block"
        roof_name = "smooth_quartz"

    wall = anvil.Block("minecraft", wall_name)
    roof = anvil.Block("minecraft", roof_name)

    for dy in range(h):
        b = roof if dy == h - 1 else wall
        try:
            region.set_block(b, bx % 512, surface_y + 1 + dy, bz % 512)
        except Exception:
            pass
