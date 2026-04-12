"""
alkis_loader.py
===============
Laedt ALKIS-Gebaeudedaten per WFS vom Geoportal Thueringen.
Liefert Gebaeude-Footprints mit GFK (Gebaeude-Funktionsklasse) fuer
semantische Differenzierung (Kirche, Schule, Rathaus, Krankenhaus...).

WFS-Endpunkt: https://www.geoproxy.geoportal-th.de/geoproxy/services/adv_alkis_wfs
Typname:      ave:Gebaeude
Attribute:    funktion (GFK-Code), gebaeudefunktion (Text), geometry

GFK-Codes (Auswahl):
  1000 = Wohngebaeude
  2000 = Gebaeude fuer Handel/Dienstleistungen
  3000 = Gebaeude fuer Gewerbe/Industrie
  3610 = Dom, Stiftskirche
  3612 = Kirche
  3613 = Kapelle
  3614 = Sakralbau
  5000 = Gebaeude fuer Bildung/Forschung
  5410 = Rathaus
  ...
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
import json

WFS_URL = "https://www.geoproxy.geoportal-th.de/geoproxy/services/adv_alkis_v2_wfs"
WFS_TYPNAME = "adv:AX_Gebaeude"

# GFK-Codes fuer religiöse / sakrale Gebaeude
GFK_KIRCHE = {
    "3610", "3611", "3612", "3613", "3614",  # Dom, Kirche, Kapelle, Sakralbau
    "3620", "3621",                            # Kloster, Stift
    "3630",                                    # Moschee
    "3640",                                    # Synagoge
}

# GFK-Codes fuer besondere oeffentliche Gebaeude
GFK_BESONDERS = {
    "5410": "Rathaus",
    "5411": "Amtsgericht",
    "5400": "Oeffentliche Einrichtung",
    "5000": "Bildung/Forschung",
    "5110": "Schule",
    "5120": "Hochschule",
    "5210": "Krankenhaus",
    "3610": "Dom/Stiftskirche",
    "3612": "Kirche",
    "3613": "Kapelle",
    "3614": "Sakralbau",
}

NS_AVE = "http://www.adv-online.de/namespaces/adv/gid/6.0"
NS_GML = "http://www.opengis.net/gml/3.2"


def _parse_polygon(geom_el):
    """Extrahiert Koordinaten aus einem GML-Polygon-Element."""
    from shapely.geometry import Polygon
    for pos_el in geom_el.iter(f"{{{NS_GML}}}posList"):
        if pos_el.text:
            coords = list(map(float, pos_el.text.split()))
            pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            if len(pts) >= 3:
                try:
                    return Polygon(pts)
                except Exception:
                    pass
    return None


def load_alkis_buildings(bbox, cache_dir="alkis_gebaeude", timeout=30):
    """
    Laedt ALKIS-Gebaeude per WFS fuer die BBox.
    
    Args:
        bbox: dict mit west, east, south, north (UTM32N)
        cache_dir: Ordner fuer Cache-Dateien (vermeidet wiederholte WFS-Abfragen)
        timeout: HTTP-Timeout in Sekunden
    
    Returns:
        Liste von dicts: {footprint, gfk, gfk_text, name}
    """
    from shapely.geometry import Polygon, MultiPolygon

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"alkis_{int(bbox['west'])}_{int(bbox['south'])}.json"

    # Cache pruefen
    if cache_file.exists():
        print(f"  [Cache] ALKIS-Gebaeude: {cache_file.name}")
        with open(cache_file) as f:
            raw = json.load(f)
        buildings = []
        for b in raw:
            pts = b.get("pts", [])
            if len(pts) >= 3:
                try:
                    fp = Polygon(pts)
                    if fp.is_valid and not fp.is_empty:
                        b["footprint"] = fp
                        buildings.append(b)
                except Exception:
                    pass
        return buildings

    # WFS-Abfrage
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']},urn:ogc:def:crs:EPSG::25832"
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": WFS_TYPNAME,
        "BBOX": bbox_str,
        "COUNT": "10000",
    }
    url = WFS_URL + "?" + urllib.parse.urlencode(params)
    print(f"  WFS-Abfrage: {len(params)} Parameter, BBox {int(bbox['west'])}-{int(bbox['east'])}")

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            content = resp.read()
    except Exception as e:
        print(f"  ⚠  WFS-Abfrage fehlgeschlagen: {e}")
        return []

    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"  ⚠  XML-Parse-Fehler: {e}")
        return []

    buildings = []
    raw_cache = []

    for member in root.iter():
        tag = member.tag.split("}")[-1] if "}" in member.tag else member.tag
        if tag != "Gebaeude":
            continue

        # GFK (Gebaeude-Funktionsklasse)
        gfk = ""
        gfk_text = ""
        for child in member.iter():
            ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if ctag in ("funktion", "gebaeudefunktion") and child.text:
                val = child.text.strip()
                if val.isdigit():
                    gfk = val
                else:
                    gfk_text = val

        # Name
        name = ""
        for child in member.iter():
            ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if ctag in ("name", "geographischerName") and child.text:
                name = child.text.strip()
                break

        # Geometrie
        fp = None
        for geom_el in member.iter():
            gtag = geom_el.tag.split("}")[-1] if "}" in geom_el.tag else geom_el.tag
            if gtag in ("Polygon", "MultiPolygon", "Surface"):
                fp = _parse_polygon(geom_el)
                if fp:
                    break

        if fp is None or fp.is_empty:
            continue

        b = {
            "footprint": fp,
            "gfk": gfk,
            "gfk_text": gfk_text,
            "name": name,
        }
        buildings.append(b)

        # Cache-Daten (ohne Shapely-Objekt)
        pts = list(fp.exterior.coords) if hasattr(fp, "exterior") else []
        raw_cache.append({"pts": pts, "gfk": gfk, "gfk_text": gfk_text, "name": name})

    print(f"  ALKIS: {len(buildings)} Gebaeude geladen")
    gfk_stats = {}
    for b in buildings:
        g = b["gfk"]
        gfk_stats[g] = gfk_stats.get(g, 0) + 1
    for gfk, count in sorted(gfk_stats.items()):
        label = GFK_BESONDERS.get(gfk, "")
        if gfk in GFK_KIRCHE or label:
            print(f"    GFK {gfk} ({label}): {count} Gebaeude")

    # Cache speichern
    with open(cache_file, "w") as f:
        json.dump(raw_cache, f)

    return buildings


def get_church_footprints(buildings):
    """Gibt Shapely-Footprints aller Kirchengebaeude zurueck."""
    return [b["footprint"] for b in buildings if b.get("gfk") in GFK_KIRCHE]


def get_special_footprints(buildings):
    """Gibt {typ: [footprints]} fuer besondere oeffentliche Gebaeude zurueck."""
    result = {}
    for b in buildings:
        g = b.get("gfk", "")
        label = GFK_BESONDERS.get(g)
        if label:
            result.setdefault(label, []).append(b["footprint"])
    return result


if __name__ == "__main__":
    # Test
    bbox = {"west": 641608, "east": 642172, "south": 5649006, "north": 5649363}
    buildings = load_alkis_buildings(bbox)
    churches = get_church_footprints(buildings)
    print(f"\nKirchengebaeude: {len(churches)}")
    for fp in churches:
        c = fp.centroid
        print(f"  @ E{c.x:.0f} N{c.y:.0f}  Flaeche={fp.area:.0f}m²")
