"""
invekos_loader.py
=================
Lädt Thüringer InVeKoS-Feldblöcke per WFS-BBox-Query.

Feldblock-Codes (FBI-Präfix):
  AL = Ackerland     → dirt (keine Bäume)
  GL = Grünland      → grass_block
  HK = Dauerkultur   → grass_block
  FH = Forstfläche   → moss_block (Wald)
  LE = Landschaftselement → grass_block
  NW = Naturschutz   → moss_block

WFS-Endpoint: https://www.geoproxy.geoportal-th.de/geoproxy/services/INVEKOS_wfs
FeatureType: ave:Feldblock (oder ähnlich — GetCapabilities prüfen)
"""

import requests
import xml.etree.ElementTree as ET
from pathlib import Path

WFS_BASE = "https://www.geoproxy.geoportal-th.de/geoproxy/services/INVEKOS_wfs"

# Feldblock-Code → ATKIS-Klasse
FBI_MAP = {
    "AL": 33,   # Ackerland → dirt, keine Bäume
    "GL": 30,   # Grünland → grass_block
    "HK": 30,   # Dauerkultur → grass_block
    "FH": 31,   # Forstfläche → moss_block
    "FO": 31,   # Forstfläche → moss_block
    "LE": 30,   # Landschaftselement → grass_block
    "NW": 31,   # Naturschutz/Biotop → moss_block
    "WA": 31,   # Wald aus Erstaufforstung → moss_block
    "SF": 30,   # Sondernutzung → grass_block
    "LF": 30,   # Landw. Nutzfläche → grass_block
    "EF": 30,   # Erstaufforstung → grass_block
}


def load_invekos_wfs(bbox_utm, cache_dir="invekos"):
    """
    Lädt InVeKoS-Feldblöcke per WFS für eine BBox.
    bbox_utm: dict mit west, east, south, north in EPSG:25832

    Gibt Liste von (shapely_geom, klassen_wert, priorität) zurück.
    Prio 6 → überschreibt ATKIS-Vegetation (Prio 1-2).
    """
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / (
        f"invekos_{int(bbox_utm['west'])}_{int(bbox_utm['south'])}.gpkg"
    )

    if cache_file.exists():
        print(f"  [Cache] InVeKoS: {cache_file.name}")
        gdf = gpd.read_file(cache_file)
    else:
        gdf = _fetch_wfs(bbox_utm)
        if gdf is not None and len(gdf) > 0:
            gdf.to_file(cache_file, driver="GPKG")
        else:
            print("  ⚠  InVeKoS WFS: Keine Daten für diese BBox")
            return []

    if gdf is None or len(gdf) == 0:
        return []

    # BBox-Clip
    bbox_poly = shapely_box(
        bbox_utm["west"], bbox_utm["south"],
        bbox_utm["east"], bbox_utm["north"]
    )
    gdf = gdf[gdf.geometry.intersects(bbox_poly)].copy()

    features = []
    n_mapped = 0
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        fbi = str(row.get("FBI", "") or row.get("fbi", "") or "")[:2].upper()
        cls = FBI_MAP.get(fbi, None)
        if cls is None:
            # Fallback: FBI-Code aus anderen Spalten lesen
            for col in ["NUTZUNG", "nutzung", "FLAECHENNUTZUNG", "LF_CODE"]:
                val = str(row.get(col, ""))[:2].upper()
                cls = FBI_MAP.get(val)
                if cls:
                    fbi = val
                    break
        if cls is None:
            continue
        features.append((geom, cls, 6))  # Prio 6 > ATKIS
        n_mapped += 1

    print(f"  InVeKoS: {n_mapped}/{len(gdf)} Feldblöcke gemappt")
    return features


def _fetch_wfs(bbox_utm):
    """WFS GetFeature mit BBox-Filter."""
    import geopandas as gpd
    import io

    # Erst GetCapabilities um FeatureType-Namen zu finden
    caps_url = (
        f"{WFS_BASE}?SERVICE=WFS&REQUEST=GetCapabilities&VERSION=1.1.0"
    )
    try:
        resp = requests.get(caps_url, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        # FeatureType-Namen suchen
        ns = {"wfs": "http://www.opengis.net/wfs"}
        ft_names = [
            ft.find("wfs:Name", ns).text
            for ft in root.findall(".//wfs:FeatureType", ns)
            if ft.find("wfs:Name", ns) is not None
        ]
        print(f"  InVeKoS WFS FeatureTypes: {ft_names[:5]}")
        # Feldblock-FeatureType finden
        ft = next(
            (n for n in ft_names if "eldblock" in n or "LDBLOCK" in n or "lpis" in n.lower()),
            ft_names[0] if ft_names else "ave:Feldblock"
        )
    except Exception as ex:
        print(f"  ⚠  GetCapabilities: {ex}")
        ft = "ave:Feldblock"

    w, s, e, n = (bbox_utm["west"], bbox_utm["south"],
                  bbox_utm["east"], bbox_utm["north"])
    url = (
        f"{WFS_BASE}?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature"
        f"&TYPENAME={ft}"
        f"&BBOX={w},{s},{e},{n},urn:ogc:def:crs:EPSG::25832"
        f"&SRSNAME=urn:ogc:def:crs:EPSG::25832"
        f"&OUTPUTFORMAT=application/json"
    )
    print(f"  ↓ InVeKoS WFS ({ft})...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        import geopandas as gpd
        import io
        gdf = gpd.read_file(io.BytesIO(resp.content))
        if gdf.crs and gdf.crs.to_epsg() != 25832:
            gdf = gdf.to_crs("EPSG:25832")
        print(f"OK ({len(gdf)} Features)")
        return gdf
    except Exception as ex:
        print(f"FEHLER: {ex}")
        # Fallback: GML
        url_gml = url.replace("application/json", "text/xml; subtype=gml/3.1.1")
        try:
            resp = requests.get(url_gml, timeout=60)
            gdf = gpd.read_file(io.BytesIO(resp.content))
            if gdf.crs and gdf.crs.to_epsg() != 25832:
                gdf = gdf.to_crs("EPSG:25832")
            print(f"OK GML ({len(gdf)} Features)")
            return gdf
        except Exception as ex2:
            print(f"FEHLER GML: {ex2}")
            return None


def get_wfs_capabilities():
    """Zeigt alle FeatureTypes des InVeKoS-WFS (Diagnose)."""
    url = f"{WFS_BASE}?SERVICE=WFS&REQUEST=GetCapabilities&VERSION=1.1.0"
    try:
        resp = requests.get(url, timeout=30)
        root = ET.fromstring(resp.content)
        ns = {"wfs": "http://www.opengis.net/wfs"}
        for ft in root.findall(".//wfs:FeatureType", ns):
            name = ft.find("wfs:Name", ns)
            title = ft.find("wfs:Title", ns)
            print(f"  {name.text if name is not None else '?'}"
                  f" → {title.text if title is not None else '?'}")
    except Exception as ex:
        print(f"FEHLER: {ex}")


if __name__ == "__main__":
    print("=== InVeKoS WFS Capabilities ===")
    get_wfs_capabilities()