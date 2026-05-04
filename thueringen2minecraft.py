"""
thueringen2minecraft.py
=======================
Konvertiert Geodaten aus dem Geoportal Thueringen in eine Minecraft Java 1.21.5 Welt.

Ordnerstruktur:
    dgm/    *.tif               <- DGM1 Gelaendemodell (1m)
    dom/    *.tif               <- DOM1 Oberflaechenmodell (1m, optional)
    LoD2/   *.gml               <- 3D-Gebaeude (CityGML LoD2)
    atkis/  ver/ gew/ sie/ veg/ <- ATKIS Basis-DLM Shapefiles

Aufruf:
    python thueringen2minecraft.py --bbox 641000 642000 5648000 5649000
    python thueringen2minecraft.py --list

Installation:
    pip install numpy scipy nbtlib geopandas shapely rasterio pyproj lxml
"""

import numpy as np
import argparse
import os
from pathlib import Path
from rasterio.warp import reproject, Resampling as RS
from scipy.ndimage import maximum_filter, label as ndlabel

from entity_writer import EntityWriter, populate_city
from tile_loader import (
    load_all_tiles, make_rasterio_transform,
    list_available_tiles, get_bbox_from_folder,
)
from lod2_loader import (
    load_all_lod2, rasterize_buildings, place_lod2_building,
)
from atkis_layer import (
    load_osm_layers, burn_osm_to_raster,
    get_osm_block, is_water, print_osm_stats, OSM_CLASSES, CHURCH_POINTS,
    load_ortslage_polygons,
)

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

DGM_DIR    = "dgm"
DOM_DIR    = "dom"
LOD2_DIR   = "LoD2"

BBOX       = {      # exakt die 9 vorhandenen Kacheln (3x3 km)
    "west":  634_000,
    "east":  637_000,
    "south": 5_615_000,
    "north": 5_618_000,
}

OUTPUT_DIR = "thuringen2minecraft_output"

MC_MIN_Y   = 0
MC_MAX_Y   = 320
REAL_MIN_H = None   # wird automatisch aus DGM berechnet (min Hoehe der BBox)
REAL_MAX_H = 900.0

THRESH_MIN_OBJECT   = 1.5
THRESH_BUILDING_MAX = 50.0
THRESH_VEG_ROUGH    = 1.2   # hoeher als vorher: DOM1 ist rauer als DOM2
TILE_WORKERS        = 4

# ─────────────────────────────────────────────
# TERRAIN
# ─────────────────────────────────────────────

def compute_ndsm(dgm, dgm_ox, dgm_oy, dgm_res,
                 dom, dom_ox, dom_oy, dom_res):
    dgm_t = make_rasterio_transform(dgm_ox, dgm_oy, dgm_res, dgm.shape[0])
    dom_t = make_rasterio_transform(dom_ox, dom_oy, dom_res, dom.shape[0])
    dom_r = np.empty_like(dgm)
    reproject(source=dom, destination=dom_r,
              src_transform=dom_t, dst_transform=dgm_t,
              src_crs="EPSG:25832", dst_crs="EPSG:25832",
              resampling=RS.bilinear)
    return np.clip(dom_r - dgm, 0, None)


def classify_ndsm(ndsm):
    """Klassifiziert nDSM → 0=Boden, 1=Gebaeude(nDSM), 2=Vegetation."""
    lmax = maximum_filter(ndsm,  size=5, mode="reflect")
    lmin = maximum_filter(-ndsm, size=5, mode="reflect")
    rough    = lmax + lmin   # = local_max - local_min = lokale Spannweite
    elevated = ndsm > THRESH_MIN_OBJECT
    is_veg   = elevated & (rough >  THRESH_VEG_ROUGH) & (ndsm < THRESH_BUILDING_MAX)
    is_braw  = elevated & (rough <= THRESH_VEG_ROUGH) & (ndsm < THRESH_BUILDING_MAX)
    labeled, _ = ndlabel(is_braw)
    sizes = np.bincount(labeled.ravel())
    small = sizes < 20;  small[0] = False
    is_bldg = is_braw & ~small[labeled]
    cls = np.zeros_like(ndsm, dtype=np.uint8)
    cls[is_veg]  = 2
    cls[is_bldg] = 1
    print(f"  nDSM-Gebaeude: {int((cls==1).sum()):,} px  |  "
          f"Vegetation: {int((cls==2).sum()):,} px")
    return cls


def height_to_mc_y(real_height):
    """1 Meter = 1 Minecraft-Block. Terrain wird auf Y=10 (Minimum) verschoben."""
    return (np.asarray(real_height) - REAL_MIN_H + 10).astype(np.int32)


# ─────────────────────────────────────────────
# BLOCK-LOGIK
# ─────────────────────────────────────────────

def get_terrain_column(surface_y, real_h, osm_val):
    blocks = []
    for mc_y in range(surface_y):
        depth = surface_y - mc_y
        if   mc_y == 0:  b = ("minecraft", "bedrock")
        elif mc_y < 30:  b = ("minecraft", "deepslate")
        elif depth > 10: b = ("minecraft", "stone")
        elif depth > 3:  b = ("minecraft", "stone")
        elif depth > 1:  b = ("minecraft", "dirt")
        else:            b = ("minecraft", "dirt")
        blocks.append((mc_y, *b))

    if osm_val > 0 and is_water(osm_val):
        deep = OSM_CLASSES.get(int(osm_val), (None, 0))[1] >= 8
        if deep:
            blocks = [(y, ns, n) for y, ns, n in blocks if y < surface_y - 1]
            blocks += [(surface_y-1, "minecraft", "gravel"),
                       (surface_y,   "minecraft", "water"),
                       (surface_y+1, "minecraft", "water")]
        else:
            blocks += [(surface_y,   "minecraft", "gravel"),
                       (surface_y+1, "minecraft", "water")]
    elif osm_val > 0:
        osm_block = get_osm_block(osm_val)
        if osm_block:
            ns, name = osm_block.split(":", 1)
            if name == "rail":
                blocks += [(surface_y,   "minecraft", "gravel"),
                           (surface_y+1, "minecraft", "rail")]
            else:
                blocks.append((surface_y, ns, name))
    else:
        if   real_h > 780: surf = ("minecraft", "snow_block")
        elif real_h > 700: surf = ("minecraft", "stone")
        else:              surf = ("minecraft", "grass_block")
        blocks.append((surface_y, *surf))

    return blocks


def _place_tree(writer, bx, bz, sy, obj_h, tree_type=None, building_mask=None):
    h_var   = (bx * 3 + bz * 7) % 7 - 3          # -3 … +3 Blöcke Höhenvariation
    h       = max(3, min(int(obj_h) + h_var, 22))
    trunk_h = max(1, h - 4)

    if tree_type is None:
        seed = (bx * 31 + bz * 17) % 4
        tree_type = ["oak", "spruce", "birch", "dark_oak"][seed]

    log    = f"minecraft:{tree_type}_log"
    leaves = f"minecraft:{tree_type}_leaves[persistent=true]"

    cy = sy + trunk_h + 1

    def safe_set(x, y, z, block):
        """Setzt Block nur wenn kein LoD2-Gebaeude an dieser (x,z)-Position."""
        if building_mask is not None:
            H, W = building_mask.shape
            if 0 <= z < H and 0 <= x < W and building_mask[z, x]:
                return
        writer.set_block(x, y, z, block)

    # Busch: kompakter Blätterhaufen, kein Stamm
    if tree_type == "bush":
        oak_l = "minecraft:oak_leaves[persistent=true]"
        for dy in range(3):
            r = 1 if dy < 2 else 0
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    safe_set(bx + dx, sy + 1 + dy, bz + dz, oak_l)
        return

    # Blaetter zuerst
    if tree_type == "spruce":
        for dy in range(-1, h - trunk_h + 1):
            r = max(0, 2 - max(0, dy))
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) + abs(dz) <= r + 1:
                        safe_set(bx + dx, cy + dy, bz + dz, leaves)
    else:
        for dy in range(-1, 3):
            r = 2 if dy <= 0 else 1
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) <= r and abs(dz) <= r:
                        safe_set(bx + dx, cy + dy, bz + dz, leaves)

    # Stamm bis zum Boden (sy+1 bis sy+trunk_h)
    for dy in range(trunk_h + 1):
        safe_set(bx, sy + dy, bz, log)


def _place_church(writer, bx, bz, sy):
    """
    Platziert einen einfachen Kirchturm (5x5 Grundriss, 20 Bloecke hoch)
    mit Spitzdach aus Ziegeltreppenstufen.
    Wird nur gesetzt wenn die Position nicht schon ein LoD2-Gebaeude hat
    (der Turm dient als Marker fuer Kirchen die nicht in LoD2 sind).
    """
    W_HALF = 2      # Halbbreite: 5x5 Grundriss
    WALL_H = 14     # Wandhoehe
    ROOF_H = 6      # Spitzdach-Hoehe

    WALL = "minecraft:stone_bricks"
    WIN  = "minecraft:glass_pane"
    ROOF = "minecraft:brick_stairs"
    TIP  = "minecraft:brick_wall"

    # Wände
    for dy in range(WALL_H):
        y = sy + 1 + dy
        for dx in range(-W_HALF, W_HALF + 1):
            for dz in range(-W_HALF, W_HALF + 1):
                is_wall = (abs(dx) == W_HALF or abs(dz) == W_HALF)
                if not is_wall:
                    continue
                # Fenster in Wandmitte (dy 6+7, nur an den Seiten)
                is_win = (dy in (6, 7) and
                          ((abs(dx) == W_HALF and dz == 0) or
                           (abs(dz) == W_HALF and dx == 0)))
                writer.set_block(bx + dx, y, bz + dz, WIN if is_win else WALL)

    # Spitzdach: quadratische Pyramide aus Treppen
    for dr in range(W_HALF + 1):
        y = sy + 1 + WALL_H + dr
        half = W_HALF - dr
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                if abs(dx) == half or abs(dz) == half:
                    # Treppe-Richtung: nach außen zeigend
                    if abs(dx) >= abs(dz):
                        face = "east" if dx > 0 else "west"
                    else:
                        face = "south" if dz > 0 else "north"
                    writer.set_block(bx + dx, y, bz + dz,
                                     f"{ROOF}[facing={face},half=bottom,shape=straight]")
    # Spitze
    writer.set_block(bx, sy + 1 + WALL_H + W_HALF + 1, bz, TIP)
    # Kreuz auf der Spitze
    for dx, dz in [(-1,0),(1,0),(0,-1),(0,1)]:
        writer.set_block(bx + dx, sy + 1 + WALL_H + W_HALF + 1, bz + dz, TIP)


def _should_place_tree(row, col, osm_raster, ndsm, ndsm_classes):
    """
    Entscheidet ob an dieser Position ein Baum steht.
    Kombiniert ATKIS-Vegetation (Klasse 31=Wald, 30=Gruenland) mit nDSM.
    """
    osm_val = int(osm_raster[row, col])
    cls     = int(ndsm_classes[row, col])
    h       = float(ndsm[row, col])

    # ATKIS sagt Wald → Baum wenn nDSM > 1.5m
    # Aber nicht jeden Pixel → Ausduennung fuer natuerlicheres Aussehen
    if osm_val == 31:   # Wald
        if h > 2.0:
            # Baeume auf ~40% der Pixel setzen (deterministisch)
            seed = (row * 7 + col * 13) % 10
            return seed < 4, h, "spruce" if (row + col) % 3 == 0 else "oak"
        return False, h, None

    if osm_val == 30:   # Gruenland / Park
        if h > 3.0:
            seed = (row * 11 + col * 7) % 10
            return seed < 2, h, "birch"   # vereinzelte Baeume
        return False, h, None

    # kein ATKIS-Vegetation → nDSM-Fallback
    if cls == 2 and h > 2.0:
        seed = (row * 5 + col * 19) % 10
        return seed < 5, h, None

    return False, h, None


def _place_lod2(writer, bx, bz, sy, height_m):
    """1 Meter Gebaeudehoehe = 1 Minecraft-Block. Proportionen stimmen weil
    auch das Raster 1m/px ist."""
    h_blocks = max(2, min(int(round(height_m)), 50))

    if   height_m < 4:  wall, roof = "minecraft:stone_bricks",      "minecraft:stone_brick_slab"
    elif height_m < 10: wall, roof = "minecraft:white_concrete",     "minecraft:gray_concrete"
    elif height_m < 20: wall, roof = "minecraft:light_gray_concrete","minecraft:gray_concrete"
    else:               wall, roof = "minecraft:quartz_block",       "minecraft:smooth_quartz"

    for dy in range(h_blocks):
        writer.set_block(bx, sy + 1 + dy, bz, roof if dy == h_blocks - 1 else wall)


# ─────────────────────────────────────────────
# WELT SCHREIBEN
# ─────────────────────────────────────────────

def write_minecraft_world(dgm, ndsm_classes, ndsm,
                          osm_raster, lod2_mask, lod2_heights,
                          lod2_roof_mc, lod2_roof_type,
                          dgm_ox=0, dgm_oy=0, dgm_res=1.0,
                          buildings=None, alkis_buildings=None,
                          bridge_mask=None, ortslage_mask=None,
                          writer=None, _is_tile=False,
                          output_dir=None, precomp_church_mask=None,
                          church_points=None):
    from anvil_writer import WorldWriter, write_level_dat
    from scipy.ndimage import maximum_filter, binary_erosion, distance_transform_edt, label as ndlabel

    _out = output_dir or OUTPUT_DIR
    os.makedirs(_out, exist_ok=True)
    H, W = dgm.shape
    print(f"  {W} x {H} Bloecke")

    if writer is None:
        writer = WorldWriter(_out)
    prog_step = max(1, H // 40)

    print("  Berechne glatte Gebaeude-Basis...")
    mc_y_grid   = height_to_mc_y(dgm)
    from scipy.ndimage import maximum_filter as _mxf
    road_mask_r = (osm_raster >= 10) & (osm_raster <= 20)
    road_y_grid = mc_y_grid.copy().astype(np.int32)
    if bridge_mask is not None and bridge_mask.any():
        # Nur Brückenpixel: max Straßenhöhe aus Umgebung (überspringt das Tal)
        road_y_map = np.where(road_mask_r & ~bridge_mask, mc_y_grid, 0).astype(np.int32)
        road_y_max = _mxf(road_y_map, size=101)
        # Fallback auf mc_y_grid wenn keine Straße in Nähe (verhindert sy=-1 Loch)
        road_y_grid[bridge_mask] = np.where(
            road_y_max[bridge_mask] > 0,
            road_y_max[bridge_mask],
            mc_y_grid[bridge_mask].astype(np.int32)
        )
    smooth_base = _mxf(mc_y_grid, size=3, mode='nearest')  # fuer Dach

    # Boden pro Gebaeude: Median der Terrain-Y-Werte aller Pixel des Gebaeudes
    # → gleichmaessiger Boden, passt zum Terrain, kein Unterfliessen
    from scipy.ndimage import label as _lbl2
    labeled_bldg, _ = _lbl2(lod2_mask == 1)
    smooth_floor = mc_y_grid.copy().astype(np.int32)
    for comp_id in np.unique(labeled_bldg):
        if comp_id == 0: continue
        comp_mask = labeled_bldg == comp_id
        median_y  = int(np.median(mc_y_grid[comp_mask]))
        smooth_floor[comp_mask] = median_y

    # Einheitliches Material pro Gebäude-Komponente (max h_blocks der ganzen Komponente)
    def mat_for_h(h):
        if   h < 4:  return "minecraft:stone_bricks",       "minecraft:stone_brick_slab"
        elif h < 10: return "minecraft:white_concrete",      "minecraft:gray_concrete"
        elif h < 20: return "minecraft:light_gray_concrete", "minecraft:gray_concrete"
        else:        return "minecraft:quartz_block",        "minecraft:smooth_quartz"

    comp_wall_mat   = {}   # comp_id → (wall_b, roof_b)
    comp_flat_top_y = {}   # comp_id → einheitliche top_y für Flachdächer
    _STORY_H_PRE    = 4
    _PITCHED_TYPES  = {2100, 3100, 3200, 4000}
    for comp_id in np.unique(labeled_bldg):
        if comp_id == 0: continue
        comp_mask = labeled_bldg == comp_id
        h_max = float((lod2_roof_mc[comp_mask] - smooth_floor[comp_mask]).max())
        comp_wall_mat[comp_id] = mat_for_h(h_max)
        # Flachdach: einheitliche Höhe per Median → keine unebenen Dächer
        if not bool(np.isin(lod2_roof_type[comp_mask], list(_PITCHED_TYPES)).any()):
            med_roof = int(np.median(lod2_roof_mc[comp_mask]))
            med_floor = int(np.median(smooth_floor[comp_mask]))
            raw_h_med = med_roof - med_floor
            h_med = max(_STORY_H_PRE,
                        int((raw_h_med + _STORY_H_PRE // 2) // _STORY_H_PRE) * _STORY_H_PRE)
            h_med = min(h_med, 80)
            comp_flat_top_y[comp_id] = med_floor + h_med
    H_m, W_m = lod2_mask.shape
    dgm_north = dgm_oy + H_m * dgm_res
    dgm_west  = dgm_ox

    # Vorab berechnete church_mask nutzen (tiled mode) oder frisch berechnen
    if precomp_church_mask is not None:
        church_mask = precomp_church_mask.astype(bool)
    else:
        church_mask = np.zeros(dgm.shape, dtype=bool)
        church_comp_ids = set()
        _church_pts = church_points if church_points is not None else CHURCH_POINTS

        if _church_pts and buildings:
            from shapely.geometry import Point
            for cx, cy in _church_pts:
                pt = Point(cx, cy)
                matched_ids = set()
                for bldg in buildings:
                    fp = bldg.get("footprint")
                    if fp is None: continue
                    if fp.distance(pt) < 150:
                        try: tp = fp.representative_point()
                        except: tp = fp.centroid
                        ci_b = int(round((tp.x - dgm_west) / dgm_res))
                        ri_b = int(round((dgm_north - tp.y) / dgm_res))
                        if 0 <= ri_b < H_m and 0 <= ci_b < W_m:
                            cid = int(labeled_bldg[ri_b, ci_b])
                            if cid > 0: matched_ids.add(cid)
                church_comp_ids.update(matched_ids)

        for comp_id in np.unique(labeled_bldg):
            if comp_id == 0: continue
            comp = labeled_bldg == comp_id
            if float(lod2_heights[comp].max()) > 40:
                church_comp_ids.add(comp_id)

        if alkis_buildings:
            try:
                from alkis_loader import get_church_footprints
                for fp in get_church_footprints(alkis_buildings):
                    try: tp = fp.representative_point()
                    except: tp = fp.centroid
                    ci_b = int(round((tp.x - dgm_west) / dgm_res))
                    ri_b = int(round((dgm_north - tp.y) / dgm_res))
                    if 0 <= ri_b < H_m and 0 <= ci_b < W_m:
                        cid = int(labeled_bldg[ri_b, ci_b])
                        if cid > 0: church_comp_ids.add(cid)
            except Exception:
                pass

        for comp_id in church_comp_ids:
            church_mask[labeled_bldg == comp_id] = True
        print(f"  Kirchengebaeude: {int(church_mask.sum()):,} px")

    print("  Berechne Gebaeude-Innen/Aussen...")
    struct        = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    building_bool = lod2_mask == 1
    eroded        = binary_erosion(building_bool, structure=struct, border_value=0)
    wall_mask     = building_bool & ~eroded

    # Dach-Hoehen-Raster: fuer jeden Pixel die hoechste Gebaeude-Oberkante in 10px Umgebung.
    # Wird genutzt um Laternen zu verhindern die innerhalb der Gebaeudewand spawnen wuerden.
    print("  Berechne Dach-Hoehen-Raster fuer Laternen-Check...")
    bldg_top_raw = np.zeros(dgm.shape, dtype=np.int32)
    bldg_top_raw[building_bool] = (
        smooth_floor[building_bool]
        + np.clip(np.round(lod2_heights[building_bool]).astype(np.int32), 2, 80)
    )
    from scipy.ndimage import maximum_filter as _maxf
    bldg_top_raster = _maxf(bldg_top_raw, size=21, mode='constant', cval=0)
    # size=21 → 10px Radius um jedes Gebaeude-Pixel

    print("  Berechne Wasser-Tal-Raster (20m Radius)...")
    _dgm_tmp = np.where(np.isnan(dgm),
                        float(np.nanpercentile(dgm[~np.isnan(dgm)], 5)),
                        dgm)
    water_max_raster = _maxf(_dgm_tmp, size=41, mode='nearest')

    print("  Berechne Spitzdach-Offsets...")
    pitched_mask = building_bool & np.isin(lod2_roof_type, [2100, 3100, 3200, 4000])
    roof_offset  = np.zeros(dgm.shape, dtype=np.int32)
    if pitched_mask.any():
        dist_all = distance_transform_edt(building_bool)
        labeled, n_comp = ndlabel(pitched_mask)
        for comp_id in range(1, n_comp + 1):
            comp  = labeled == comp_id
            max_d = float(dist_all[comp].max())
            if max_d < 1:
                continue
            # Dachneigung: max 4 Bloecke (nicht zu aggressiv)
            max_offset = min(max_d, 4)
            scale = max_offset / max_d
            roof_offset[comp] = np.clip(
                (dist_all[comp] * scale).astype(np.int32), 0, 4)
        print(f"    Spitzdach-Pixel: {int(pitched_mask.sum()):,}  "
              f"max Offset: {int(roof_offset.max())} Bloecke")

    # ── Vorberechnung: Tueren-Maske ────────────────────────────────────
    # Wand-Pixel die an mindestens einer Seite kein Gebaeude-Nachbar haben
    # und nicht an einer Strasse grenzen → Tuer-Kandidaten
    print("  Berechne Tuerpositionen...")
    # Wand-Pixel die an Nicht-Gebaeude grenzen (Aussenwand)
    from scipy.ndimage import binary_dilation
    outer_wall  = wall_mask & binary_dilation(~building_bool,
                                               structure=np.ones((3,3), bool))
    # Alle ~8 Pixel eine Tuer (deterministisch)
    row_idx, col_idx = np.where(outer_wall)
    door_mask = np.zeros(dgm.shape, dtype=bool)
    for r, c in zip(row_idx, col_idx):
        if (int(r) * 13 + int(c) * 7) % 8 == 0:
            door_mask[r, c] = True
    # Tuer-Richtung: wohin zeigt die Aussenseite?
    # Pruefe welcher Nachbar kein Gebaeude ist → Tuer zeigt dorthin
    door_facing = np.full(dgm.shape, "south", dtype=object)
    rows_d, cols_d = np.where(door_mask)
    for r, c in zip(rows_d, cols_d):
        r, c = int(r), int(c)
        # Finde Richtungen wo KEIN Gebaeude ist
        free_dirs = []
        for fname, dr, dc in [("south",1,0),("north",-1,0),("east",0,1),("west",0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and not building_bool[nr, nc]:
                free_dirs.append((fname, dr, dc, int(mc_y_grid[nr, nc])))
        if free_dirs:
            # Beste Richtung = niedrigstes Terrain davor
            best = min(free_dirs, key=lambda x: x[3])
            door_facing[r, c] = best[0]
        else:
            door_mask[r, c] = False

    # Laternen-Maske vorberechnen: exakt 1 Laterne pro 20×20-Zelle am Straßenrand
    print("  Berechne Laternen-Positionen...")
    from scipy.ndimage import binary_erosion as _be_lt
    _road_px   = (osm_raster >= 10) & (osm_raster <= 20)
    _road_edge = _road_px & ~_be_lt(_road_px)
    _LSPC      = 20
    lantern_mask = np.zeros(osm_raster.shape, dtype=bool)
    for _gi in range(0, H, _LSPC):
        for _gj in range(0, W, _LSPC):
            _cell = _road_edge[_gi:_gi+_LSPC, _gj:_gj+_LSPC]
            if not _cell.any():
                continue
            _ys, _xs = np.where(_cell)
            _cy, _cx = _LSPC // 2, _LSPC // 2
            _best = int(np.argmin(np.abs(_ys - _cy) + np.abs(_xs - _cx)))
            lantern_mask[_gi + _ys[_best], _gj + _xs[_best]] = True
    print(f"    {int(lantern_mask.sum()):,} Laternen-Kandidaten")

    # Bürgersteig-Maske: 2px-Puffer um Hauptstraßen, nur in Ortslage
    print("  Berechne Bürgersteig-Maske...")
    from scipy.ndimage import binary_dilation as _bd_sw
    _road_px_sw  = np.isin(osm_raster, [10, 11, 12, 13, 14])
    _road_dil_sw = _bd_sw(_road_px_sw, iterations=2)
    _excl_sw     = np.isin(osm_raster, list(range(10, 22)) + [16, 35])  # Straßen, Gleise, Wege, Schotter
    _ort_sw      = ortslage_mask.astype(bool) if ortslage_mask is not None else np.ones(osm_raster.shape, dtype=bool)
    sidewalk_mask = _road_dil_sw & ~_excl_sw & ~(lod2_mask == 1) & _ort_sw
    print(f"    {int(sidewalk_mask.sum()):,} Bürgersteig-Pixel")

    # Baum-Positionen sammeln (nach Terrain setzen, damit Gras nicht ueberschreibt)
    tree_queue = []
    eff_top_grid = np.zeros(dgm.shape, dtype=np.int32)  # tatsächliches top_y nach Runden
    door_clear_list = []
    post_ramp_list  = []  # Treppe HOCH: nach Terrain-Loop setzen
    for row in range(H):
        if row % prog_step == 0:
            print(f"  {row/H*100:.0f}% ...", end="\r", flush=True)
        for col in range(W):
            real_h = dgm[row, col]
            if np.isnan(real_h):
                continue

            sy      = int(mc_y_grid[row, col])
            if bridge_mask is not None and bridge_mask[row, col]:
                # Nur echte Fluss-Überquerungen (>2m Taltiefe) → Brücke absenken
                rdy = int(road_y_grid[row, col])
                if rdy > 0 and float(water_max_raster[row, col]) - float(real_h) > 2.0:
                    sy = rdy - 1
            osm_val = int(osm_raster[row, col])
            bx, bz  = col, row

            # Unterirdische Saeule
            for mc_y in range(sy):
                if   mc_y == 0:          b = "minecraft:bedrock"
                elif mc_y < 30:          b = "minecraft:deepslate"
                elif sy - mc_y > 1:      b = "minecraft:stone"
                else:                    b = "minecraft:dirt"
                writer.set_block(bx, mc_y, bz, b)

            # Oberflaeche
            # Gebaeude hat immer Vorrang (auch vor Strassen)
            if lod2_mask[row, col] == 1:
                # Einheitlicher Boden fuer alle Gebaeude-Pixel inkl. Tueren
                floor_y   = int(smooth_floor[row, col])
                base_y    = int(smooth_base[row, col])
                height_m  = float(lod2_heights[row, col])
                roof_mc_y = int(lod2_roof_mc[row, col])
                r_type    = int(lod2_roof_type[row, col])
                r_offset  = int(roof_offset[row, col])
                top_y     = roof_mc_y if roof_mc_y > base_y else base_y + max(3, int(round(height_m)))
                top_y    += r_offset
                # Flachdach: einheitliche Höhe per Komponente (kein unebenes Dach)
                cid_pre = int(labeled_bldg[row, col])
                if r_type not in _PITCHED_TYPES and cid_pre in comp_flat_top_y:
                    top_y = comp_flat_top_y[cid_pre] + r_offset
                is_door   = bool(door_mask[row, col])
                if is_door:
                    if height_m < 4.0 or (top_y - floor_y) < 3:
                        is_door = False
                facing    = str(door_facing[row, col]) if is_door else "south"

                # Unterirdisch bis Boden fuellen
                for fill_y in range(floor_y - 2, floor_y):
                    writer.set_block(bx, fill_y, bz, "minecraft:stone")

                if is_door:
                    dr = {"north":(-1,0),"south":(1,0),"west":(0,-1),"east":(0,1)}[facing]

                    # Türrahmen = Komponentenmaterial
                    cid_door = int(labeled_bldg[row, col])
                    if church_mask[row, col]:
                        frame_b = "minecraft:stone_bricks"
                    else:
                        frame_b = comp_wall_mat.get(cid_door, ("minecraft:white_concrete",))[0]

                    _ROAD_CLS = {10,11,12,13,14,15,16,17,20,35}
                    perp = [(dr[1], dr[0]), (-dr[1], -dr[0])]
                    for pr, pc in perp:
                        sr, sc = row+pr, col+pc
                        if 0 <= sr < H and 0 <= sc < W and not building_bool[sr, sc]:
                            if int(osm_raster[sr, sc]) in _ROAD_CLS:
                                continue  # kein Türrahmen auf Straße/Gravel
                            ext_frame = int(mc_y_grid[sr, sc])
                            for fy in range(ext_frame, floor_y):
                                writer.set_block(int(sc), fy, int(sr), frame_b)
                            for fy in range(floor_y, floor_y + 4):
                                writer.set_block(int(sc), fy, int(sr), frame_b)

                    flip = {"north":"south","south":"north","west":"east","east":"west"}
                    stair_face_up   = flip[facing]  # terrain tiefer (runter): zum Haus zeigend
                    stair_face_down = facing         # terrain hoeher (hoch): weg vom Haus zeigend

                    nr1 = row + dr[0]; nc1 = col + dr[1]
                    if 0 <= nr1 < H and 0 <= nc1 < W and not building_bool[nr1, nc1]:
                        # Kein Treppenhaus wenn direkt vor der Tür Straße/Gravel liegt
                        if int(osm_raster[int(nr1), int(nc1)]) not in _ROAD_CLS:
                            ext_sy = int(mc_y_grid[nr1, nc1])

                            if ext_sy > floor_y:
                                door_clear_list.append((int(nc1), ext_sy, int(nr1)))
                                post_ramp_list.append(("stone", int(nc1), floor_y, int(nr1)))
                                nr2 = row + dr[0]*2; nc2 = col + dr[1]*2
                                if 0 <= nr2 < H and 0 <= nc2 < W and not building_bool[nr2, nc2]:
                                    ext_sy2 = int(mc_y_grid[nr2, nc2])
                                    if ext_sy2 > floor_y:
                                        for fy in range(floor_y + 2, ext_sy2 + 1):
                                            door_clear_list.append((int(nc2), fy, int(nr2)))
                                        post_ramp_list.append(("stone", int(nc2), floor_y, int(nr2)))
                                        post_ramp_list.append(("stair_down", int(nc2), floor_y + 1, int(nr2), stair_face_down))

                            elif ext_sy == floor_y - 1:
                                # Terrain genau 1 Block tiefer: einzelne Eingangsstufe
                                if int(osm_raster[int(nr1), int(nc1)]) not in _ROAD_CLS:
                                    door_clear_list.append((int(nc1), ext_sy + 1, int(nr1)))
                                    post_ramp_list.append(("stair_up_1", int(nc1), floor_y, int(nr1), stair_face_up))

                            elif ext_sy <= floor_y - 2:
                                # Terrain 2+ Bloecke tiefer: Treppe RUNTER
                                diff = floor_y - ext_sy
                                for step in range(min(diff, 3)):
                                    snr = row + dr[0]*(step+1)
                                    snc = col + dr[1]*(step+1)
                                    if not (0 <= snr < H and 0 <= snc < W) or building_bool[snr, snc]:
                                        break
                                    if int(osm_raster[int(snr), int(snc)]) in _ROAD_CLS:
                                        break  # keine Treppe auf Straße/Gravel
                                    stair_y = floor_y - step
                                    s_ext = int(mc_y_grid[snr, snc])
                                    writer.set_block(int(snc), s_ext, int(snr), "minecraft:stone_bricks")
                                    for fy in range(s_ext + 1, stair_y):
                                        writer.set_block(int(snc), fy, int(snr), "minecraft:stone_bricks")
                                    writer.set_block(int(snc), stair_y, int(snr),
                                        f"minecraft:stone_brick_stairs[facing={stair_face_up},half=bottom,shape=straight]")
                                    door_clear_list.append((int(snc), stair_y + 1, int(snr)))

                cid = cid_pre
                mat = comp_wall_mat.get(cid, (None, None))
                actual_top = _place_building_pixel(writer, bx, bz, floor_y, top_y,
                                      is_wall=bool(wall_mask[row, col]),
                                      roof_type=r_type,
                                      is_door=is_door,
                                      door_facing=facing,
                                      is_church=bool(church_mask[row, col]),
                                      forced_wall_b=mat[0],
                                      forced_roof_b=mat[1])
                eff_top_grid[row, col] = actual_top

            elif sidewalk_mask[row, col]:
                writer.set_block(bx, sy, bz, "minecraft:smooth_stone")

            elif osm_val > 0 and not is_water(osm_val):
                osm_block = get_osm_block(osm_val)
                if osm_block:
                    if osm_block == "minecraft:rail":
                        writer.set_block(bx, sy,     bz, "minecraft:gravel")
                        writer.set_block(bx, sy + 1, bz, "minecraft:rail")
                    else:
                        writer.set_block(bx, sy, bz, osm_block)

                # Bäume nur auf Wald (31) und Wiese (30)
                if osm_val in (30, 31):
                    h        = float(ndsm[row, col])
                    ndsm_cls_val = int(ndsm_classes[row, col])
                    if osm_val == 31 and h < 4.0:
                        # Wald ohne DOM-Daten: synthetische Höhe
                        h = 7.0 + (row * 3 + col * 5) % 5 - 1
                    # Wiese: nur echter DOM-Nachweis (bestätigte Vegetation, cls==2)
                    # Baum: h≥3.0m | Busch: 1.5m≤h<3.0m
                    wiese_baum  = (osm_val == 30 and ndsm_cls_val == 2 and h >= 3.0)
                    wiese_busch = (osm_val == 30 and ndsm_cls_val == 2 and 1.5 <= h < 3.0)
                    wald_ok     = (osm_val == 31 and h > 2.0 and ndsm_cls_val != 1)
                    if (wiese_baum or wiese_busch or wald_ok) and not building_bool[row, col]:
                        # Bäume: 5px Abstand, Büsche: 2px Abstand von Gebäuden
                        _nb_r = 2 if wiese_busch else 5
                        near_building = building_bool[
                            max(0,row-_nb_r):min(H,row+_nb_r+1),
                            max(0,col-_nb_r):min(W,col+_nb_r+1)
                        ].any()
                        # Straße im 1px-Kern blocken (nicht Wegesrand/Parkplatzrand)
                        _core = osm_raster[max(0,row-1):min(H,row+2), max(0,col-1):min(W,col+2)]
                        _on_road = bool(((_core >= 10) & (_core <= 20)).any())
                        if not near_building and not _on_road:
                            seed = (row * 7 + col * 13) % 10
                            if wald_ok:
                                threshold = 4          # Wald: 40%
                                min_gap   = 3 + (row + col) % 3
                                ttype_list = ["spruce","spruce","spruce","spruce","spruce","oak","oak","oak","dark_oak","dark_oak"]
                                use_h = h
                            elif wiese_baum:
                                threshold = 2          # Wiese Baum: 20%
                                min_gap   = 7 + (row + col) % 3
                                ttype_list = ["birch","birch","birch","birch","birch","birch","oak","oak","oak","dark_oak"]
                                use_h = h
                            else:                      # wiese_busch: 40%
                                threshold = 4
                                min_gap   = 3 + (row + col) % 2
                                ttype_list = ["bush"] * 10
                                use_h = 2.5
                            if seed < threshold:
                                too_close = any(
                                    abs(tx - col) <= min_gap and abs(tz - row) <= min_gap
                                    for tx, tz, *_ in tree_queue[-100:]
                                )
                                if not too_close:
                                    type_seed = (row * 11 + col * 7) % 10
                                    tree_queue.append((bx, bz, sy, use_h, ttype_list[type_seed]))

                # Deko auch auf ATKIS-Flaechen
                _place_decoration(writer, bx, bz, sy, row, col,
                                  osm_val, osm_raster, lod2_mask, H, W,
                                  bldg_top_raster=bldg_top_raster,
                                  in_ortslage=ortslage_mask[row, col] if ortslage_mask is not None else True,
                                  lantern_mask=lantern_mask)

            elif osm_val > 0 and is_water(osm_val):
                # class 2 = stehendes Gewässer (Teich/See): immer Wasser
                # class 1 = Fließgewässer: nur wenn Umgebung >1m höher (Talcheck)
                is_still = (osm_val == 2)
                is_valley = is_still or (float(water_max_raster[row, col]) - float(real_h) > 1.0)

                if is_valley:
                    writer.set_block(bx, sy - 1, bz, "minecraft:gravel")
                    writer.set_block(bx, sy,     bz, "minecraft:water[level=7]")
                    writer.set_block(bx, sy + 1, bz, "minecraft:water[level=7]")
                else:
                    # Flache Drainage o.ä. → als Gras rendern
                    writer.set_block(bx, sy, bz, "minecraft:grass_block")
                    _place_decoration(writer, bx, bz, sy, row, col,
                                      30, osm_raster, lod2_mask, H, W,
                                      bldg_top_raster=bldg_top_raster,
                                      in_ortslage=ortslage_mask[row, col] if ortslage_mask is not None else True)

            else:
                # Natuerliche Oberflaeche
                if   real_h > 780: surf = "minecraft:snow_block"
                elif real_h > 700: surf = "minecraft:stone"
                else:              surf = "minecraft:grass_block"
                writer.set_block(bx, sy, bz, surf)

                h = float(ndsm[row, col])
                if h > 3.5 and int(ndsm_classes[row, col]) == 2:
                    _core2 = osm_raster[max(0,row-1):min(H,row+2), max(0,col-1):min(W,col+2)]
                    _on_road2 = bool(((_core2 >= 10) & (_core2 <= 20)).any())
                    if not _on_road2:
                        seed = (row * 5 + col * 19) % 10
                        if seed < 2:
                            tree_queue.append((bx, bz, sy, h, None))

                # Dekoration
                _place_decoration(writer, bx, bz, sy, row, col,
                                  osm_val, osm_raster, lod2_mask, H, W,
                                  bldg_top_raster=bldg_top_raster,
                                  in_ortslage=ortslage_mask[row, col] if ortslage_mask is not None else True,
                                  lantern_mask=lantern_mask)

    # Wandverlaengerung erfolgt jetzt direkt in _place_building_pixel via extended_top
    # Separater Pass: Lücken zwischen verschieden hohen Gebäudeteilen schließen.
    # Für jeden Gebäudepixel: wenn ein Nachbar höher ist, diesen Pixel von top_y bis
    # nb_top mit solidem Material auffüllen (OBERHALB des bereits gerenderten Teils).
    print("  Fülle Höhenübergänge zwischen Gebäudeteilen...")
    STORY_H = 4
    for r in range(H):
        for c in range(W):
            if not building_bool[r, c]: continue
            this_top   = int(eff_top_grid[r, c])
            if this_top == 0: continue
            this_floor = int(smooth_floor[r, c])

            cid = int(labeled_bldg[r, c])
            if church_mask[r, c]:
                wc = "minecraft:stone_bricks"
            else:
                wc = comp_wall_mat.get(cid, (None,))[0] or "minecraft:white_concrete"

            # Basishoehe = ohne Spitzdach-Offset (r_offset)
            this_offset   = int(roof_offset[r, c])
            this_base_top = this_top - this_offset
            max_fill_top  = this_base_top

            for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr2, nc2 = r+dr2, c+dc2
                if not (0 <= nr2 < H and 0 <= nc2 < W): continue
                if not building_bool[nr2, nc2]: continue
                nb_base_t = int(eff_top_grid[nr2, nc2]) - int(roof_offset[nr2, nc2])
                if nb_base_t > max_fill_top and nb_base_t - this_base_top < 80:
                    max_fill_top = nb_base_t

            if max_fill_top <= this_base_top: continue
            # Fülle nur oberhalb des bereits gerenderten (tatsächlichen) Top
            max_nb_top = max_fill_top

            bx = c
            bz = r
            # Bereich OBERHALB des bereits gerenderten Teils mit Fenstern füllen
            for fy in range(this_top, max_nb_top):
                dy = fy - this_floor
                v_win = (dy % STORY_H == 2)
                has_win = ((bx % 3 < 2) or (bz % 3 < 2)) and v_win
                if fy == max_nb_top - 1:
                    # oberster Block = Dachfarbe des Nachbarn
                    cid_nb = int(labeled_bldg[r, c])
                    nb_roof = comp_wall_mat.get(cid_nb, (wc, wc))[1]
                    writer.set_block(bx, fy, bz, nb_roof)
                else:
                    writer.set_block(bx, fy, bz, "minecraft:glass" if has_win else wc)
    for cx, cy, cz in door_clear_list:
        writer.delete_block(cx, cy,     cz)
        writer.delete_block(cx, cy + 1, cz)
        writer.delete_block(cx, cy + 2, cz)
    _ROAD_SET = {10,11,12,13,14,15,16,17,20,35}
    for item in post_ramp_list:
        if item[0] == "stone":
            _, x, y, z = item
            if int(osm_raster[z, x]) in _ROAD_SET:
                continue
            writer.set_block(x, y, z, "minecraft:stone_bricks")
        elif item[0] == "stair_down":
            _, x, y, z, face = item
            if int(osm_raster[z, x]) in _ROAD_SET:
                continue
            writer.set_block(x, y, z,
                f"minecraft:stone_brick_stairs[facing={face},half=bottom,shape=straight]")
        elif item[0] == "stair_up_1":
            _, x, y, z, face = item
            if int(osm_raster[z, x]) in _ROAD_SET:
                continue
            writer.set_block(x, y, z,
                f"minecraft:stone_brick_stairs[facing={face},half=bottom,shape=straight]")
    print(f"\n  Setze {len(tree_queue):,} Baeume...")
    for bx, bz, sy, h, ttype in tree_queue:
        _place_tree(writer, bx, bz, sy, h, ttype, building_mask=building_bool)

    # ── Kirchtuerme aus ATKIS ──────────────────────────────────────────
    if CHURCH_POINTS:
        print(f"\n  Setze {len(CHURCH_POINTS)} Kirchtuerme...")
        for utm_x, utm_y in CHURCH_POINTS:
            # UTM → Raster-Index
            col_f = (utm_x - dgm_ox) / dgm_res
            row_f = (dgm_oy + dgm.shape[0] * dgm_res - utm_y) / dgm_res
            col_i = int(round(col_f))
            row_i = int(round(row_f))
            if not (0 <= row_i < dgm.shape[0] and 0 <= col_i < dgm.shape[1]):
                continue
            bx, bz = col_i, row_i
            sy = int(mc_y_grid[row_i, col_i])
            _place_church(writer, bx, bz, sy)

    print("\n  Bevoelkere die Stadt...")
    entity_writer = EntityWriter(_out,
                                 x_offset=writer.x_offset,
                                 z_offset=writer.z_offset,
                                 tile_min_x=writer._xmin,
                                 tile_max_x=writer._xmax,
                                 tile_min_z=writer._zmin,
                                 tile_max_z=writer._zmax)
    populate_city(writer, entity_writer,
                  lod2_mask, lod2_heights, lod2_roof_mc, door_mask,
                  smooth_floor, mc_y_grid, dgm.shape)
    entity_writer.save()

    if _is_tile:
        print("\n  Speichere Tile-Regions...")
        writer.save()
        return

    print("\n  Speichere Regions...")
    writer.save()

    write_level_dat(OUTPUT_DIR, spawn_x=100, spawn_y=100, spawn_z=8)

    # Automatisch in Minecraft saves kopieren
    import shutil
    saves_dir = os.path.join(os.environ.get("APPDATA", ""), ".minecraft", "saves", OUTPUT_DIR)
    if os.path.exists(saves_dir):
        shutil.rmtree(saves_dir)
    shutil.copytree(OUTPUT_DIR, saves_dir)
    print(f"  → Kopiert nach: {saves_dir}")

    print(f"\n  ✓  {OUTPUT_DIR}/")
    print("    → %AppData%\\.minecraft\\saves\\  (Windows)")


def _place_decoration(writer, bx, bz, sy, row, col,
                      osm_val, osm_raster, lod2_mask, H, W,
                      bldg_top_raster=None, in_ortslage=True, lantern_mask=None):
    seed  = (row * 17 + col * 31) % 100
    seed2 = (row * 53 + col * 7)  % 10
    # Blumen-Cluster: ~30% der 25×25-Kacheln sind Blumenzonen
    _zone_seed = ((row // 25) * 41 + (col // 25) * 67) % 100
    in_flower_zone = _zone_seed < 30

    # Keine Deko auf Strassen/Wegen/Bahn (10-20) oder Schotter/Acker (32-33, 35-36)
    if 10 <= osm_val <= 20 or osm_val in (32, 35, 36):
        # Laternen NUR auf echten Straßen innerhalb Ortslage, vorberechnete Maske
        _on_grid = lantern_mask is not None and bool(lantern_mask[row, col])
        if osm_val in (10, 11, 12, 13, 14) and _on_grid and in_ortslage:
            # Outward-Richtung: erster Nicht-Straße-Nachbar = Bürgersteig/Gras
            outward = None
            for _dr, _dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                _nr, _nc = row+_dr, col+_dc
                if 0 <= _nr < H and 0 <= _nc < W and not (10 <= int(osm_raster[_nr, _nc]) <= 20):
                    outward = (_dr, _dc, _nr, _nc)
                    break
            if outward is not None:
                _odr, _odc, _onr, _onc = outward
                if lod2_mask[_onr, _onc] == 1:
                    # Outward-Pixel ist Gebäude → Fahrbahnkante
                    lbx, lbz = col, row
                elif (bldg_top_raster is not None
                      and 0 < int(bldg_top_raster[_onr, _onc]) <= sy + 5):
                    # Gebäudedach auf Laternen-Höhe oder darunter → überspringen
                    return
                else:
                    lbx, lbz = _onc, _onr
            else:
                # Schmale Straße: kein freier Nachbar → Fahrbahnkante (keine Dach-Prüfung)
                lbx, lbz = col, row
            for _dy, _f in enumerate(["up","down","up","down","up"], start=1):
                writer.set_block(lbx, sy + _dy, lbz, f"minecraft:end_rod[facing={_f}]")
            writer.set_block(lbx, sy + 6, lbz, "minecraft:lantern[hanging=false]")
        return

    near_building = lod2_mask[
        max(0,row-3):min(H,row+4),
        max(0,col-3):min(W,col+4)
    ].any()

    FLOWERS = [
        "minecraft:dandelion", "minecraft:poppy",
        "minecraft:blue_orchid", "minecraft:allium",
        "minecraft:azure_bluet", "minecraft:cornflower",
        "minecraft:oxeye_daisy", "minecraft:red_tulip",
        "minecraft:white_tulip", "minecraft:orange_tulip",
        "minecraft:pink_tulip", "minecraft:lily_of_the_valley",
    ]
    TALL_FLOWERS = [
        "minecraft:sunflower", "minecraft:lilac",
        "minecraft:rose_bush", "minecraft:peony",
    ]

    near_road = bool(((osm_raster[
        max(0,row-2):min(H,row+3),
        max(0,col-2):min(W,col+3)
    ] >= 10) & (osm_raster[
        max(0,row-2):min(H,row+3),
        max(0,col-2):min(W,col+3)
    ] <= 20)).any())


    # Blumen / hohe Pflanzen nah an Gebaeuden (Vorgaerten)
    # Dichte nur in Blumenzonen, sonst sehr spärlich
    flower_thresh = 20 if in_flower_zone else 3
    if near_building and seed < flower_thresh:
        if seed2 >= 8:
            tf = TALL_FLOWERS[seed % len(TALL_FLOWERS)]
            writer.set_block(bx, sy + 1, bz, f"{tf}[half=lower]")
            writer.set_block(bx, sy + 2, bz, f"{tf}[half=upper]")
        else:
            writer.set_block(bx, sy + 1, bz, FLOWERS[seed % len(FLOWERS)])
        return

    # Blumen auf Wiesen — nur in Blumenzonen dicht
    meadow_flower_thresh = 18 if in_flower_zone else 2
    if not near_building and seed < meadow_flower_thresh:
        if seed2 == 9:
            tf = TALL_FLOWERS[seed % len(TALL_FLOWERS)]
            writer.set_block(bx, sy + 1, bz, f"{tf}[half=lower]")
            writer.set_block(bx, sy + 2, bz, f"{tf}[half=upper]")
        else:
            writer.set_block(bx, sy + 1, bz, FLOWERS[seed % len(FLOWERS)])
        return

    # Hohes Gras / Farn / normales Gras auf Wiesen
    if not near_building and meadow_flower_thresh <= seed < meadow_flower_thresh + 30:
        if seed2 < 3:
            writer.set_block(bx, sy + 1, bz, "minecraft:tall_grass[half=lower]")
            writer.set_block(bx, sy + 2, bz, "minecraft:tall_grass[half=upper]")
        elif seed2 < 8:
            writer.set_block(bx, sy + 1, bz, "minecraft:short_grass")
        else:
            writer.set_block(bx, sy + 1, bz, "minecraft:fern")
        return

    # Zuckerrohr am Wasser
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = row+dr, col+dc
        if 0 <= nr < H and 0 <= nc < W:
            ov = int(osm_raster[nr, nc])
            if ov > 0 and is_water(ov) and seed < 40:
                writer.set_block(bx, sy + 1, bz, "minecraft:sugar_cane")
                if seed < 20:
                    writer.set_block(bx, sy + 2, bz, "minecraft:sugar_cane")
                return



def _is_corner_wall(row, col, wall_mask, building_bool, H, W):
    """True wenn dieser Pixel in einer Ecke sitzt: hat Wandnachbarn in 2 verschiedenen Achsen."""
    has_ns = False  # Nord oder Sued
    has_ew = False  # Ost oder West
    for dr, dc in [(-1,0),(1,0)]:
        nr, nc = row+dr, col+dc
        if 0 <= nr < H and 0 <= nc < W and wall_mask[nr, nc]:
            has_ns = True
    for dr, dc in [(0,-1),(0,1)]:
        nr, nc = row+dr, col+dc
        if 0 <= nr < H and 0 <= nc < W and wall_mask[nr, nc]:
            has_ew = True
    return has_ns and has_ew  # Ecke = Waende in beiden Achsen


def _place_building_pixel(writer, bx, bz, floor_y, top_y, is_wall,
                          roof_type=1000, is_door=False, door_facing="south",
                          is_church=False, forced_wall_b=None, forced_roof_b=None):
    raw_h    = top_y - floor_y
    STORY_H  = 4
    has_pitched = roof_type in (2100, 3100, 3200, 4000)
    if has_pitched:
        # Spitzdach: rohe Höhe behalten (r_offset formt bereits die Pyramide)
        h_blocks = min(max(3, raw_h), 80)
    else:
        # Flachdach: auf Vielfaches von STORY_H runden + 1 Dachkante
        # +1 damit hängende Innen-Laternen (lamp_y = n*STORY_H - 1) UNTER dem Dachblock liegen
        h_blocks = max(STORY_H, int((raw_h + STORY_H // 2) // STORY_H) * STORY_H) + 1
        h_blocks = min(h_blocks, 80)

    if is_church:
        wall_b = "minecraft:stone_bricks"
        if   h_blocks < 4:  roof_b = "minecraft:stone_brick_slab"
        elif h_blocks < 10: roof_b = "minecraft:gray_concrete"
        elif h_blocks < 20: roof_b = "minecraft:gray_concrete"
        else:               roof_b = "minecraft:stone_bricks"
    elif forced_wall_b:
        wall_b = forced_wall_b
        roof_b = forced_roof_b or wall_b
    elif h_blocks < 4:  wall_b, roof_b = "minecraft:stone_bricks",       "minecraft:stone_brick_slab"
    elif h_blocks < 10: wall_b, roof_b = "minecraft:white_concrete",      "minecraft:gray_concrete"
    elif h_blocks < 20: wall_b, roof_b = "minecraft:light_gray_concrete", "minecraft:gray_concrete"
    else:               wall_b, roof_b = "minecraft:quartz_block",        "minecraft:smooth_quartz"

    FLOOR_B     = "minecraft:oak_planks"
    RED_ROOF    = "minecraft:red_concrete"
    has_pitched = roof_type in (2100, 3100, 3200, 4000)

    def has_window(x, z):
        return (x % 3 < 2) or (z % 3 < 2)

    for fill_y in range(floor_y - 2, floor_y):
        writer.set_block(bx, fill_y, bz, "minecraft:stone")

    # Dachabschluss: bei Spitzdach = Wandfarbe als Kante, sonst Dachfarbe
    top_block = wall_b if has_pitched else roof_b

    if is_wall:
        for dy in range(h_blocks):
            mc_y = floor_y + dy
            if dy == h_blocks - 1:
                writer.set_block(bx, mc_y, bz, top_block)
            elif dy % STORY_H == 0:
                writer.set_block(bx, mc_y, bz, wall_b)
            elif is_door and dy == 1:
                flip = {"north":"south","south":"north","west":"east","east":"west"}
                inner_facing = flip[door_facing]
                writer.set_block(bx, mc_y,     bz,
                    f"minecraft:oak_door[half=lower,facing={inner_facing},hinge=left,open=false]")
                writer.set_block(bx, mc_y + 1, bz,
                    f"minecraft:oak_door[half=upper,facing={inner_facing},hinge=left,open=false]")
            elif is_door and dy == 2:
                pass
            else:
                v_win = (dy % STORY_H == 2)
                writer.set_block(bx, mc_y, bz,
                                 "minecraft:glass" if (v_win and has_window(bx, bz)) else wall_b)
        if has_pitched:
            writer.set_block(bx, floor_y + h_blocks, bz, RED_ROOF)

    else:
        n_full_stories = h_blocks // STORY_H
        for story in range(n_full_stories + 1):
            story_y = floor_y + story * STORY_H
            if story_y < floor_y + h_blocks - 2:   # nicht direkt unter Dach
                writer.set_block(bx, story_y, bz, FLOOR_B)
        writer.set_block(bx, floor_y + h_blocks - 1, bz, top_block)
        if has_pitched:
            writer.set_block(bx, floor_y + h_blocks, bz, RED_ROOF)
        # Hängende Deckenlaterne pro Stockwerk, jede ~4x4-Zelle einmal
        if not has_pitched and (bx % 4 == 2) and (bz % 4 == 2):
            for story in range(n_full_stories):
                lamp_y = floor_y + (story + 1) * STORY_H - 1
                writer.set_block(bx, lamp_y, bz, "minecraft:lantern[hanging=true]")

    return floor_y + h_blocks  # tatsächliche gerenderte Top-Y


def _write_level_dat():
    import nbtlib, time
    data = nbtlib.Compound({
        "Data": nbtlib.Compound({
            "version":           nbtlib.Int(19133),
            "DataVersion":       nbtlib.Int(3700),
            "LevelName":         nbtlib.String("Thueringen"),
            "SpawnX":            nbtlib.Int(1500),
            "SpawnY":            nbtlib.Int(200),
            "SpawnZ":            nbtlib.Int(1500),
            "GameType":          nbtlib.Int(1),
            "hardcore":          nbtlib.Byte(0),
            "Difficulty":        nbtlib.Byte(2),
            "allowCommands":     nbtlib.Byte(1),
            "initialized":       nbtlib.Byte(1),
            "LastPlayed":        nbtlib.Long(int(time.time() * 1000)),
            "Time":              nbtlib.Long(6000),
            "DayTime":           nbtlib.Long(6000),
            "raining":           nbtlib.Byte(0),
            "thundering":        nbtlib.Byte(0),
            "clearWeatherTime":  nbtlib.Int(100000),
            "WorldGenSettings":  nbtlib.Compound({
                "bonus_chest":       nbtlib.Byte(0),
                "generate_features": nbtlib.Byte(0),
                "seed":              nbtlib.Long(0),
                "dimensions": nbtlib.Compound({
                    "minecraft:overworld": nbtlib.Compound({
                        "type": nbtlib.String("minecraft:overworld"),
                        "generator": nbtlib.Compound({
                            "type": nbtlib.String("minecraft:flat"),
                            "settings": nbtlib.Compound({
                                "biome":    nbtlib.String("minecraft:plains"),
                                "layers":   nbtlib.List[nbtlib.Compound]([]),
                                "lakes":    nbtlib.Byte(0),
                                "features": nbtlib.Byte(0),
                            }),
                        }),
                    }),
                }),
            }),
            "Version": nbtlib.Compound({
                "Id":       nbtlib.Int(3700),
                "Name":     nbtlib.String("1.20.4"),
                "Series":   nbtlib.String("main"),
                "Snapshot": nbtlib.Byte(0),
            }),
        })
    })
    out = os.path.join(OUTPUT_DIR, "level.dat")
    nbtlib.File(data).save(out, gzipped=True)
    print(f"  level.dat geschrieben")


# ─────────────────────────────────────────────
# VORSCHAU
# ─────────────────────────────────────────────

def save_preview(dgm, ndsm_cls, ndsm, osm_raster, lod2_mask):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Thueringen → Minecraft  |  Layer-Vorschau", fontsize=15)

        axes[0,0].imshow(dgm, cmap="terrain", origin="upper")
        axes[0,0].set_title("DGM – Gelaendehoehe (m ue. NN)")
        axes[0,0].axis("off")

        axes[0,1].imshow(ndsm, cmap="YlOrBr", vmin=0, vmax=20, origin="upper")
        axes[0,1].set_title("nDSM – Objekte ueber Gelaende (m)")
        axes[0,1].axis("off")

        axes[0,2].imshow(lod2_mask, cmap="Reds", origin="upper")
        axes[0,2].set_title("LoD2-Gebaeude (echte Grundrisse)")
        axes[0,2].axis("off")

        cmap3 = mcolors.ListedColormap(["#7cba5a", "#e8d5a3", "#3a7d44"])
        axes[1,0].imshow(ndsm_cls, cmap=cmap3, vmin=0, vmax=2, origin="upper")
        axes[1,0].set_title("nDSM-Klassifikation: Boden / Gebaeude / Vegetation")
        axes[1,0].legend(handles=[
            mpatches.Patch(color="#7cba5a", label="Boden"),
            mpatches.Patch(color="#e8d5a3", label="Gebaeude (nDSM)"),
            mpatches.Patch(color="#3a7d44", label="Vegetation"),
        ], loc="lower right", fontsize=8)
        axes[1,0].axis("off")

        # OSM
        CM = {1:(30,144,255),2:(0,191,255),10:(60,60,60),11:(100,100,100),
              12:(140,140,140),13:(180,180,180),14:(210,210,210),15:(160,120,80),
              16:(200,180,140),17:(80,160,80),20:(140,60,180),30:(100,180,80)}
        vis = np.full((*osm_raster.shape, 3), [240,235,225], dtype=np.uint8)
        for v, c in CM.items():
            vis[osm_raster == v] = c
        axes[1,1].imshow(vis, origin="upper")
        axes[1,1].set_title("OSM: Strassen · Wasser · Bahn")
        leg = [mpatches.Patch(color=np.array(c)/255, label=OSM_CLASSES[k][3])
               for k, c in CM.items() if (osm_raster == k).any()]
        if leg:
            axes[1,1].legend(handles=leg, loc="lower right", fontsize=7, ncol=2)
        axes[1,1].axis("off")

        # Kombiniert
        combined = np.zeros((*dgm.shape, 3), dtype=np.uint8)
        mc_y_norm = height_to_mc_y(dgm).astype(float)
        mc_y_norm = ((mc_y_norm - mc_y_norm.min()) /
                     (np.ptp(mc_y_norm) + 1e-6) * 200 + 55).astype(np.uint8)
        combined[:,:,0] = combined[:,:,1] = combined[:,:,2] = mc_y_norm
        combined[ndsm_cls == 2] = [34, 100, 34]
        combined[lod2_mask == 1] = [200, 200, 210]
        combined[osm_raster == 1] = [30, 100, 200]
        combined[osm_raster == 2] = [30, 100, 200]
        for v in range(10, 18):
            combined[osm_raster == v] = [90, 90, 90]
        axes[1,2].imshow(combined, origin="upper")
        axes[1,2].set_title("Kombiniert: Terrain + LoD2 + OSM + Vegetation")
        axes[1,2].axis("off")

        out = os.path.join(OUTPUT_DIR, "vorschau.png")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Vorschau: {out}")
    except ImportError:
        print("  (matplotlib fehlt – keine Vorschau)")


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# TILED MODE
# ─────────────────────────────────────────────

def _tile_worker(args):
    """Top-level Worker-Funktion für ProcessPoolExecutor (muss pickelbar sein)."""
    (c0, c1, r0, r1, lc0, lc1, lr0, lr1,
     dgm_t, ndsm_cls_t, ndsm_t, osm_t,
     lod2_mask_t, lod2_h_t, lod2_rmc_t, lod2_rtype_t,
     bridge_t, ortslage_t, church_mask_t,
     tile_ox, tile_oy, dgm_res,
     church_points, alkis_buildings, output_dir, real_min_h) = args

    global REAL_MIN_H
    REAL_MIN_H = real_min_h

    from anvil_writer import WorldWriter

    writer = WorldWriter(
        output_dir,
        x_offset=lc0, z_offset=lr0,
        tile_min_x=c0, tile_max_x=c1 - 1,
        tile_min_z=r0, tile_max_z=r1 - 1,
    )
    write_minecraft_world(
        dgm_t, ndsm_cls_t, ndsm_t,
        osm_t, lod2_mask_t, lod2_h_t,
        lod2_rmc_t, lod2_rtype_t,
        dgm_ox=tile_ox, dgm_oy=tile_oy, dgm_res=dgm_res,
        buildings=None,                   # Geometrien nicht pickle-n; Kirche via precomp
        alkis_buildings=alkis_buildings,
        bridge_mask=bridge_t, ortslage_mask=ortslage_t,
        writer=writer, _is_tile=True,
        output_dir=output_dir,
        precomp_church_mask=church_mask_t,
        church_points=church_points,
    )
    return c0, r0


def _write_tiled(dgm, ndsm_classes, ndsm,
                 osm_raster, lod2_mask, lod2_heights,
                 lod2_roof_mc, lod2_roof_type,
                 dgm_ox=0, dgm_oy=0, dgm_res=1.0,
                 buildings=None, alkis_buildings=None,
                 bridge_mask=None, ortslage_mask=None,
                 tile_size=512, n_workers=4):
    import gc
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from anvil_writer import WorldWriter, write_level_dat

    H, W    = dgm.shape
    OVERLAP = 64
    n_tx    = max(1, -(-W // tile_size))
    n_tz    = max(1, -(-H // tile_size))
    total   = n_tx * n_tz
    print(f"  Tiled-Modus: {n_tx}×{n_tz} = {total} Tiles, {n_workers} Worker")

    # Kirchen einmal im Main-Prozess berechnen → als numpy-Mask weitergeben
    print("  Vorberechnung Kirchen-Maske...")
    from atkis_layer import CHURCH_POINTS as _CP
    _church_pts = list(_CP) if _CP else []
    # Einfache Vorab-Berechnung: nur Methode 2 (>40m) geht ohne Geometrien
    # Methode 1+3 werden in den Workern via precomp_church_mask geliefert
    from scipy.ndimage import label as _ndlabel
    _bldg_bool   = lod2_mask == 1
    _labeled, _  = _ndlabel(_bldg_bool)
    _church_mask = np.zeros(dgm.shape, dtype=bool)
    # Methode 2: >40m
    for _cid in np.unique(_labeled):
        if _cid == 0: continue
        _comp = _labeled == _cid
        if float(lod2_heights[_comp].max()) > 40:
            _church_mask[_comp] = True
    # Methode 1: ATKIS CHURCH_POINTS + buildings-Geometrien (Main-Prozess hat sie)
    if _church_pts and buildings:
        from shapely.geometry import Point
        _dgm_north = dgm_oy + H * dgm_res
        for _cx, _cy in _church_pts:
            _pt = Point(_cx, _cy)
            for _bldg in buildings:
                _fp = _bldg.get("footprint")
                if _fp is None or _fp.distance(_pt) >= 150: continue
                try: _tp = _fp.representative_point()
                except: _tp = _fp.centroid
                _ci = int(round((_tp.x - dgm_ox) / dgm_res))
                _ri = int(round((_dgm_north - _tp.y) / dgm_res))
                if 0 <= _ri < H and 0 <= _ci < W:
                    _cid2 = int(_labeled[_ri, _ci])
                    if _cid2 > 0: _church_mask[_labeled == _cid2] = True
    print(f"    {int(_church_mask.sum()):,} Kirchen-Pixel")

    # Tile-Args zusammenstellen
    tile_args = []
    for tz in range(n_tz):
        for tx in range(n_tx):
            c0 = tx * tile_size;  c1 = min(c0 + tile_size, W)
            r0 = tz * tile_size;  r1 = min(r0 + tile_size, H)
            lc0 = max(0, c0 - OVERLAP);  lc1 = min(W, c1 + OVERLAP)
            lr0 = max(0, r0 - OVERLAP);  lr1 = min(H, r1 + OVERLAP)

            def _sl(a, _lr0=lr0, _lr1=lr1, _lc0=lc0, _lc1=lc1):
                return a[_lr0:_lr1, _lc0:_lc1].copy() if a is not None else None

            tile_ox = dgm_ox + lc0 * dgm_res
            tile_oy = dgm_oy + (H - lr1) * dgm_res

            tile_args.append((
                c0, c1, r0, r1, lc0, lc1, lr0, lr1,
                _sl(dgm), _sl(ndsm_classes), _sl(ndsm), _sl(osm_raster),
                _sl(lod2_mask), _sl(lod2_heights), _sl(lod2_roof_mc), _sl(lod2_roof_type),
                _sl(bridge_mask), _sl(ortslage_mask), _sl(_church_mask),
                tile_ox, tile_oy, dgm_res,
                _church_pts, alkis_buildings, OUTPUT_DIR, REAL_MIN_H,
            ))

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = {exe.submit(_tile_worker, a): (a[0], a[2]) for a in tile_args}
        for fut in as_completed(futs):
            c0_d, r0_d = futs[fut]
            done += 1
            try:
                fut.result()
                print(f"  [{done}/{total}] Tile col={c0_d} row={r0_d}  ✓")
            except Exception as exc:
                print(f"  [{done}/{total}] Tile col={c0_d} row={r0_d}  FEHLER: {exc}")

    write_level_dat(OUTPUT_DIR, spawn_x=W // 2, spawn_y=100, spawn_z=H // 2)
    import shutil
    saves_dir = os.path.join(os.environ.get("APPDATA", ""), ".minecraft", "saves", OUTPUT_DIR)
    if os.path.exists(saves_dir):
        shutil.rmtree(saves_dir)
    shutil.copytree(OUTPUT_DIR, saves_dir)
    print(f"  → Kopiert nach: {saves_dir}")


# ─────────────────────────────────────────────
# ARGS & MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Thueringen DGM/DOM/LoD2 → Minecraft"
    )
    p.add_argument("--list",   action="store_true",
                   help="Verfuegbare Kacheln auflisten")
    p.add_argument("--bbox",   nargs=4, type=float,
                   metavar=("W","E","S","N"),
                   help="Ausschnitt in UTM32N, z.B.: --bbox 634000 636000 5614000 5616000")
    p.add_argument("--dgm-dir",  default=DGM_DIR)
    p.add_argument("--dom-dir",  default=DOM_DIR)
    p.add_argument("--lod2-dir", default=LOD2_DIR)
    p.add_argument("--out",      default=OUTPUT_DIR)
    p.add_argument("--workers",  type=int, default=TILE_WORKERS)
    p.add_argument("--no-osm",   action="store_true",
                   help="OSM-Schritt ueberspringen (offline / schneller)")
    p.add_argument("--no-lod2",  action="store_true",
                   help="LoD2-Schritt ueberspringen")
    p.add_argument("--tiled",    action="store_true",
                   help="Tile-by-Tile verarbeiten (spart RAM bei großen Gebieten)")
    p.add_argument("--tile-size", type=int, default=512,
                   help="Tile-Breite in Metern / Blöcken (default: 512 = 1 Region)")
    return p.parse_args()


def main():
    args = parse_args()
    global OUTPUT_DIR
    OUTPUT_DIR = args.out

    if args.list:
        for d in [args.dgm_dir, args.dom_dir, args.lod2_dir]:
            if Path(d).exists():
                list_available_tiles(d)
        return

    bbox = ({"west": args.bbox[0], "east": args.bbox[1],
             "south": args.bbox[2], "north": args.bbox[3]}
            if args.bbox else BBOX)

    print("=" * 58)
    print("  Thüringen → Minecraft  (v4 · TIF + LoD2 + OSM)")
    print("=" * 58)
    if bbox:
        print(f"  BBox: W{bbox['west']} E{bbox['east']} "
              f"S{bbox['south']} N{bbox['north']}")

    # 1. DGM
    print(f"\n[1/7] DGM aus {args.dgm_dir}/")
    dgm, dgm_ox, dgm_oy, dgm_res = load_all_tiles(
        args.dgm_dir, bbox=bbox, workers=args.workers, label="DGM")

    # REAL_MIN_H automatisch aus DGM berechnen
    global REAL_MIN_H
    valid_dgm = dgm[~np.isnan(dgm)]
    REAL_MIN_H = float(np.percentile(valid_dgm, 1)) if len(valid_dgm) else 0.0
    print(f"  REAL_MIN_H (auto): {REAL_MIN_H:.1f} m NN")

    # 2. DOM
    print(f"\n[2/7] DOM aus {args.dom_dir}/")
    dom_path = Path(args.dom_dir)
    if dom_path.exists() and (list(dom_path.glob("*.tif")) or list(dom_path.glob("*.xyz"))):
        dom, dom_ox, dom_oy, dom_res = load_all_tiles(
            args.dom_dir, bbox=bbox, workers=args.workers, label="DOM")
    else:
        print("  ⚠  Kein DOM gefunden – nDSM entfaellt, nur LoD2-Gebaeude")
        dom, dom_ox, dom_oy, dom_res = dgm.copy(), dgm_ox, dgm_oy, dgm_res

    # 3. nDSM
    print("\n[3/7] nDSM berechnen (DOM − DGM) fuer Vegetation...")
    ndsm    = compute_ndsm(dgm, dgm_ox, dgm_oy, dgm_res,
                           dom, dom_ox, dom_oy, dom_res)
    # nDSM-Klassifikation fuer Gebaeude die NICHT in LoD2 sind (z.B. Kaufland, Eishalle)
    # → verhindert dass grosse Flachdach-Gebaeude als Wald gerendert werden
    ndsm_cls = classify_ndsm(ndsm)

    # 4. LoD2
    lod2_mask      = np.zeros(dgm.shape, dtype=np.uint8)
    lod2_heights   = np.zeros(dgm.shape, dtype=np.float32)
    lod2_roof_mc   = np.zeros(dgm.shape, dtype=np.int32)
    lod2_roof_type = np.zeros(dgm.shape, dtype=np.uint16)

    if not args.no_lod2 and Path(args.lod2_dir).exists():
        print(f"\n[4/7] LoD2-Gebaeude aus {args.lod2_dir}/")
        actual_bbox = {
            "west":  dgm_ox,
            "east":  dgm_ox + dgm.shape[1] * dgm_res,
            "south": dgm_oy,
            "north": dgm_oy + dgm.shape[0] * dgm_res,
        }
        # Kleiner Puffer (100m) nur fuer Gebaeude die exakt auf Kachelgrenzen liegen.
        # Kein grosser Buffer mehr - verhindert dass GML-Dateien anderer Staedte geladen werden.
        LOD2_BUFFER = 100
        lod2_load_bbox = {
            "west":  actual_bbox["west"]  - LOD2_BUFFER,
            "east":  actual_bbox["east"]  + LOD2_BUFFER,
            "south": actual_bbox["south"] - LOD2_BUFFER,
            "north": actual_bbox["north"] + LOD2_BUFFER,
        }
        buildings = load_all_lod2(args.lod2_dir, bbox=lod2_load_bbox)
        if buildings:
            lod2_mask, lod2_heights, lod2_roof_mc, lod2_roof_type = rasterize_buildings(
                buildings, actual_bbox, dgm.shape,
                real_min_h=REAL_MIN_H)  # korrekter Offset fuer jede Stadt!
    else:
        print("\n[4/7] LoD2 uebersprungen")

    # ALKIS-Gebaeude laden (GFK-Codes fuer semantische Typen: Kirche, Schule, Rathaus...)
    alkis_buildings = []
    if not args.no_osm:
        try:
            from alkis_loader import load_alkis_buildings, get_church_footprints, GFK_KIRCHE
            print("\n  Lade ALKIS-Gebaeude (GFK-Codes)...")
            alkis_buildings = load_alkis_buildings(actual_bbox)
            alkis_church_footprints = get_church_footprints(alkis_buildings)
            if alkis_church_footprints:
                print(f"  ALKIS Kirchen: {len(alkis_church_footprints)} Gebaeude mit GFK-Code")
        except Exception as e:
            print(f"  ⚠  ALKIS-Gebaeude nicht geladen: {e}")

    # 5. OSM
    osm_raster = np.zeros(dgm.shape, dtype=np.uint8)
    if not args.no_osm:
        print("\n[5/7] OSM-Daten laden...")
        actual_bbox = {
            "west":  dgm_ox,
            "east":  dgm_ox + dgm.shape[1] * dgm_res,
            "south": dgm_oy,
            "north": dgm_oy + dgm.shape[0] * dgm_res,
        }
        features   = load_osm_layers(actual_bbox)
        osm_raster = burn_osm_to_raster(features, actual_bbox, dgm.shape)

        # Originale Wasserroute rastern (vor Priority-Overwrite durch Gravel/Straße)
        # → wird später für Infrastrukturgebäude-Bereinigung genutzt
        from rasterio.features import rasterize as _rast_w
        from rasterio.transform import from_bounds as _tfb_w
        _w_transform = _tfb_w(actual_bbox["west"], actual_bbox["south"],
                               actual_bbox["east"],  actual_bbox["north"],
                               dgm.shape[1], dgm.shape[0])
        water_path_mask = np.zeros(dgm.shape, dtype=bool)
        for _wg, _wc, _wp in features:
            if _wc in (1, 2):
                try:
                    _wb = _rast_w([(_wg, 1)], out_shape=dgm.shape,
                                  transform=_w_transform, fill=0,
                                  dtype=np.uint8, all_touched=True)
                    water_path_mask |= _wb > 0
                except Exception:
                    pass

        # Ortslage-Maske: Laternen nur innerhalb AX_Ortslage
        ortslage_polys = load_ortslage_polygons(actual_bbox)
        ortslage_mask = np.zeros(dgm.shape, dtype=bool)
        if ortslage_polys:
            from rasterio.features import rasterize as _rasterize
            from rasterio.transform import from_bounds as _from_bounds
            _transform = _from_bounds(actual_bbox["west"], actual_bbox["south"],
                                      actual_bbox["east"], actual_bbox["north"],
                                      dgm.shape[1], dgm.shape[0])
            _burned = _rasterize([(g, 1) for g in ortslage_polys],
                                  out_shape=dgm.shape, transform=_transform,
                                  fill=0, dtype=np.uint8)
            ortslage_mask = _burned > 0
            print(f"  Ortslage-Maske: {int(ortslage_mask.sum()):,} px")

        # Straßen haben höchste Priorität (prio 15-20) → überschreiben alles im Raster.
        # Brücken: Straßenpixel die auf der originalen Wasserroute liegen → bridge_mask.
        road_mask    = (osm_raster >= 10) & (osm_raster <= 20)
        water_mask   = np.isin(osm_raster, [1, 2])
        bldg_mask_2d = lod2_mask == 1
        bridge_mask  = np.zeros(osm_raster.shape, dtype=bool)

        cross_mask = water_path_mask & road_mask
        if cross_mask.any():
            bridge_mask |= cross_mask
            print(f"  Brücken erkannt: {int(cross_mask.sum())} px (Straße × Wasserroute)")

        # Sichtbares Wasser unter Gebäuden entfernen
        remove_mask = water_mask & bldg_mask_2d
        osm_raster[remove_mask] = 0
        if remove_mask.any():
            print(f"  Wasser an Gebaeuden entfernt: {int(remove_mask.sum())} px")

        # Straße/Gravel überschreibt LoD2 (<6m) → keine stone_bricks auf Fahrbahn
        _road_gravel = road_mask | (osm_raster == 35)
        lod2_mask[_road_gravel & (lod2_heights < 6.0)] = 0

        # Kleine Infrastrukturgebäude (<4m) entlang der ORIGINALEN Wasserroute entfernen.
        # water_path_mask enthält die Route vor Priority-Overwrite durch Gravel/Straße,
        # damit Pumpwerke auch dann entfernt werden wenn das Wasser jetzt Gravel ist.
        if water_path_mask.any() and (lod2_mask == 1).any():
            small_flat = (lod2_mask == 1) & (lod2_heights < 4.0) & water_path_mask
            n_removed = int(small_flat.sum())
            if n_removed:
                lod2_mask[small_flat] = 0
                print(f"  Kleine Infrastrukturgebaeude auf Wasser entfernt: {n_removed} px")

        print_osm_stats(osm_raster)
    else:
        print("\n[5/7] OSM uebersprungen (--no-osm)")

    # 6. Vorschau
    print("\n[6/7] Vorschau...")
    save_preview(dgm, ndsm_cls, ndsm, osm_raster, lod2_mask)

    # 7. Welt schreiben
    print("\n[7/7] Minecraft-Welt schreiben...")
    H_px, W_px = dgm.shape
    auto_tiled = (H_px * W_px) > (1024 * 1024)
    if auto_tiled and not args.tiled:
        print(f"  (Auto-Tiled: {W_px}×{H_px} px > 1024×1024 — tiled wird automatisch aktiviert)")
    if args.tiled or auto_tiled:
        _write_tiled(dgm, ndsm_cls, ndsm,
                     osm_raster, lod2_mask, lod2_heights,
                     lod2_roof_mc, lod2_roof_type,
                     dgm_ox=dgm_ox, dgm_oy=dgm_oy, dgm_res=dgm_res,
                     buildings=buildings, alkis_buildings=alkis_buildings,
                     bridge_mask=bridge_mask, ortslage_mask=ortslage_mask,
                     tile_size=args.tile_size, n_workers=args.workers)
    else:
        write_minecraft_world(dgm, ndsm_cls, ndsm,
                              osm_raster, lod2_mask, lod2_heights,
                              lod2_roof_mc, lod2_roof_type,
                              dgm_ox=dgm_ox, dgm_oy=dgm_oy, dgm_res=dgm_res,
                              buildings=buildings,
                              alkis_buildings=alkis_buildings,
                              bridge_mask=bridge_mask,
                              ortslage_mask=ortslage_mask)

    print("\n✓ Fertig! 🎮")


if __name__ == "__main__":
    main()