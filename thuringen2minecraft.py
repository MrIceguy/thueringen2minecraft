"""
ilmenau_to_minecraft.py  (v4)
==============================
Ordnerstruktur erwartet:
    dgm/    *.tif  oder  *.xyz   <- DGM-Kacheln
    dom/    *.tif  oder  *.xyz   <- DOM-Kacheln
    LoD2/   *.gml               <- 3D-Gebaeude

Aufruf:
    python ilmenau_to_minecraft.py               # alle Kacheln
    python ilmenau_to_minecraft.py --list        # Kacheln anzeigen
    python ilmenau_to_minecraft.py --bbox 634000 636000 5614000 5616000

Installation:
    pip install rasterio numpy scipy matplotlib anvil-parser osmnx pyproj shapely lxml
"""

import numpy as np
import argparse
import os
from pathlib import Path
from rasterio.warp import reproject, Resampling as RS
from scipy.ndimage import uniform_filter, label as ndlabel

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
    get_osm_block, is_water, print_osm_stats, OSM_CLASSES,
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
REAL_MIN_H = 450.0
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
    lmax = uniform_filter(ndsm,  size=5, mode="reflect")
    lmin = uniform_filter(-ndsm, size=5, mode="reflect")
    rough    = lmax + lmin
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


def _place_tree(writer, bx, bz, sy, obj_h, tree_type=None):
    h       = max(3, min(int(obj_h), 22))
    trunk_h = max(1, h - 4)

    if tree_type is None:
        seed = (bx * 31 + bz * 17) % 4
        tree_type = ["oak", "spruce", "birch", "dark_oak"][seed]

    log    = f"minecraft:{tree_type}_log"
    leaves = f"minecraft:{tree_type}_leaves[persistent=true]"

    cy = sy + trunk_h + 1

    # Blaetter zuerst
    if tree_type == "spruce":
        for dy in range(-1, h - trunk_h + 1):
            r = max(0, 2 - max(0, dy))
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) + abs(dz) <= r + 1:
                        writer.set_block(bx + dx, cy + dy, bz + dz, leaves)
    else:
        for dy in range(-1, 3):
            r = 2 if dy <= 0 else 1
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) <= r and abs(dz) <= r:
                        writer.set_block(bx + dx, cy + dy, bz + dz, leaves)

    # Stamm bis zum Boden (sy+1 bis sy+trunk_h)
    for dy in range(trunk_h + 1):
        writer.set_block(bx, sy + dy, bz, log)


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
                          lod2_roof_mc, lod2_roof_type):
    from anvil_writer import WorldWriter, write_level_dat
    from scipy.ndimage import maximum_filter, binary_erosion, distance_transform_edt, label as ndlabel

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    H, W = dgm.shape
    print(f"  {W} x {H} Bloecke")

    writer    = WorldWriter(OUTPUT_DIR)
    prog_step = max(1, H // 40)

    print("  Berechne glatte Gebaeude-Basis...")
    mc_y_grid   = height_to_mc_y(dgm)
    smooth_base = maximum_filter(mc_y_grid, size=3, mode='nearest')  # fuer Dach

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

    print("  Berechne Gebaeude-Innen/Aussen...")
    struct        = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    building_bool = lod2_mask == 1
    eroded        = binary_erosion(building_bool, structure=struct, border_value=0)
    wall_mask     = building_bool & ~eroded

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

    # Baum-Positionen sammeln (nach Terrain setzen, damit Gras nicht ueberschreibt)
    tree_queue = []
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

                    # Wandfarbe fuer Tuerrahmen (passend zur Hauswand)
                    h_est = top_y - floor_y
                    if   h_est < 4:  frame_b = "minecraft:stone_bricks"
                    elif h_est < 10: frame_b = "minecraft:white_concrete"
                    elif h_est < 20: frame_b = "minecraft:light_gray_concrete"
                    else:            frame_b = "minecraft:quartz_block"

                    perp = [(dr[1], dr[0]), (-dr[1], -dr[0])]
                    for pr, pc in perp:
                        sr, sc = row+pr, col+pc
                        if 0 <= sr < H and 0 <= sc < W and not building_bool[sr, sc]:
                            ext_frame = int(mc_y_grid[sr, sc])
                            # Nach unten bis Terrain auffuellen
                            for fy in range(ext_frame, floor_y):
                                writer.set_block(int(sc), fy, int(sr), frame_b)
                            # Rahmen 4 Bloecke hoch
                            for fy in range(floor_y, floor_y + 4):
                                writer.set_block(int(sc), fy, int(sr), frame_b)

                    flip = {"north":"south","south":"north","west":"east","east":"west"}
                    stair_face_up   = flip[facing]  # terrain tiefer (runter): zum Haus zeigend
                    stair_face_down = facing         # terrain hoeher (hoch): weg vom Haus zeigend

                    nr1 = row + dr[0]; nc1 = col + dr[1]
                    if 0 <= nr1 < H and 0 <= nc1 < W and not building_bool[nr1, nc1]:
                        ext_sy = int(mc_y_grid[nr1, nc1])

                        if ext_sy > floor_y:
                            # Terrain hoeher: loeschen, dann pruefen ob Treppe noetig
                            door_clear_list.append((int(nc1), ext_sy, int(nr1)))
                            post_ramp_list.append(("stone", int(nc1), floor_y, int(nr1)))
                            nr2 = row + dr[0]*2; nc2 = col + dr[1]*2
                            if 0 <= nr2 < H and 0 <= nc2 < W and not building_bool[nr2, nc2]:
                                ext_sy2 = int(mc_y_grid[nr2, nc2])
                                if ext_sy2 > floor_y:
                                    # Zweiter Block auch erhoeht: Treppe setzen
                                    for fy in range(floor_y + 2, ext_sy2 + 1):
                                        door_clear_list.append((int(nc2), fy, int(nr2)))
                                    post_ramp_list.append(("stone", int(nc2), floor_y, int(nr2)))
                                    post_ramp_list.append(("stair_down", int(nc2), floor_y + 1, int(nr2), stair_face_down))
                                # else: ebenerdig nach erstem Block, keine Treppe noetig

                        elif ext_sy <= floor_y - 2:
                            # Terrain 2+ Bloecke tiefer: Treppe RUNTER
                            diff = floor_y - ext_sy
                            for step in range(min(diff, 3)):
                                snr = row + dr[0]*(step+1)
                                snc = col + dr[1]*(step+1)
                                if not (0 <= snr < H and 0 <= snc < W) or building_bool[snr, snc]:
                                    break
                                stair_y = floor_y - step
                                s_ext = int(mc_y_grid[snr, snc])
                                writer.set_block(int(snc), s_ext, int(snr), "minecraft:stone_bricks")
                                for fy in range(s_ext + 1, stair_y):
                                    writer.set_block(int(snc), fy, int(snr), "minecraft:stone_bricks")
                                writer.set_block(int(snc), stair_y, int(snr),
                                    f"minecraft:stone_brick_stairs[facing={stair_face_up},half=bottom,shape=straight]")
                                door_clear_list.append((int(snc), stair_y + 1, int(snr)))
                        # diff 0 oder -1: ebenerdig, nichts noetig

                _place_building_pixel(writer, bx, bz, floor_y, top_y,
                                      is_wall=bool(wall_mask[row, col]),
                                      roof_type=r_type,
                                      is_door=is_door,
                                      door_facing=facing)

            elif osm_val > 0 and not is_water(osm_val):
                osm_block = get_osm_block(osm_val)
                if osm_block:
                    if osm_block == "minecraft:rail":
                        writer.set_block(bx, sy,     bz, "minecraft:gravel")
                        writer.set_block(bx, sy + 1, bz, "minecraft:rail")
                    else:
                        writer.set_block(bx, sy, bz, osm_block)

                # Baeume auf Wald/Gruenland-Pixeln — nicht nah an Gebaeuden

                # Baeume auf Wald/Gruenland-Pixeln — nicht nah an Gebaeuden
                if osm_val in (30, 31):
                    h = float(ndsm[row, col])
                    if h > 2.0 and not building_bool[row, col]:
                        near_building = building_bool[
                            max(0,row-4):min(H,row+5),
                            max(0,col-4):min(W,col+5)
                        ].any()
                        if not near_building:
                            seed = (row * 7 + col * 13) % 10
                            threshold = 2 if osm_val == 31 else 1
                            if seed < threshold:
                                too_close = any(
                                    abs(tx - col) <= 5 and abs(tz - row) <= 5
                                    for tx, tz, *_ in tree_queue[-100:]
                                )
                                if not too_close:
                                    ttype = "spruce" if (row + col) % 3 == 0 else "oak"
                                    if osm_val == 30:
                                        ttype = "birch"
                                    tree_queue.append((bx, bz, sy, h, ttype))

                # Deko auch auf ATKIS-Flaechen
                _place_decoration(writer, bx, bz, sy, row, col,
                                  osm_val, osm_raster, lod2_mask, H, W)

            elif osm_val > 0 and is_water(osm_val) and not (11 <= osm_val <= 20):
                writer.set_block(bx, sy - 1, bz, "minecraft:gravel")
                writer.set_block(bx, sy,     bz, "minecraft:water")

            else:
                # Natuerliche Oberflaeche
                if   real_h > 780: surf = "minecraft:snow_block"
                elif real_h > 700: surf = "minecraft:stone"
                else:              surf = "minecraft:grass_block"
                writer.set_block(bx, sy, bz, surf)

                h = float(ndsm[row, col])
                if h > 2.0:
                    seed = (row * 5 + col * 19) % 10
                    if seed < 3:
                        tree_queue.append((bx, bz, sy, h, None))

                # Dekoration
                _place_decoration(writer, bx, bz, sy, row, col,
                                  osm_val, osm_raster, lod2_mask, H, W)

    # Baeume NACH dem Terrain setzen
    # Waende zwischen Gebaeuden: vom hoeheren Dach bis zum niedrigeren Dach
    for r in range(H):
        for c in range(W):
            if not building_bool[r, c]: continue
            comp_id = labeled_bldg[r, c]
            this_top = int(lod2_roof_mc[r, c])
            if this_top == 0: continue  # kein gueltiger Dach-Wert
            this_floor = int(smooth_floor[r, c])
            h_est = this_top - this_floor
            if   h_est < 4:  wall_col = "minecraft:stone_bricks"
            elif h_est < 10: wall_col = "minecraft:white_concrete"
            elif h_est < 20: wall_col = "minecraft:light_gray_concrete"
            else:             wall_col = "minecraft:quartz_block"
            for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr2, nc2 = r+dr2, c+dc2
                if not (0 <= nr2 < H and 0 <= nc2 < W): continue
                nb_comp = labeled_bldg[nr2, nc2]
                if nb_comp == comp_id or nb_comp == 0: continue
                nb_top = int(lod2_roof_mc[nr2, nc2])
                if nb_top == 0 or this_top <= nb_top: continue
                diff = this_top - nb_top
                if diff > 50: continue  # sanity check
                for fy in range(nb_top, this_top + 1):
                    writer.set_block(c, fy, r, wall_col)
    # Erst: Bloecke vor Tueren nachtraeglich loeschen (Terrain ueberschreibt sonst die Luft)
    for cx, cy, cz in door_clear_list:
        writer.delete_block(cx, cy,     cz)
        writer.delete_block(cx, cy + 1, cz)
        writer.delete_block(cx, cy + 2, cz)
    for item in post_ramp_list:
        if item[0] == "stone":
            _, x, y, z = item
            writer.set_block(x, y, z, "minecraft:stone_bricks")
        elif item[0] == "stair_down":
            _, x, y, z, face = item
            writer.set_block(x, y, z,
                f"minecraft:stone_brick_stairs[facing={face},half=bottom,shape=straight]")
    print(f"\n  Setze {len(tree_queue):,} Baeume...")
    for bx, bz, sy, h, ttype in tree_queue:
        _place_tree(writer, bx, bz, sy, h, ttype)

    print("\n  Bevoelkere die Stadt...")
    entity_writer = EntityWriter(OUTPUT_DIR)
    populate_city(writer, entity_writer,
                  lod2_mask, lod2_heights, lod2_roof_mc, door_mask,
                  smooth_floor, mc_y_grid, dgm.shape)
    entity_writer.save()

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
                      osm_val, osm_raster, lod2_mask, H, W):
    seed  = (row * 17 + col * 31) % 100
    seed2 = (row * 53 + col * 7)  % 10

    # Keine Deko auf Strassen/Wegen/Bahn (ATKIS-Codes 10-20)
    if 10 <= osm_val <= 20:
        if seed == 0:  # Laterne alle ~100m
            writer.set_block(bx, sy + 1, bz, "minecraft:oak_fence")
            writer.set_block(bx, sy + 2, bz, "minecraft:lantern")
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

    # Blumen nah an Gebaeuden (Vorgaerten) — haeufiger
    if near_building and seed < 15:
        writer.set_block(bx, sy + 1, bz, FLOWERS[seed % len(FLOWERS)])
        return

    # Blumen auf Wiesen — moderat
    if not near_building and seed < 8:
        writer.set_block(bx, sy + 1, bz, FLOWERS[seed % len(FLOWERS)])
        return

    # Gras/Farn auf Wiesen
    if not near_building and 8 <= seed < 25:
        writer.set_block(bx, sy + 1, bz,
                         "minecraft:grass" if seed2 < 7 else "minecraft:fern")
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

    # Laternen auf Strassen alle ~12m
    if osm_val in (11, 12, 13) and seed == 0:
        writer.set_block(bx, sy + 1, bz, "minecraft:oak_fence")
        writer.set_block(bx, sy + 2, bz, "minecraft:lantern")


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
                          roof_type=1000, is_door=False, door_facing="south"):
    h_blocks = min(max(3, top_y - floor_y), 80)

    if   h_blocks < 4:  wall_b, roof_b = "minecraft:stone_bricks",       "minecraft:stone_brick_slab"
    elif h_blocks < 10: wall_b, roof_b = "minecraft:white_concrete",      "minecraft:gray_concrete"
    elif h_blocks < 20: wall_b, roof_b = "minecraft:light_gray_concrete", "minecraft:gray_concrete"
    else:               wall_b, roof_b = "minecraft:quartz_block",        "minecraft:smooth_quartz"

    FLOOR_B     = "minecraft:oak_planks"
    RED_ROOF    = "minecraft:red_concrete"
    has_pitched = roof_type in (2100, 3100, 3200, 4000)
    STORY_H     = 4  # 1 Boden + 3 Wandbloecke

    # Horizontale Fensterperiode
    h_slot = (bx + bz * 2) % 3

    for fill_y in range(floor_y - 2, floor_y):
        writer.set_block(bx, fill_y, bz, "minecraft:stone")

    if is_wall:
        for dy in range(h_blocks):
            mc_y = floor_y + dy

            # Dachkante
            if dy == h_blocks - 1:
                # Unterste Dachreihe = Wandfarbe, darueber rot
                writer.set_block(bx, mc_y, bz, wall_b if has_pitched else roof_b)

            # Stockwerksboden (dy=0, 4, 8...) — immer Wand aussen
            elif dy % STORY_H == 0:
                writer.set_block(bx, mc_y, bz, wall_b)

            # Tuer
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
                # Fenster in Stockwerksmitte (dy%STORY_H == 2)
                v_win = (dy % STORY_H == 2)
                h_win = (h_slot < 2)
                writer.set_block(bx, mc_y, bz,
                                 "minecraft:glass" if (v_win and h_win) else wall_b)

        if has_pitched:
            writer.set_block(bx, floor_y + h_blocks, bz, RED_ROOF)

    else:
        for story in range(h_blocks // STORY_H + 1):
            story_y = floor_y + story * STORY_H
            if story_y < floor_y + h_blocks - 1:
                writer.set_block(bx, story_y, bz, FLOOR_B)
        # Unterste Dachreihe = Wandfarbe, darueber rot
        writer.set_block(bx, floor_y + h_blocks - 1, bz,
                         wall_b if has_pitched else roof_b)
        if has_pitched:
            writer.set_block(bx, floor_y + h_blocks, bz, RED_ROOF)


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
    ndsm_cls = np.zeros(dgm.shape, dtype=np.uint8)   # Gebaeude kommen aus LoD2

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
        buildings = load_all_lod2(args.lod2_dir, bbox=actual_bbox)
        if buildings:
            lod2_mask, lod2_heights, lod2_roof_mc, lod2_roof_type = rasterize_buildings(
                buildings, actual_bbox, dgm.shape)
    else:
        print("\n[4/7] LoD2 uebersprungen")

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
        print_osm_stats(osm_raster)
    else:
        print("\n[5/7] OSM uebersprungen (--no-osm)")

    # 6. Vorschau
    print("\n[6/7] Vorschau...")
    save_preview(dgm, ndsm_cls, ndsm, osm_raster, lod2_mask)

    # 7. Welt schreiben
    print("\n[7/7] Minecraft-Welt schreiben...")
    write_minecraft_world(dgm, ndsm_cls, ndsm,
                          osm_raster, lod2_mask, lod2_heights,
                          lod2_roof_mc, lod2_roof_type)

    print("\n✓ Fertig! 🎮")


if __name__ == "__main__":
    main()
