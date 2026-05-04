"""
entity_writer.py – Minecraft 1.18+ Entity Region Writer
=========================================================
Schreibt Entities (Dorfbewohner) in die separaten Entity-Region-Dateien
unter dimensions/minecraft/overworld/entities/r.X.Z.mca

Dorfbewohner brauchen:
  - Bett (gesetzt als Block in der Welt)
  - Arbeitsstation für Beruf
  - Spawn-Position nahe einer Tür
"""

import nbtlib
import zlib
import struct
import math
import io
import os
import time
import random
from pathlib import Path
from collections import defaultdict

DATA_VERSION = 4671   # Minecraft 1.21.5

# Dorfbewohner-Berufe → Arbeitsstation
PROFESSIONS = [
    ("minecraft:farmer",      "minecraft:composter"),
    ("minecraft:librarian",   "minecraft:lectern"),
    ("minecraft:armorer",     "minecraft:blast_furnace"),
    ("minecraft:weaponsmith",  "minecraft:grindstone"),
    ("minecraft:toolsmith",   "minecraft:smithing_table"),
    ("minecraft:butcher",     "minecraft:smoker"),
    ("minecraft:cartographer","minecraft:cartography_table"),
    ("minecraft:cleric",      "minecraft:brewing_stand"),
    ("minecraft:fisherman",   "minecraft:barrel"),
    ("minecraft:fletcher",    "minecraft:fletching_table"),
    ("minecraft:leatherworker","minecraft:cauldron"),
    ("minecraft:shepherd",    "minecraft:loom"),
    ("minecraft:stonecutter", "minecraft:stonecutter"),
]

BED_COLORS = [
    "red", "blue", "green", "yellow", "white",
    "orange", "cyan", "purple", "brown", "light_blue",
]

CARPET_COLORS = [
    "red", "blue", "green", "yellow", "white",
    "orange", "cyan", "purple", "brown", "light_blue",
]

FLOWER_POTS = [
    "minecraft:potted_dandelion",
    "minecraft:potted_red_tulip",
    "minecraft:potted_blue_orchid",
    "minecraft:potted_fern",
    "minecraft:potted_cornflower",
    "minecraft:potted_oak_sapling",
    "minecraft:potted_cactus",
]


def _make_villager_nbt(x, y, z, profession=None, uid=None):
    """Erstellt ein vollständiges Dorfbewohner-NBT-Compound."""
    if profession is None:
        idx = abs(hash((x, z))) % len(PROFESSIONS)
        profession_id, _ = PROFESSIONS[idx]
    else:
        profession_id = profession

    # Eindeutige UUID (4 ints)
    if uid is None:
        rng = abs(hash((x, y, z, time.time())))
        uid = [
            (rng >> 0)  & 0x7FFFFFFF,
            (rng >> 31) & 0x7FFFFFFF,
            (rng >> 62) & 0x7FFFFFFF,
            (rng >> 93) & 0x7FFFFFFF,
        ]

    return nbtlib.Compound({
        "id":           nbtlib.String("minecraft:villager"),
        "UUID":         nbtlib.IntArray(uid),
        "Pos":          nbtlib.List[nbtlib.Double]([
                            nbtlib.Double(float(x) + 0.5),
                            nbtlib.Double(float(y)),
                            nbtlib.Double(float(z) + 0.5),
                        ]),
        "Motion":       nbtlib.List[nbtlib.Double]([
                            nbtlib.Double(0.0), nbtlib.Double(0.0), nbtlib.Double(0.0)
                        ]),
        "Rotation":     nbtlib.List[nbtlib.Float]([
                            nbtlib.Float(float(abs(hash((x,z))) % 360)),
                            nbtlib.Float(0.0),
                        ]),
        "Health":       nbtlib.Float(20.0),
        "FallDistance": nbtlib.Float(0.0),
        "Fire":         nbtlib.Short(-1),
        "Air":          nbtlib.Short(300),
        "OnGround":     nbtlib.Byte(1),
        "Invulnerable": nbtlib.Byte(0),
        "PortalCooldown": nbtlib.Int(0),
        "PersistenceRequired": nbtlib.Byte(1),   # nicht despawnen!
        "VillagerData": nbtlib.Compound({
            "type":       nbtlib.String("minecraft:plains"),
            "profession": nbtlib.String(profession_id),
            "level":      nbtlib.Int(1),
        }),
        "Inventory":    nbtlib.List[nbtlib.Compound]([]),
        "Gossips":      nbtlib.List[nbtlib.Compound]([]),
        "Offers":       nbtlib.Compound({
            "Recipes": nbtlib.List[nbtlib.Compound]([]),
        }),
        "SleepingX":    nbtlib.Int(x),
        "SleepingY":    nbtlib.Int(y),
        "SleepingZ":    nbtlib.Int(z),
        "Brain":        nbtlib.Compound({
            "memories": nbtlib.Compound({}),
        }),
        "FoodLevel":      nbtlib.Byte(0),
        "Xp":             nbtlib.Int(0),
        "LastRestock":    nbtlib.Long(0),
        "LastGossipDecay": nbtlib.Long(0),
        "RestocksToday":  nbtlib.Int(0),
    })


def _write_entity_region(rx, rz, entity_chunks, output_dir):
    """
    Schreibt eine Entity-Region-Datei.
    entity_chunks: { (cx, cz): [entity_nbt, ...] }
    """
    path = (Path(output_dir) / "dimensions" / "minecraft" /
            "overworld" / "entities" / f"r.{rx}.{rz}.mca")
    path.parent.mkdir(parents=True, exist_ok=True)

    payloads = {}
    for (cx, cz), entities in entity_chunks.items():
        lcx = cx & 31
        lcz = cz & 31

        chunk_nbt = nbtlib.Compound({
            "DataVersion": nbtlib.Int(DATA_VERSION),
            "Position":    nbtlib.IntArray([cx, cz]),
            "Entities":    nbtlib.List[nbtlib.Compound](entities),
        })

        buf = io.BytesIO()
        nbtlib.File(chunk_nbt).write(buf)
        compressed = zlib.compress(buf.getvalue(), level=1)
        raw = struct.pack(">IB", len(compressed) + 1, 2) + compressed
        n_secs = math.ceil(len(raw) / 4096)
        padded  = raw + b'\x00' * (n_secs * 4096 - len(raw))
        payloads[(lcx, lcz)] = (n_secs, padded)

    locations  = bytearray(4096)
    timestamps = bytearray(4096)
    body       = bytearray()
    cur_sector = 2

    for (lcx, lcz), (n_secs, padded) in payloads.items():
        idx = lcx + lcz * 32
        struct.pack_into(">I", locations,  idx * 4,
                         (cur_sector << 8) | (n_secs & 0xFF))
        struct.pack_into(">I", timestamps, idx * 4, int(time.time()))
        body += padded
        cur_sector += n_secs

    with open(path, "wb") as f:
        f.write(locations)
        f.write(timestamps)
        f.write(body)


class EntityWriter:
    """Sammelt Entities und schreibt Entity-Region-Dateien."""

    def __init__(self, output_dir, x_offset=0, z_offset=0,
                 tile_min_x=None, tile_max_x=None,
                 tile_min_z=None, tile_max_z=None):
        self.output_dir = output_dir
        self.x_offset   = x_offset
        self.z_offset   = z_offset
        self._xmin = tile_min_x
        self._xmax = tile_max_x
        self._zmin = tile_min_z
        self._zmax = tile_max_z
        # region → chunk → [entity_nbt]
        self._data = defaultdict(lambda: defaultdict(list))
        self._count = 0

    def _in_bounds(self, gx, gz):
        if self._xmin is not None and not (self._xmin <= gx <= self._xmax):
            return False
        if self._zmin is not None and not (self._zmin <= gz <= self._zmax):
            return False
        return True

    def add_entity(self, x, y, z, nbt):
        gx = x + self.x_offset
        gz = z + self.z_offset
        if not self._in_bounds(gx, gz):
            return
        cx = gx >> 4;  cz = gz >> 4
        rx = cx >> 5;  rz = cz >> 5
        self._data[(rx, rz)][(cx, cz)].append(nbt)
        self._count += 1

    def spawn_villager(self, x, y, z, profession=None):
        gx, gz = x + self.x_offset, z + self.z_offset
        if not self._in_bounds(gx, gz):
            return
        nbt = _make_villager_nbt(gx, y, gz, profession)
        cx = gx >> 4;  cz = gz >> 4
        rx = cx >> 5;  rz = cz >> 5
        self._data[(rx, rz)][(cx, cz)].append(nbt)
        self._count += 1

    def save(self):
        regions = list(self._data.items())
        print(f"  {self._count:,} Entities in {len(regions)} Entity-Regions")
        for (rx, rz), chunks in regions:
            _write_entity_region(rx, rz, chunks, self.output_dir)
        print(f"  ✓ Entity-Regions gespeichert")


# ─────────────────────────────────────────────
# Betten und Arbeitsstationen als Blöcke
# ─────────────────────────────────────────────

def get_bed_block(x, z):
    """Gibt einen Bett-Block in zufälliger Farbe zurück (deterministisch)."""
    color = BED_COLORS[abs(hash((x, z))) % len(BED_COLORS)]
    return f"minecraft:{color}_bed[part=head,facing=south]"

def get_bed_foot_block(x, z):
    color = BED_COLORS[abs(hash((x, z))) % len(BED_COLORS)]
    return f"minecraft:{color}_bed[part=foot,facing=south]"

def get_workstation_block(x, z):
    """Gibt eine Arbeitsstation passend zum Beruf zurück."""
    idx = abs(hash((x, z))) % len(PROFESSIONS)
    return PROFESSIONS[idx][1]


# ─────────────────────────────────────────────
# Bewohner für eine ganze Stadt spawnen
# ─────────────────────────────────────────────

def populate_city(writer_blocks, entity_writer,
                  lod2_mask, lod2_heights, lod2_roof_mc, door_mask,
                  smooth_floor, mc_y_grid, dgm_shape):
    """
    Setzt Betten + Arbeitsstationen in Gebäuden und spawnt Dorfbewohner.

    writer_blocks:  WorldWriter-Instanz (für Betten/Arbeitsstationen)
    entity_writer:  EntityWriter-Instanz (für Villager)
    lod2_mask:      uint8-Array, 1 = Gebäude
    smooth_floor:   int32-Array, Boden-Y pro Pixel
    door_mask:      bool-Array, True = Tür
    """
    from scipy.ndimage import label as ndlabel
    from scipy.ndimage import binary_erosion as _be
    import numpy as np

    H, W = dgm_shape
    building_bool = lod2_mask == 1

    # Innen-Pixel: erodiert (nicht Wand)
    from scipy.ndimage import binary_erosion
    struct   = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    interior = binary_erosion(building_bool, structure=struct, border_value=0)

    # Gebäude-Komponenten
    labeled, n_comp = ndlabel(building_bool)
    print(f"  {n_comp:,} Gebäude-Komponenten für Bewohner")

    n_villagers = 0
    n_beds      = 0
    n_ok = 0
    n_skip_area = 0
    n_skip_height = 0
    n_skip_dim = 0
    n_skip_interior = 0

    for comp_id in range(1, n_comp + 1):
        comp          = labeled == comp_id
        comp_interior = comp & interior
        comp_pixels   = np.argwhere(comp)
        area          = int(comp.sum())

        if area < 4: continue

        r0, c0   = comp_pixels[0]
        height_m = float(lod2_heights[r0, c0])

        rows_c = comp_pixels[:, 0]; cols_c = comp_pixels[:, 1]
        width  = int(cols_c.max()-cols_c.min()+1)
        depth  = int(rows_c.max()-rows_c.min()+1)
        min_dim = min(width, depth)
        max_dim = max(width, depth)

        if min_dim < 4 or (max_dim / max(min_dim, 1)) > 4:
            n_skip_dim += 1; continue
        if height_m < 4.0:
            n_skip_height += 1; continue
        if area < 16:
            n_skip_area += 1; continue

        interior_pixels = np.argwhere(comp_interior)
        if len(interior_pixels) < 4:
            n_skip_interior += 1; continue

        n_ok += 1

        # Bewohner nach Gebäudetyp und Fläche
        # Wohnhaus (~50m²): 1-2 Bewohner
        # Mehrfamilienhaus (~200m²): 3-6 Bewohner  
        # Büro/Schule (~500m²+): max 8
        if   area < 30:   n_res = 1
        elif area < 80:   n_res = 2
        elif area < 200:  n_res = 3
        elif area < 400:  n_res = 5
        else:             n_res = 8

        interior_pixels = np.argwhere(comp_interior)
        if len(interior_pixels) < 4:
            continue

        # Boden-Y: Median des echten Terrains dieser Gebäude-Komponente
        comp_floor_y = int(np.median(mc_y_grid[comp]))

        sample_r = int(interior_pixels[0][0])
        sample_c = int(interior_pixels[0][1])
        roof_y   = int(lod2_roof_mc[sample_r, sample_c])
        STORY_H  = 4
        n_stories = (roof_y - comp_floor_y) // STORY_H

        # Garagen / kleine Nebengebäude: einstöckig + kleine Fläche → kein Spawn
        if n_stories <= 1 and area < 25:
            n_skip_area += 1; continue

        # Wand-nahe Pixel: genau 1px von Wand entfernt → ideal für Betten
        wall_adjacent = comp_interior & ~(_be(comp_interior, structure=struct, border_value=0))
        wall_adj_pixels = np.argwhere(wall_adjacent & comp)
        bed_candidates = wall_adj_pixels if len(wall_adj_pixels) >= 4 else interior_pixels

        # Treppenhaus-Zone vorberechnen (damit Betten dort nicht spawnen)
        stair_zone = set()
        if n_stories >= 2:
            deep2 = _be(comp_interior, structure=struct, border_value=0) & comp
            deep2_arr = np.argwhere(deep2) if deep2.any() else interior_pixels
            rows2 = sorted(set(int(p[0]) for p in deep2_arr))
            for row2 in rows2:
                if row2 + 1 not in rows2:
                    continue
                cols2_r0 = sorted(int(p[1]) for p in deep2_arr if int(p[0]) == row2)
                cols2_r1 = sorted(int(p[1]) for p in deep2_arr if int(p[0]) == row2+1)
                common2 = sorted(set(cols2_r0) & set(cols2_r1))
                for i2 in range(len(common2) - 3):
                    if common2[i2+3] - common2[i2] == 3:
                        for c2 in range(common2[i2], common2[i2]+4):
                            stair_zone.add((row2, c2))
                            stair_zone.add((row2+1, c2))
                        break

        used = set()
        beds_placed = 0
        n_eff_stories = max(1, n_stories)
        for i in range(min(n_res * 10, len(bed_candidates))):
            if beds_placed >= n_res:
                break
            idx = (i * 7) % len(bed_candidates)
            r, c = int(bed_candidates[idx][0]), int(bed_candidates[idx][1])
            if (r, c) in used or door_mask[r, c] or (r, c) in stair_zone:
                continue
            near_door = door_mask[max(0,r-2):min(dgm_shape[0],r+3),
                                  max(0,c-2):min(dgm_shape[1],c+3)].any()
            if near_door:
                continue

            story  = beds_placed % n_eff_stories
            bed_y  = comp_floor_y + story * STORY_H + 1
            eff_roof_rc = max(int(lod2_roof_mc[r, c]), int(smooth_floor[r, c]) + STORY_H)
            if eff_roof_rc < comp_floor_y + (story + 1) * STORY_H:
                continue
            color  = BED_COLORS[abs(hash((c, r))) % len(BED_COLORS)]

            for facing, fr, fc in [("north", r+1, c), ("south", r-1, c),
                                    ("west",  r,   c+1), ("east", r,   c-1)]:
                if (0 <= fr < dgm_shape[0] and 0 <= fc < dgm_shape[1]
                        and comp_interior[fr, fc]
                        and not door_mask[fr, fc]
                        and (fr, fc) not in used
                        and (fr, fc) not in stair_zone
                        and max(int(lod2_roof_mc[fr, fc]), int(smooth_floor[fr, fc]) + STORY_H) >= comp_floor_y + (story + 1) * STORY_H):
                    writer_blocks.set_block(c,  bed_y, r,  f"minecraft:{color}_bed[part=head,facing={facing},occupied=false]")
                    writer_blocks.set_block(fc, bed_y, fr, f"minecraft:{color}_bed[part=foot,facing={facing},occupied=false]")
                    used.add((r, c)); used.add((fr, fc))
                    beds_placed += 1; n_beds += 1
                    break

        # Arbeitsstationen am Boden, in der Mitte des Raums
        if len(interior_pixels) > 0:
            center_r = int(np.median(interior_pixels[:, 0]))
            center_c = int(np.median(interior_pixels[:, 1]))
            best_ws = None; best_dist = 9999
            for ip_r, ip_c in interior_pixels:
                ir, ic = int(ip_r), int(ip_c)
                if (ir, ic) in used: continue
                if door_mask[max(0,ir-2):min(dgm_shape[0],ir+3),
                             max(0,ic-2):min(dgm_shape[1],ic+3)].any(): continue
                dist = abs(ir - center_r) + abs(ic - center_c)
                if dist < best_dist:
                    best_dist = dist; best_ws = (ir, ic)
            if best_ws:
                wr, wc = best_ws
                writer_blocks.set_block(wc, comp_floor_y + 1, wr, get_workstation_block(wc, wr))
                used.add((wr, wc))

        # Gemütlichkeit: Bücherregale, Teppiche, Blumentöpfe pro Stockwerk
        wa_px = wall_adj_pixels if len(wall_adj_pixels) >= 2 else interior_pixels
        for story in range(n_eff_stories):
            story_y = comp_floor_y + story * STORY_H
            story_min_roof = comp_floor_y + (story + 1) * STORY_H
            # Bücherregal an Wand (y = Boden+1), Topf obendrauf (y = Boden+2)
            for k in range(len(wa_px)):
                wi = (k * 13 + story * 7) % len(wa_px)
                wr2, wc2 = int(wa_px[wi][0]), int(wa_px[wi][1])
                near_door = door_mask[max(0,wr2-1):min(dgm_shape[0],wr2+2),
                                      max(0,wc2-1):min(dgm_shape[1],wc2+2)].any()
                if ((wr2, wc2) not in used and (wr2, wc2) not in stair_zone
                        and not near_door
                        and max(int(lod2_roof_mc[wr2, wc2]), int(smooth_floor[wr2, wc2]) + STORY_H) >= story_min_roof):
                    writer_blocks.set_block(wc2, story_y + 1, wr2, "minecraft:bookshelf")
                    pot = FLOWER_POTS[abs(hash((wc2, wr2))) % len(FLOWER_POTS)]
                    writer_blocks.set_block(wc2, story_y + 2, wr2, pot)
                    used.add((wr2, wc2)); break
            # Teppich: zentriert, 3×3 (nur Interior-Pixel, keine Möbelkonflikte)
            ip_set = {(int(p[0]), int(p[1])) for p in interior_pixels}
            cr = int(np.median(interior_pixels[:, 0]))
            cc = int(np.median(interior_pixels[:, 1]))
            carpet = CARPET_COLORS[abs(hash((comp_id, story))) % len(CARPET_COLORS)]
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    pr, pc = cr + dr, cc + dc
                    if ((pr, pc) in ip_set and (pr, pc) not in stair_zone
                            and max(int(lod2_roof_mc[pr, pc]), int(smooth_floor[pr, pc]) + STORY_H) >= story_min_roof):
                        writer_blocks.set_block(pc, story_y + 1, pr, f"minecraft:{carpet}_carpet")

        # Villager stockwerksweise spawnen
        spawn_pixels = interior_pixels
        for i in range(n_res):
            sp_idx  = (i * 5) % len(spawn_pixels)
            sr, sc  = spawn_pixels[sp_idx]
            story   = i % n_eff_stories
            sr_i, sc_i = int(sr), int(sc)
            if max(int(lod2_roof_mc[sr_i, sc_i]), int(smooth_floor[sr_i, sc_i]) + STORY_H) < comp_floor_y + (story + 1) * STORY_H:
                continue
            vill_y  = comp_floor_y + story * STORY_H + 1
            entity_writer.spawn_villager(int(sc), vill_y, int(sr))
            n_villagers += 1

        # Treppenhäuser: alle ~50px ein neues, bevorzugt an Türen
        if n_stories >= 2:
            deep = _be(comp_interior, structure=struct, border_value=0) & comp
            deep_arr = np.argwhere(deep) if deep.any() else interior_pixels

            # Kandidaten-Positionen sammeln
            stair_candidates = []
            rows_available = sorted(set(int(p[0]) for p in deep_arr))
            for row_r in rows_available:
                if row_r + 1 not in rows_available:
                    continue
                cols_r0 = sorted(int(p[1]) for p in deep_arr if int(p[0]) == row_r)
                cols_r1 = sorted(int(p[1]) for p in deep_arr if int(p[0]) == row_r + 1)
                common = sorted(set(cols_r0) & set(cols_r1))
                for i in range(len(common) - 3):
                    if common[i+3] - common[i] == 3:
                        stair_candidates.append((row_r, common[i]))

            # Treppenhäuser alle 50px platzieren
            placed_stairs = []  # (r, c) schon gesetzt
            for stair_r, stair_c in stair_candidates:
                # Mindestabstand 50px zu schon gesetzten Treppenhäusern
                too_close = any(abs(stair_r - pr) + abs(stair_c - pc) < 50
                                for pr, pc in placed_stairs)
                if too_close:
                    continue
                placed_stairs.append((stair_r, stair_c))

                for story in range(n_stories - 1):
                    y_lower = comp_floor_y + story * STORY_H
                    y_upper = comp_floor_y + (story + 1) * STORY_H
                    row = stair_r + (story % 2)
                    going_east = (story % 2 == 0)
                    facing     = "east" if going_east else "west"
                    facing_dec = "west" if going_east else "east"

                    top_tc = None
                    for step in range(STORY_H):
                        tc = stair_c + step if going_east else stair_c + STORY_H - 1 - step
                        sy = y_lower + 1 + step
                        if not (0 <= tc < dgm_shape[1]): continue
                        if step == STORY_H - 1:
                            top_tc = tc
                        if step >= 1:
                            writer_blocks.delete_block(tc, sy - 1, row)
                            writer_blocks.set_block(tc, sy - 1, row,
                                f"minecraft:oak_stairs[facing={facing_dec},half=top,shape=straight]")
                        writer_blocks.delete_block(tc, sy, row)
                        writer_blocks.set_block(tc, sy, row,
                            f"minecraft:oak_stairs[facing={facing},half=bottom,shape=straight]")
                        for dy in [1, 2, 3]:
                            writer_blocks.delete_block(tc, sy + dy, row)
                        stair_zone.add((row, tc))

                    for tc in range(stair_c, stair_c + STORY_H):
                        if tc == top_tc:
                            continue
                        writer_blocks.delete_block(tc, y_upper,     row)
                        writer_blocks.delete_block(tc, y_upper + 1, row)
                        writer_blocks.delete_block(tc, y_upper + 2, row)

        # Licht: hängende Laternen nur wo ein Stockwerksboden darüber ist
        if comp_floor_y < roof_y <= comp_floor_y + 40:
            for story in range(n_stories):
                floor_above = comp_floor_y + (story + 1) * STORY_H
                lamp_y = floor_above - 1
                if floor_above > roof_y or lamp_y <= comp_floor_y:
                    break  # kein echter Boden darüber
                for ip_r, ip_c in interior_pixels:
                    ir, ic = int(ip_r), int(ip_c)
                    # Kein Spawn im Treppenhaus
                    if (ir, ic) in stair_zone:
                        continue
                    px_roof = int(lod2_roof_mc[ir, ic])
                    if px_roof < floor_above + 1:
                        continue
                    if (ir * 3 + ic * 5) % 64 == 0:
                        writer_blocks.set_block(ic, lamp_y, ir,
                            "minecraft:lantern[hanging=true]")

    print(f"  Filter: ok={n_ok} dim={n_skip_dim} height={n_skip_height} area={n_skip_area} interior={n_skip_interior}")
    print(f"  Bewohner: {n_villagers:,}  |  Betten: {n_beds:,}")
    return n_villagers
