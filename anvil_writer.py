"""
anvil_writer.py – Minecraft Anvil Format Writer (Java 1.18+)
=============================================================
Korrekte Block-Reihenfolge: Y→Z→X (Minecraft-Standard)
Kompatibel mit Java Edition 1.18–1.21 (DataVersion 3700 = 1.20.4)
"""

import nbtlib
import zlib
import struct
import math
import os
import io
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_VERSION = 4671   # Minecraft 1.21.5 (26.1)
AIR          = "minecraft:air"

# Section Y-Bereich (1.18+: Y -64..319, Sections -4..19)
SEC_MIN = -4
SEC_MAX = 19


# ─────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────

def _block_index(lx, ly, lz):
    """Block-Index im Section-Array: Reihenfolge Y→Z→X."""
    return (ly * 16 + lz) * 16 + lx


def _bits_needed(n):
    """Bits pro Eintrag für n Palette-Einträge (mind. 4)."""
    if n <= 1:
        return 0
    return max(4, math.ceil(math.log2(n)))


def _pack_states(indices, bits):
    """Packt 4096 Block-Indizes in ein LongArray."""
    if bits == 0:
        return nbtlib.LongArray([])

    values_per_long = 64 // bits
    n_longs = math.ceil(4096 / values_per_long)
    longs = [0] * n_longs

    for i, idx in enumerate(indices):
        long_i   = i // values_per_long
        bit_off  = (i % values_per_long) * bits
        longs[long_i] |= (int(idx) & ((1 << bits) - 1)) << bit_off

    # unsigned → signed int64
    result = []
    for v in longs:
        if v >= (1 << 63):
            v -= (1 << 64)
        result.append(v)
    return nbtlib.LongArray(result)


# ─────────────────────────────────────────────
# Section bauen
# ─────────────────────────────────────────────

def _parse_block(block_str):
    """
    Parst 'minecraft:oak_leaves[persistent=true]'
    → ("minecraft:oak_leaves", {"persistent": "true"})
    """
    if "[" in block_str:
        name, props_str = block_str.rstrip("]").split("[", 1)
        props = {}
        for pair in props_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                props[k.strip()] = v.strip()
        return name, props
    return block_str, {}


def _make_section(sy, block_map):
    """
    block_map: dict { (lx, ly, lz): "minecraft:block_name[prop=val,...]" }
    """
    palette   = [AIR]
    name_to_i = {AIR: 0}

    indices = []
    for ly in range(16):
        for lz in range(16):
            for lx in range(16):
                name = block_map.get((lx, ly, lz), AIR)
                if name not in name_to_i:
                    name_to_i[name] = len(palette)
                    palette.append(name)
                indices.append(name_to_i[name])

    bits = _bits_needed(len(palette))

    # Palette mit Properties aufbauen
    palette_nbt_list = []
    for entry in palette:
        block_name, props = _parse_block(entry)
        compound = nbtlib.Compound({"Name": nbtlib.String(block_name)})
        if props:
            compound["Properties"] = nbtlib.Compound({
                k: nbtlib.String(v) for k, v in props.items()
            })
        palette_nbt_list.append(compound)

    palette_nbt = nbtlib.List[nbtlib.Compound](palette_nbt_list)

    block_states = nbtlib.Compound({"palette": palette_nbt})
    if bits > 0:
        block_states["data"] = _pack_states(indices, bits)

    return nbtlib.Compound({
        "Y":           nbtlib.Byte(sy),
        "block_states": block_states,
        "biomes": nbtlib.Compound({
            "palette": nbtlib.List[nbtlib.String](
                [nbtlib.String("minecraft:plains")]
            ),
        }),
    })


# ─────────────────────────────────────────────
# Chunk-NBT bauen
# ─────────────────────────────────────────────

def _make_heightmap(sections_data):
    """
    WORLD_SURFACE: höchster nicht-Luft-Block + 1, als Offset von Y=-64.
    Also: Block bei Y=64 → Heightmap-Wert = 64 - (-64) + 1 = 129
    Kodiert als 256 × 9-Bit-Werte in LongArray (Y→Z→X Reihenfolge).
    """
    # Initialisierung: 0 = kein Block gefunden
    heights = [0] * 256

    for sy, bmap in sections_data.items():
        # Unterste absolute Y-Koordinate dieser Section
        base_y = (sy + 4) * 16 - 64   # sy=-4 → base_y=-64, sy=0 → base_y=0
        for (lx, ly, lz), name in bmap.items():
            if name == AIR:
                continue
            abs_y   = base_y + ly          # absolute Y-Koordinate des Blocks
            hm_val  = abs_y - (-64) + 1   # Offset von Y=-64, +1 weil "oben drauf"
            idx     = lz * 16 + lx
            if hm_val > heights[idx]:
                heights[idx] = hm_val

    # 9 Bit pro Wert packen
    bits          = 9
    vals_per_long = 64 // bits            # = 7
    n_longs       = math.ceil(256 / vals_per_long)
    longs = [0] * n_longs
    for i, h in enumerate(heights):
        li  = i // vals_per_long
        off = (i % vals_per_long) * bits
        longs[li] |= (int(h) & 0x1FF) << off

    result = []
    for v in longs:
        if v >= (1 << 63):
            v -= (1 << 64)
        result.append(v)
    return nbtlib.LongArray(result)


def _make_chunk(cx, cz, sections_data, block_entities=None):
    """
    sections_data: dict { sy: { (lx,ly,lz): block_name } }
    block_entities: list of nbtlib.Compound
    """
    sections_nbt = nbtlib.List[nbtlib.Compound]()

    for sy in range(SEC_MIN, SEC_MAX + 1):
        bmap = sections_data.get(sy, {})
        sections_nbt.append(_make_section(sy, bmap))

    hm = _make_heightmap(sections_data)

    be_list = nbtlib.List[nbtlib.Compound](block_entities or [])

    return nbtlib.Compound({
        "DataVersion":    nbtlib.Int(DATA_VERSION),
        "xPos":           nbtlib.Int(cx),
        "yPos":           nbtlib.Int(SEC_MIN),
        "zPos":           nbtlib.Int(cz),
        "Status":         nbtlib.String("minecraft:full"),
        "LastUpdate":     nbtlib.Long(int(time.time())),
        "sections":       sections_nbt,
        "Heightmaps":     nbtlib.Compound({
            "WORLD_SURFACE":             hm,
            "OCEAN_FLOOR":               hm,
            "MOTION_BLOCKING":           hm,
            "MOTION_BLOCKING_NO_LEAVES": hm,
        }),
        "block_entities": be_list,
        "fluid_ticks":    nbtlib.List[nbtlib.Compound](),
        "block_ticks":    nbtlib.List[nbtlib.Compound](),
        "PostProcessing": nbtlib.List[nbtlib.List](),
        "structures":     nbtlib.Compound({
            "References": nbtlib.Compound({}),
            "starts":     nbtlib.Compound({}),
        }),
        "InhabitedTime":  nbtlib.Long(0),
        "isLightOn":      nbtlib.Byte(0),
    })


# ─────────────────────────────────────────────
# Region (.mca) schreiben
# ─────────────────────────────────────────────

def _write_region(rx, rz, chunks, output_dir):
    """
    chunks: dict { (cx, cz): nbt_compound }
    """
    path = Path(output_dir) / "dimensions" / "minecraft" / "overworld" / "region" / f"r.{rx}.{rz}.mca"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Chunk-Payloads vorbereiten
    payloads = {}
    for (cx, cz), nbt in chunks.items():
        lcx = cx & 31
        lcz = cz & 31

        buf = io.BytesIO()
        nbtlib.File(nbt).write(buf)
        compressed = zlib.compress(buf.getvalue(), level=1)
        # 4-Byte-Länge + 1-Byte-Kompressionstyp (2=zlib) + Daten
        raw = struct.pack(">IB", len(compressed) + 1, 2) + compressed
        # Auf 4096-Byte-Sektoren auffüllen
        n_secs = math.ceil(len(raw) / 4096)
        padded  = raw + b'\x00' * (n_secs * 4096 - len(raw))
        payloads[(lcx, lcz)] = (n_secs, padded)

    # Header-Tabellen
    locations  = bytearray(4096)   # 1024 × 4 Bytes
    timestamps = bytearray(4096)   # 1024 × 4 Bytes
    body       = bytearray()

    current_sector = 2   # Sektoren 0+1 = Header
    for (lcx, lcz), (n_secs, padded) in payloads.items():
        idx = lcx + lcz * 32
        # Location: 3-Byte-Offset + 1-Byte-Sektorzahl
        struct.pack_into(">I", locations,  idx * 4,
                         (current_sector << 8) | (n_secs & 0xFF))
        struct.pack_into(">I", timestamps, idx * 4, int(time.time()))
        body += padded
        current_sector += n_secs

    with open(path, "wb") as f:
        f.write(locations)
        f.write(timestamps)
        f.write(body)


# ─────────────────────────────────────────────
# level.dat schreiben
# ─────────────────────────────────────────────

def write_level_dat(output_dir, world_name="Thueringen2Minecraft",
                    spawn_x=1500, spawn_y=200, spawn_z=1500):
    overworld_dim = nbtlib.Compound({
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
    })

    data = nbtlib.Compound({
        "Data": nbtlib.Compound({
            "version":          nbtlib.Int(19133),
            "DataVersion":      nbtlib.Int(DATA_VERSION),
            "LevelName":        nbtlib.String(world_name),
            "SpawnX":           nbtlib.Int(spawn_x),
            "SpawnY":           nbtlib.Int(spawn_y),
            "SpawnZ":           nbtlib.Int(spawn_z),
            "GameType":         nbtlib.Int(1),
            "hardcore":         nbtlib.Byte(0),
            "Difficulty":       nbtlib.Byte(0),
            "allowCommands":    nbtlib.Byte(1),
            "initialized":      nbtlib.Byte(1),
            "LastPlayed":       nbtlib.Long(int(time.time() * 1000)),
            "Time":             nbtlib.Long(6000),
            "DayTime":          nbtlib.Long(6000),
            "raining":          nbtlib.Byte(0),
            "thundering":       nbtlib.Byte(0),
            "clearWeatherTime": nbtlib.Int(100000),
            "WorldGenSettings": nbtlib.Compound({
                "bonus_chest":       nbtlib.Byte(0),
                "generate_features": nbtlib.Byte(0),
                "seed":              nbtlib.Long(0),
                "dimensions": nbtlib.Compound({
                    "minecraft:overworld": overworld_dim,
                }),
            }),
            "Version": nbtlib.Compound({
                "Id":       nbtlib.Int(DATA_VERSION),
                "Name":     nbtlib.String("1.21.5"),
                "Series":   nbtlib.String("main"),
                "Snapshot": nbtlib.Byte(0),
            }),
        })
    })

    os.makedirs(output_dir, exist_ok=True)
    nbtlib.File(data).save(
        os.path.join(output_dir, "level.dat"), gzipped=True
    )
    print(f"  level.dat → {output_dir}/level.dat")


# ─────────────────────────────────────────────
# WorldWriter
# ─────────────────────────────────────────────

class WorldWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._data     = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict)
            )
        )
        # Block entities: region → chunk → list of compounds
        self._block_entities = defaultdict(lambda: defaultdict(list))
        self._n_blocks = 0

    def set_block(self, x, y, z, block_name):
        if y < -64 or y > 319:
            return
        if block_name == AIR:
            return

        cx  = x >> 4;   cz  = z >> 4
        rx  = cx >> 5;  rz  = cz >> 5
        sy  = (y + 64) >> 4
        sy -= 4
        lx  = x & 15
        ly  = (y + 64) & 15
        lz  = z & 15

        self._data[(rx, rz)][(cx, cz)][sy][(lx, ly, lz)] = block_name
        self._n_blocks += 1

        # Betten brauchen eine Block Entity für die Farbe
        if "_bed" in block_name and "bedrock" not in block_name:
            # Farbe aus Block-Namen extrahieren
            color_name = block_name.split(":")[1].split("_bed")[0]
            COLOR_IDS = {
                "white":0,"orange":1,"magenta":2,"light_blue":3,
                "yellow":4,"lime":5,"pink":6,"gray":7,
                "light_gray":8,"cyan":9,"purple":10,"blue":11,
                "brown":12,"green":13,"red":14,"black":15
            }
            color_id = COLOR_IDS.get(color_name, 14)
            be = nbtlib.Compound({
                "id":    nbtlib.String("minecraft:bed"),
                "x":     nbtlib.Int(x),
                "y":     nbtlib.Int(y),
                "z":     nbtlib.Int(z),
                "color": nbtlib.Int(color_id),
                "keepPacked": nbtlib.Byte(0),
            })
            rx2 = cx >> 5; rz2 = cz >> 5
            self._block_entities[(rx2, rz2)][(cx, cz)].append(be)

    def delete_block(self, x, y, z):
        """Entfernt einen Block (setzt ihn auf Luft) — funktioniert auch wenn er schon gesetzt wurde."""
        if y < -64 or y > 319:
            return
        cx = x >> 4; cz = z >> 4
        rx = cx >> 5; rz = cz >> 5
        sy = ((y + 64) >> 4) - 4
        lx = x & 15; ly = (y + 64) & 15; lz = z & 15
        key = (lx, ly, lz)
        sect = self._data.get((rx, rz), {}).get((cx, cz), {}).get(sy, {})
        if key in sect:
            del sect[key]

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        regions = list(self._data.items())
        print(f"  {self._n_blocks:,} Blöcke  |  {len(regions)} Regions")

        for i, ((rx, rz), chunks_dict) in enumerate(regions):
            print(f"  [{i+1}/{len(regions)}] r.{rx}.{rz}.mca "
                  f"({len(chunks_dict)} Chunks)",
                  end="\r", flush=True)
            chunks_nbt = {
                (cx, cz): _make_chunk(cx, cz, secs,
                    self._block_entities.get((rx,rz), {}).get((cx,cz), []))
                for (cx, cz), secs in chunks_dict.items()
            }
            _write_region(rx, rz, chunks_nbt, self.output_dir)

        print(f"\n  ✓ Fertig")
