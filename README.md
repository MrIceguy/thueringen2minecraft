# thuringen2minecraft

Konvertiert echte Geodaten aus dem Geoportal Thueringen in eine spielbare Minecraft Java 1.21.5 Welt.

## Ergebnis

Eine 1:1-Rekonstruktion von Thueringer Staedten und Landschaften in Minecraft:
- Echtes Gelaende aus DGM1 (1m Auflosung)
- 3D-Gebaeude aus LoD2 CityGML mit Stockwerken, Treppen, Tueren, Fenstern
- Strassennetze, Gewaesser und Vegetation aus ATKIS Basis-DLM
- Bewohner (Villager), Betten, Arbeitsstationen

## Voraussetzungen

```
pip install numpy scipy nbtlib geopandas shapely rasterio pyproj
```

## Datenbezug

Alle Geodaten kostenlos unter: https://www.geoportal-th.de/

Benoetigt:
- **DGM1** – Digitales Gelaendemodell 1m (`.tif` Kacheln) → Ordner `dgm/`
- **DOM1** – Digitales Oberflaechenmodell 1m (`.tif` Kacheln) → Ordner `dom/`
- **LoD2** – 3D-Gebaeude CityGML (`.gml` Dateien) → Ordner `LoD2/`
- **ATKIS Basis-DLM** – Vektordaten (Shapefiles) → Ordner `atkis/`
  - `atkis/ver/` – Verkehr (Strassen, Plaetze)
  - `atkis/gew/` – Gewaesser
  - `atkis/sie/` – Siedlung
  - `atkis/veg/` – Vegetation

## Verwendung

```bash
# Kleine Testregion (300x300m)
python thuringen2minecraft.py --bbox 634700 635000 5616700 5617000

# Groessere Region (3x3km)
python thuringen2minecraft.py --bbox 634000 637000 5615000 5618000

# Eigene Region (ETRS89 / UTM Zone 32N Koordinaten)
python thuringen2minecraft.py --bbox WEST OST SUED NORD
```

Die generierte Welt wird automatisch nach `%AppData%\.minecraft\saves\` kopiert (Windows).

## Projektstruktur

| Datei | Beschreibung |
|-------|--------------|
| `thuringen2minecraft.py` | Hauptskript, Pipeline-Steuerung |
| `tile_loader.py` | Laedt DGM/DOM GeoTIFF-Kacheln |
| `lod2_loader.py` | Parst CityGML LoD2 Gebaeude |
| `atkis_layer.py` | Rastert ATKIS-Vektordaten |
| `anvil_writer.py` | Schreibt Minecraft Anvil .mca Dateien |
| `entity_writer.py` | Platziert Villager, Betten, Treppen, Laternen |

## Konfiguration

In `thuringen2minecraft.py`:

```python
REAL_MIN_H = 450.0   # Minimale Hoehe in der BBox (Meter ueber NN)
STORY_H    = 4       # Stockwerkshoehe in Bloecken (1 Boden + 3 Wand)
```

## Koordinatensystem

Die Eingangsdaten verwenden **ETRS89 / UTM Zone 32N (EPSG:25832)**.
1 Meter Realwelt = 1 Block in Minecraft.

## Lizenz

MIT
