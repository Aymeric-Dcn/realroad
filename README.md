# realroad

**🇫🇷 [Français](#-français) · 🇬🇧 [English](#-english)**

Routes réelles → pistes de simulation, générées automatiquement à partir de
données ouvertes. / Real-world roads → driving-sim tracks, generated
automatically from open data.

Auteur / Author : **Aymeric Duchene**

---

## 🇫🇷 Français

### L'idée

Rouler dans un simulateur (Assetto Corsa) sur les routes qu'on connaît, avec
leur vraie géométrie : pente, dévers, virages, bosses. Au lieu de modéliser
chaque route à la main pendant des mois, `realroad` la génère à partir des
données publiques de l'IGN (LiDAR HD, BD TOPO, orthophotos) et
d'OpenStreetMap.

Premier tronçon : **Route Napoléon (D6085), Saint-Vallier-de-Thiey →
Escragnolles** (17,4 km, Pas de la Faye à ~980 m).

### Installation (Windows)

```powershell
cd realroad
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Utilisation

Le tronçon se décrit dans `config.yaml` (points de départ/arrivée, numéro de
route, largeur du couloir…).

```powershell
python fetch.py osm      # tracé (OpenStreetMap, sinon BD TOPO IGN)
python fetch.py mnt      # relief LiDAR HD 50 cm (~16 Mo par km²)
python preview.py        # profil, pentes, virages, ponts → preview.png
python fetch.py ortho    # photos aériennes 20 cm
python fetch.py lidar    # nuage de points du couloir (~10 min, ~120 Mo gardés)
python road_profile.py   # vrai profil de la chaussée : altitude, dévers, ponts
```

Toutes les commandes peuvent être relancées : ce qui est déjà là est sauté.
Les données vont dans `data/<nom-du-tronçon>/` (ignoré par Git).

### Comment ça marche

| Étape | Source | Détail |
|---|---|---|
| Tracé | OSM (Overpass) ou BD TOPO | Plus court chemin entre départ et arrivée sur les tronçons portant le numéro de route ; les ronds-points sans numéro sont recousus (≤ 60 m). |
| Relief | MNT LiDAR HD | Dalles 1 km², GeoTIFF float32 au pas de 50 cm, listées par la couche WFS `IGNF_LIDAR-HD_METADONNEE:metadata`. |
| Ponts | MNT | Le MNT est le sol nu et plonge sous les ponts : creux étroits repérés et interpolés (aperçu). La vraie altitude viendra des points LiDAR classe 17. |
| Profil route | Nuage LiDAR | Stations tous les 1 m ; droite robuste z = z0 + c·t sur ±2,5 m autour de l'axe → altitude, dévers, rugosité. Tabliers de pont (classe 17) prioritaires ; axe signalé suspect si dévers > 12 % ou rms > 12 cm. |
| Nuage de points | LiDAR HD COPC | Dalles téléchargées puis découpées au couloir (±40 m) et supprimées ; on garde le sol (2) et les tabliers de pont (17). `--remote` : lecture partielle HTTP Range (plus lente, limitée par l'IGN). |

### Feuille de route

1. ✅ Tracé, relief, aperçu
2. ✅ Nuage de points LiDAR du couloir
3. 🔄 Profil en long et en travers depuis le LiDAR (altitude, dévers, ponts) ; reste : axe précis, bosses
4. Maillage route + terrain dans Blender (script Python)
5. Export FBX → ksEditor → piste Assetto Corsa (surfaces, AI line, départs)
6. Végétation, bâtiments, panneaux

---

## 🇬🇧 English

### The idea

Drive the roads you know in a sim (Assetto Corsa) with their real geometry:
grade, camber, curves, bumps. Instead of hand-modelling each road for months,
`realroad` generates it from French public data (IGN LiDAR HD, BD TOPO,
orthophotos) and OpenStreetMap.

First section: **Route Napoléon (D6085), Saint-Vallier-de-Thiey →
Escragnolles** (17.4 km, Pas de la Faye at ~980 m).

### Install (Windows)

```powershell
cd realroad
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

A section is described in `config.yaml` (start/end points, road number,
corridor width…).

```powershell
python fetch.py osm      # centreline (OpenStreetMap, IGN BD TOPO fallback)
python fetch.py mnt      # 50 cm LiDAR HD terrain (~16 MB per km²)
python preview.py        # profile, grades, curves, bridges → preview.png
python fetch.py ortho    # 20 cm aerial imagery
python fetch.py lidar    # corridor point cloud (~10 min, ~120 MB kept)
python road_profile.py   # real road surface: elevation, crossfall, bridges
```

Every command is resumable: anything already downloaded is skipped. Data goes
to `data/<section-name>/` (git-ignored).

### How it works

| Step | Source | Notes |
|---|---|---|
| Centreline | OSM (Overpass) or BD TOPO | Shortest path between start and end over ways carrying the road number; unnumbered roundabouts are stitched (≤ 60 m). |
| Terrain | LiDAR HD DTM | 1 km² tiles, float32 GeoTIFF at 50 cm, listed by the WFS layer `IGNF_LIDAR-HD_METADONNEE:metadata`. |
| Bridges | DTM | The DTM is bare earth and dives under bridges: narrow dips are detected and interpolated (preview). Real deck height will come from class-17 LiDAR points. |
| Road profile | LiDAR cloud | 1 m stations; robust line z = z0 + c·t over ±2.5 m around the axis → elevation, crossfall, roughness. Bridge decks (class 17) take priority; axis flagged as suspect if crossfall > 12 % or rms > 12 cm. |
| Point cloud | LiDAR HD COPC | Tiles downloaded, cropped to the corridor (±40 m) and deleted; ground (2) and bridge decks (17) are kept. `--remote`: partial HTTP Range reads (slower, rate-limited by IGN). |

### Roadmap

1. ✅ Centreline, terrain, preview
2. ✅ Corridor LiDAR point cloud
3. 🔄 Longitudinal and cross profiles from LiDAR (elevation, crossfall, bridges); to do: precise centreline, bumps
4. Road + terrain mesh in Blender (Python script)
5. FBX export → ksEditor → Assetto Corsa track (surfaces, AI line, spawns)
6. Vegetation, buildings, signs

---

## Données / Data

- IGN LiDAR HD, BD TOPO, BD ORTHO — © IGN,
  [Licence Ouverte Etalab 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)
- OpenStreetMap — © OpenStreetMap contributors,
  [ODbL](https://www.openstreetmap.org/copyright)
