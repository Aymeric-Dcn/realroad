# realroad

**🇫🇷 [Français](#-français) · 🇬🇧 [English](#-english) · 📘 [Tutoriel / Tutorial](docs/TUTORIAL.md) · 📍 [État / Status](docs/STATUS.md)**

Routes réelles → pistes de simulation, générées automatiquement à partir de
données ouvertes. / Real-world roads → driving-sim tracks, generated
automatically from open data.

Auteur / Author : **Aymeric Duchene**

| En jeu — départ de Saint-Vallier / In game — leaving Saint-Vallier | Après l'épingle / After the hairpin |
|---|---|
| ![start](docs/img/ingame/01-start.jpg) | ![hairpin](docs/img/ingame/02-after-hairpin.jpg) |

*Premier essai (v0.1) : géométrie réelle, textures du terrain encore à régler. / First drive (v0.1): real geometry, terrain textures still to fix.*

![Bords mesurés / Measured edges](docs/img/edges_overlay.png)

![Marquages réels / Real markings](docs/img/road_markings.png)

*Marquages détectés (rouge : rive droite, vert : axe, bleu : rive gauche) sur l'orthophoto « déroulée » ; en haut, tout le tracé : blanc = continu, jaune = pointillés, gris = rien. / Detected markings (red: right edge, green: centre, blue: left edge) on the unrolled orthophoto; top, the whole road: white = continuous, yellow = dashed, grey = none.*

| Profil LiDAR de la route / LiDAR road profile | Route « déroulée » / Unrolled road |
|---|---|
| ![profile](docs/img/road_profile.png) | ![unrolled](docs/img/edges_unrolled.png) |

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
route, largeur du couloir…). Ensuite, une seule commande :

```powershell
python realroad.py --install "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
```

Elle enchaîne toutes les étapes, qu'on peut aussi lancer une par une :

```powershell
python fetch.py osm      # tracé (OpenStreetMap, sinon BD TOPO IGN)
python fetch.py mnt      # relief LiDAR HD 50 cm (~16 Mo par km²)
python fetch.py ortho    # photos aériennes 20 cm
python fetch.py lidar    # nuage de points du couloir (~10 min, ~120 Mo gardés)
python road_profile.py   # vrai profil de la chaussée : altitude, dévers, ponts
python road_edges.py     # vrais bords de la chaussée : largeur et centre réels
python road_markings.py  # vrais marquages : ligne centrale et lignes de rive
python build_track.py    # géométrie de la piste (route, marquages, terrain, départs AC)
python kn5_export.py --install "<dossier assettocorsa>"   # piste AC complète
```

`kn5_export.py` écrit directement le `.kn5` (plus besoin de Blender ni de
ksEditor), l'AI line, la minimap, les caméras de replay et les vignettes de
Content Manager. `python realroad.py --from build_track` ne relance que les
dernières étapes.

Les téléchargements reprennent là où ils s'étaient arrêtés.
Les données vont dans `data/<nom-du-tronçon>/` (ignoré par Git).

### Comment ça marche

| Étape | Source | Détail |
|---|---|---|
| Tracé | OSM (Overpass) ou BD TOPO | Plus court chemin entre départ et arrivée sur les tronçons portant le numéro de route ; les ronds-points sans numéro sont recousus (≤ 60 m). |
| Relief | MNT LiDAR HD | Dalles 1 km², GeoTIFF float32 au pas de 50 cm, listées par la couche WFS `IGNF_LIDAR-HD_METADONNEE:metadata`. |
| Ponts | MNT | Le MNT est le sol nu et plonge sous les ponts : creux étroits repérés et interpolés (aperçu). La vraie altitude viendra des points LiDAR classe 17. |
| Profil route | Nuage LiDAR | Stations tous les 1 m ; droite robuste z = z0 + c·t sur ±2,5 m autour de l'axe → altitude, dévers, rugosité. Tabliers de pont (classe 17) prioritaires ; axe signalé suspect si dévers > 12 % ou rms > 12 cm. |
| Bords de chaussée | LiDAR + orthophoto + BD TOPO | Lignes de rive (pics d'intensité LiDAR + blanc sur l'ortho) → bord = ligne + 0,3 m ; ligne cachée reconstruite par symétrie autour de la ligne centrale ; saut d'intensité LiDAR dans le relief sinon ; BD TOPO comme garde-fou. Largeur médiane 7,35 m (6,3–9 m) sur la Route Napoléon. |
| Marquages | Orthophoto + LiDAR | Ligne centrale et lignes de rive suivies mètre par mètre (pic blanc sur l'ortho + pic d'intensité LiDAR), découpées en tronçons continus / pointillés / absents ; pointillés gardés là où ils sont vus, complétés au pas local. Bandes blanches 2 cm au-dessus du bitume. |
| Lissage | — | Écrêtage (médiane 15 m) puis gaussienne σ 6 m (altitude) / 10 m (dévers) : rayon vertical mini ~520 m, écart max au LiDAR < 10 cm. |
| Piste AC | — | `.kn5` écrit en Python (format vérifié sur les pistes Kunos), textures DDS DXT1 avec mipmaps, AI line `fast_lane.ai`, minimap, caméras. Terrain détaillé à 120 m puis anneau grossier jusqu'à ~300 m. |
| Nuage de points | LiDAR HD COPC | Dalles téléchargées puis découpées au couloir (±40 m) et supprimées ; on garde le sol (2) et les tabliers de pont (17). `--remote` : lecture partielle HTTP Range (plus lente, limitée par l'IGN). |

### Feuille de route

1. ✅ Tracé, relief, aperçu
2. ✅ Nuage de points LiDAR du couloir
3. ✅ Profil en long et en travers depuis le LiDAR (altitude, dévers, ponts)
4. ✅ Maillage route + terrain, export FBX via Blender
5. ✅ ksEditor → piste Assetto Corsa roulable (premier essai le 16/10/2026) : route reconnaissable, bosses lissées, largeur réelle mesurée
6. ✅ kn5 écrit directement (plus de ksEditor), AI line, caméras, minimap, miniatures ; marquages réels ; terrain étendu (v0.2, à tester en jeu)
7. 🔜 Végétation (arbres depuis le LiDAR), relief lointain à l'horizon, accotements goudronnés
8. Bâtiments, panneaux, glissières
9. Contrôle qualité automatique : règles de génie civil (pente, dévers ↔ courbure), cohérence entre sources, score de confiance par mètre, reconstruction droites/clothoïdes/paraboles, télémétrie d'une IA qui roule sur la piste

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
corridor width…). Then one command:

```powershell
python realroad.py --install "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
```

It chains every step, which can also be run one by one:

```powershell
python fetch.py osm      # centreline (OpenStreetMap, IGN BD TOPO fallback)
python fetch.py mnt      # 50 cm LiDAR HD terrain (~16 MB per km²)
python fetch.py ortho    # 20 cm aerial imagery
python fetch.py lidar    # corridor point cloud (~10 min, ~120 MB kept)
python road_profile.py   # real road surface: elevation, crossfall, bridges
python road_edges.py     # real carriageway edges: true width and centre
python road_markings.py  # real markings: centre line and edge lines
python build_track.py    # track geometry (road, markings, terrain, AC spawn points)
python kn5_export.py --install "<assettocorsa folder>"   # complete AC track
```

`kn5_export.py` writes the `.kn5` directly (no Blender, no ksEditor any more),
plus the AI line, minimap, replay cameras and Content Manager previews.
`python realroad.py --from build_track` only re-runs the last steps.

Downloads resume where they stopped. Data goes to `data/<section-name>/`
(git-ignored).

### How it works

| Step | Source | Notes |
|---|---|---|
| Centreline | OSM (Overpass) or BD TOPO | Shortest path between start and end over ways carrying the road number; unnumbered roundabouts are stitched (≤ 60 m). |
| Terrain | LiDAR HD DTM | 1 km² tiles, float32 GeoTIFF at 50 cm, listed by the WFS layer `IGNF_LIDAR-HD_METADONNEE:metadata`. |
| Bridges | DTM | The DTM is bare earth and dives under bridges: narrow dips are detected and interpolated (preview). Real deck height will come from class-17 LiDAR points. |
| Road profile | LiDAR cloud | 1 m stations; robust line z = z0 + c·t over ±2.5 m around the axis → elevation, crossfall, roughness. Bridge decks (class 17) take priority; axis flagged as suspect if crossfall > 12 % or rms > 12 cm. |
| Carriageway edges | LiDAR + orthophoto + BD TOPO | Edge lines (LiDAR intensity peaks + white in the orthophoto) → edge = line + 0.3 m; hidden line mirrored about the centre marking; LiDAR intensity step within the terrain otherwise; BD TOPO as a guard rail. Median width 7.35 m (6.3-9 m) on the Route Napoléon. |
| Markings | Orthophoto + LiDAR | Centre and edge lines followed metre by metre (white peak in the orthophoto + LiDAR intensity peak), split into continuous / dashed / absent stretches; dashes kept where they are seen, completed at the local spacing. White strips 2 cm above the asphalt. |
| Smoothing | — | Despike (15 m median) then Gaussian sigma 6 m (elevation) / 10 m (crossfall): min vertical radius ~520 m, max deviation from LiDAR < 10 cm. |
| AC track | — | `.kn5` written from Python (format checked against Kunos tracks), DXT1 DDS textures with mipmaps, `fast_lane.ai` AI line, minimap, cameras. Detailed terrain to 120 m, then a coarse ring to ~300 m. |
| Point cloud | LiDAR HD COPC | Tiles downloaded, cropped to the corridor (±40 m) and deleted; ground (2) and bridge decks (17) are kept. `--remote`: partial HTTP Range reads (slower, rate-limited by IGN). |

### Roadmap

1. ✅ Centreline, terrain, preview
2. ✅ Corridor LiDAR point cloud
3. ✅ Longitudinal and cross profiles from LiDAR (elevation, crossfall, bridges)
4. ✅ Road + terrain mesh, FBX export through Blender
5. ✅ ksEditor → drivable Assetto Corsa track (first drive on 2026-10-16): recognisable road, bumps smoothed, real width measured
6. ✅ kn5 written directly (no more ksEditor), AI line, cameras, minimap, previews; real markings; wider terrain (v0.2, to test in game)
7. 🔜 Vegetation (trees from the LiDAR), distant relief on the horizon, paved shoulders
8. Buildings, signs, guardrails
9. Automatic QA: road-design rules (grade, crossfall vs curvature), cross-source consistency, per-metre confidence score, line/clothoid/parabola reconstruction, telemetry from an AI driving the track

---

## Données / Data

- IGN LiDAR HD, BD TOPO, BD ORTHO — © IGN,
  [Licence Ouverte Etalab 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)
- OpenStreetMap — © OpenStreetMap contributors,
  [ODbL](https://www.openstreetmap.org/copyright)
