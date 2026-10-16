# Tutoriel / Tutorial — de zéro à la piste dans Assetto Corsa

**🇫🇷 [Français](#-français) · 🇬🇧 [English](#-english)**

---

## 🇫🇷 Français

Testé sur Windows 11, Python 3.13, Blender 5.2, Assetto Corsa + Content Manager + CSP.

### 0. Prérequis

- Python 3.11+ et Git
- [Blender](https://www.blender.org/) (n'importe quelle version récente, utilisé sans interface)
- Assetto Corsa, avec son SDK : `assettocorsa\sdk\editor\ksEditor.exe` est livré avec le jeu
- ~1,5 Go d'espace disque par tronçon de ~17 km (dont ~1 Go de temporaire pour le LiDAR)

```powershell
git clone <ce dépôt> realroad
cd realroad
python -m venv .venv
.venv\Scripts\activate          # à refaire dans chaque nouveau terminal
pip install -r requirements.txt
```

### 1. Décrire le tronçon

Dans `config.yaml` : points de départ/arrivée, numéro de route (`osm_ref_regex`)
et section `track` (identifiant AC, largeur de route, lissage…).

### 2. Données et profil (~15 min)

```powershell
python fetch.py osm         # tracé ; vérifier route_wgs84.geojson sur geojson.io
python fetch.py mnt         # relief
python fetch.py ortho       # photos aériennes (relancer si interrompu, ça reprend)
python fetch.py lidar       # nuage de points du couloir
python road_profile.py      # altitude, dévers, ponts → road_profile.png à vérifier
python road_edges.py        # bords réels de la chaussée → road_edges.png à vérifier
```

### 3. Géométrie et FBX (~2 min)

```powershell
python build_track.py
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --factory-startup -b -P blender_export.py -- data\<nom>\track_build
```

Si les voitures apparaissent à l'envers en jeu, relancer la commande Blender
avec `--spawn-flip` à la fin.

### 4. ksEditor : FBX → kn5

1. Lancer `assettocorsa\sdk\editor\ksEditor.exe`
2. **File → Open FBX** → `data\<nom>\track_build\<id>.fbx` (1 à 2 min de chargement)
3. Vérifier l'onglet **Scene** : objets `1ROAD_000`, `…GRASS_terrain_…`, `AC_START_0`…
4. **File → Save KN5 → track** → enregistrer sous
   `data\<nom>\track_build\ac_track\<id>\<id>.kn5` (même nom que le dossier, en minuscules)

<!-- capture : ksEditor après import -->

### 5. Installer et rouler

```powershell
Copy-Item -Recurse -Force "data\<nom>\track_build\ac_track\<id>" "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\content\tracks\"
```

Content Manager → **Drive → Quick Drive** → choisir la piste → mode **Practice**.

<!-- capture : en jeu -->

### Problèmes rencontrés

| Symptôme | Cause / solution |
|---|---|
| `No module named …` | le venv n'est pas activé (`.venv\Scripts\activate`) |
| Blender : `blender_export.py could not be opened` | lancer la commande depuis le dossier `realroad` |
| `Orthophotos only cover N %` | téléchargement interrompu : relancer `python fetch.py ortho` |
| `HTTP 429` sur le LiDAR | limite de l'IGN : ne pas utiliser `--remote` |
| Petites bosses en roulant | augmenter `track.z_sigma_m` dans `config.yaml` |

---

## 🇬🇧 English

Tested on Windows 11, Python 3.13, Blender 5.2, Assetto Corsa + Content Manager + CSP.

### 0. Requirements

- Python 3.11+ and Git
- [Blender](https://www.blender.org/) (any recent version, run headless)
- Assetto Corsa and its SDK: `assettocorsa\sdk\editor\ksEditor.exe` ships with the game
- ~1.5 GB of disk per ~17 km section (~1 GB of it temporary, for the LiDAR)

```powershell
git clone <this repo> realroad
cd realroad
python -m venv .venv
.venv\Scripts\activate          # again in every new terminal
pip install -r requirements.txt
```

### 1. Describe the section

In `config.yaml`: start/end points, road number (`osm_ref_regex`) and the
`track` section (AC id, road width, smoothing…).

### 2. Data and profile (~15 min)

```powershell
python fetch.py osm         # centreline; check route_wgs84.geojson on geojson.io
python fetch.py mnt         # terrain
python fetch.py ortho       # aerial imagery (run again if interrupted, it resumes)
python fetch.py lidar       # corridor point cloud
python road_profile.py      # elevation, crossfall, bridges → check road_profile.png
python road_edges.py        # real carriageway edges → check road_edges.png
```

### 3. Geometry and FBX (~2 min)

```powershell
python build_track.py
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --factory-startup -b -P blender_export.py -- data\<name>\track_build
```

If cars spawn facing backwards in game, run the Blender command again with
`--spawn-flip` at the end.

### 4. ksEditor: FBX → kn5

1. Start `assettocorsa\sdk\editor\ksEditor.exe`
2. **File → Open FBX** → `data\<name>\track_build\<id>.fbx` (1-2 min to load)
3. Check the **Scene** tab: objects `1ROAD_000`, `…GRASS_terrain_…`, `AC_START_0`…
4. **File → Save KN5 → track** → save as
   `data\<name>\track_build\ac_track\<id>\<id>.kn5` (same name as the folder, lowercase)

<!-- screenshot: ksEditor after import -->

### 5. Install and drive

```powershell
Copy-Item -Recurse -Force "data\<name>\track_build\ac_track\<id>" "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\content\tracks\"
```

Content Manager → **Drive → Quick Drive** → pick the track → **Practice** mode.

<!-- screenshot: in game -->

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No module named …` | the venv is not active (`.venv\Scripts\activate`) |
| Blender: `blender_export.py could not be opened` | run the command from the `realroad` folder |
| `Orthophotos only cover N %` | interrupted download: run `python fetch.py ortho` again |
| `HTTP 429` on LiDAR | IGN rate limit: don't use `--remote` |
| Small bumps while driving | raise `track.z_sigma_m` in `config.yaml` |
