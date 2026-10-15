# realroad — routes réelles → Assetto Corsa

Génère automatiquement des routes roulables à partir de données ouvertes
(IGN LiDAR HD, BD ORTHO, OpenStreetMap). Premier tronçon : Route Napoléon
(D6085) entre Saint-Vallier-de-Thiey et Escragnolles.

## Installation (Windows)

```powershell
cd realroad
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Étape 1 : données

```powershell
python fetch.py osm      # tracé → data\<nom>\route_wgs84.geojson (à vérifier sur geojson.io)
python fetch.py mnt      # relief 50 cm
python preview.py        # profil, pentes, virages → preview.png
python fetch.py ortho    # photos aériennes 20 cm
python fetch.py lidar    # nuage de points, plusieurs Go en transit, découpé au couloir
```

Chaque commande peut être relancée : ce qui est déjà téléchargé est sauté.

## Feuille de route

1. ✅ Récupération des données
2. Axe de route précis + échantillonnage LiDAR (pente, dévers, bosses)
3. Maillage route + terrain dans Blender (script Python)
4. Export FBX → ksEditor → piste AC (surfaces physiques, AI line, spawns)
5. Végétation, bâtiments, panneaux
