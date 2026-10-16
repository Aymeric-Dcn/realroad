# État du projet / Project status — 16/10/2026

## 🇫🇷

**Fait**
- Pipeline complet de la donnée ouverte à une piste Assetto Corsa, en une
  commande (`realroad.py`) : tracé → relief → LiDAR → profil → bords réels →
  marquages réels → maillage → `.kn5` (sans Blender ni ksEditor).
- Route Napoléon (D6085) St-Vallier → Escragnolles, 17,4 km : v0.1 roulée
  (route reconnaissable, voitures dans le bon sens, largeur mesurée cohérente).
- v0.2 (pas encore roulée) :
  - `.kn5` écrit en Python, format vérifié sur les pistes Kunos et sur le kn5 de ksEditor ;
  - terrain corrigé : ksEditor mettait ksAmbient 0 / ksSpecular 0,8, d'où le blanc ;
    valeurs reprises des pistes Kunos ;
  - textures DDS avec mipmaps ;
  - AI line, minimap, caméras de replay, vignettes ;
  - marquages réels (axe + rives, continus / pointillés) ;
  - terrain grossier jusqu'à ~300 m.

**À tester en jeu (pistes installées le 16/10)**
| Piste (Content Manager) | Contenu |
|---|---|
| `realroad_rn_v2` | kn5 direct + IA + minimap + caméras |
| `realroad_rn_v2_noai` | idem sans IA (si v2 plante au chargement) |
| `realroad_rn_v3` | v2 + marquages réels |

Le terrain étendu (~300 m) n'est dans aucune : il faut relancer
`python realroad.py --from road_markings --install "…"` sur le PC (build_track
a besoin de rasterio). Ensuite les pistes de test peuvent être supprimées.

À regarder : couleur du terrain et de la route, marquages (scintillement au
loin ?), minimap, IA (Track Day), caméras de replay.

**Limites connues**
- Sous les arbres et dans la traversée de Saint-Vallier, les bords sont estimés.
- Les accotements goudronnés au-delà des lignes sont traités comme de l'accotement.
- Relief lointain (`fetch.py far`) écrit mais pas encore testé : le nom de la couche IGN d'altitude est à confirmer au premier lancement.
- Pas d'arbres en volume (l'orthophoto seule).

**Prochaines étapes proposées** : végétation depuis le LiDAR (classes 3-5),
accotements goudronnés, puis contrôle qualité automatique.

## 🇬🇧

**Done**: full pipeline from open data to an Assetto Corsa track in one command
(`realroad.py`), writing the `.kn5` directly (no Blender, no ksEditor). Route
Napoléon v0.1 driven (recognisable road, right spawn direction, measured width
looks right). v0.2, not driven yet:
- the kn5 is written from Python;
- terrain fix: ksEditor's ksAmbient 0 / ksSpecular 0.8 washed it out;
- DDS textures with mipmaps;
- AI line, minimap, replay cameras;
- real markings;
- coarse terrain to ~300 m.

**To test in game**:
- `realroad_rn_v2`: direct kn5 with AI;
- `realroad_rn_v2_noai`: the same without AI;
- `realroad_rn_v3`: v2 with real markings.

The wider terrain needs a run on the PC (`python realroad.py --from road_markings --install "…"`).

**Known limits**:
- estimated edges under trees and through Saint-Vallier;
- paved shoulders beyond the edge lines are treated as verge;
- distant relief (`fetch.py far`) written but not tested yet;
- no 3D trees.

**Next**: vegetation from the LiDAR, paved shoulders, automatic QA.
