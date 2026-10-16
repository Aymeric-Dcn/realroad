# État du projet / Project status — 16/10/2026

## 🇫🇷

**Fait**
- Pipeline complet de la donnée ouverte à une piste Assetto Corsa roulable :
  tracé → relief → LiDAR → profil de la chaussée → bords réels → maillage → FBX → kn5.
- Route Napoléon (D6085) St-Vallier → Escragnolles, 17,4 km : testée en jeu,
  route reconnaissable ; bosses parasites supprimées (rayon vertical mini ~520 m) ;
  largeur réelle mesurée (médiane 7,35 m, 6,3–9 m).

**À tester en jeu**
- La version avec largeur mesurée (`road_edges.py`, commit 049575b) n'a pas encore
  été roulée : relancer `road_edges.py` → `build_track.py` → Blender → ksEditor.
- Sens des voitures au départ : si elles sont à l'envers, `--spawn-flip` dans la commande Blender.

**Limites connues**
- Sous les arbres et dans la traversée de Saint-Vallier, les bords sont estimés
  (pas de ligne de rive visible) : plausibles mais moins précis.
- Les accotements goudronnés au-delà des lignes sont traités comme de l'accotement.
- Étape manuelle : ksEditor (Open FBX → Save KN5 → track).

**Prochaine étape proposée** : écrire le `.kn5` directement depuis Python (format
documenté par la communauté) pour supprimer ksEditor, puis AI line, caméras, carte.

## 🇬🇧

**Done**: full pipeline from open data to a drivable Assetto Corsa track. Route
Napoléon section tested in game: recognisable road, spurious bumps removed, real
width measured (median 7.35 m, 6.3-9 m).

**To test in game**: the measured-width build (`road_edges.py`, commit 049575b);
spawn direction (`--spawn-flip` if cars face backwards).

**Known limits**: estimated edges under trees and through Saint-Vallier; paved
shoulders beyond the edge lines treated as verge; manual ksEditor step.

**Next**: write the `.kn5` directly from Python to drop ksEditor, then AI line,
cameras, map.
