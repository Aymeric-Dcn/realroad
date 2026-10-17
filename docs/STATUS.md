# État du projet / Project status — 17/10/2026

## 🇫🇷

**Fait et validé en jeu (v0.2)**
- Pipeline complet en une commande (`realroad.py`), `.kn5` écrit directement (plus de Blender ni de ksEditor).
- Route Napoléon (D6085) St-Vallier → Escragnolles, 17,4 km :
  - route reconnaissable, largeur mesurée, marquages réels ;
  - terrain jusqu'à 300 m et relief lointain (« le décor est super ») ;
  - couleurs réglées le 16/10 au soir (« vraiment bien, de loin le paysage est très ressemblant ») ;
  - bord de route refait : photo aérienne, raccord progressif, terrain découpé autour de la route.

**Fait cette nuit, à tester (v0.3)** — demande trois téléchargements sur le PC :
- **Arbres** (`fetch.py mnh` + `trees.py`) : hauteur de végétation LiDAR HD. Mesuré sur 3 zones du tracé :
  50 à 120 arbres/ha, hauteur médiane 10 à 13 m, soit ~40 à 55 000 arbres à moins de 140 m de la route.
  Billboards croisés (shader ksTree, comme Kunos), textures générées dans les couleurs de l'ortho.
- **Bâtiments** (`fetch.py buildings`) : BD TOPO, avec hauteur à l'égout et au faîtage (champs vérifiés
  sur le service IGN), toits texturés par l'ortho, façades génériques provençales.
- **Photos de rue** (`fetch.py streetview` + `streetview_sheet.py`) : l'IGN a photographié toute la route
  en 360° le 27/04/2026 (Panoramax, Etalab 2.0), 100 % du tracé à moins de 5 m. Planche « vue conducteur »
  tous les 250 m pour comparer avec le jeu. Rien n'est mis dans Git.

À regarder en jeu : densité et allure des arbres (trop ? pas assez ? couleur ?), performances (images/s),
bâtiments de Saint-Vallier, puis comparer avec `streetview_sheet.jpg`.

**Retours du 17/10 au matin (v0.3 roulée)** : ville « pas mal », pré reconnaissable ; trop de
bâtiments hors ville et certains trop grands ; arbres à peine visibles, plutôt transparents.
Corrigé (v0.4, à tester) :
- bâtiments : abris / serres (`construction_legere`) retirés, hauteur BD TOPO = égout (et non faîtage),
  toit limité par l'emprise ; 776 bâtiments au lieu de 1133 ;
- arbres : 3 plans au lieu de 2, feuillage plus dense, couleurs éclaircies (l'ortho donnait un vert-bleu
  sombre), matériau plus lumineux ;
- **bord de route en photos de rue** (expérimental) : les photos Panoramax sont projetées sur
  l'accotement (bande de 12 m), poses recalées sur la route, test de cohérence entre deux photos pour
  écarter ce qui n'est pas du sol (voitures, murs, arbres projetés), fondu vers l'ortho au bord.

**Limites connues**
- Arbres et bâtiments sans collision (on les traverse).
- Façades génériques (une seule texture) ; arbres stylisés (4 modèles).
- Bord de route en photos : ne marche bien que sur du sol (accotement, herbe, gravier) ; murs, haies et
  voitures ne peuvent pas être projetés sur un sol plat : il faudra leur géométrie 3D (LiDAR complet).
- Sous les arbres et dans Saint-Vallier, bords de route estimés.
- Accotements goudronnés au-delà des lignes traités comme de l'accotement.

**Prochaines étapes proposées** : régler arbres et bâtiments d'après le premier essai ; textures tirées des
photos Panoramax (façades, talus, murets) ; export glTF pour Unreal ; contrôle qualité automatique.

## 🇬🇧

**Done and approved in game (v0.2)**:
- one-command pipeline that writes the `.kn5` directly;
- Route Napoléon with measured width and real markings;
- terrain to 300 m and distant relief ("the scenery is great");
- colours tuned on the evening of 16/10;
- roadside redone.

**Done overnight, to test (v0.3)** — needs three downloads on the PC:
- **trees** from the LiDAR HD canopy height: 50-120 trees/ha measured, ~40-55k trees within 140 m;
  ksTree billboards with textures in the orthophoto colours;
- **buildings** from BD TOPO, with eaves and ridge heights, orthophoto roofs and generic façades;
- **street photos**: IGN shot the whole road in 360° on 27/04/2026 (Panoramax, Etalab 2.0); a driver-view
  sheet every 250 m to compare with the game, kept out of Git.

**Feedback 17/10 morning (v0.3)**:
- town OK;
- too many buildings outside town;
- trees barely visible.

Fixed in v0.4:
- sheds removed and heights corrected;
- trees use 3 planes with denser, lighter foliage;
- experimental roadside textured from the street photos.

**Known limits**:
- no collisions on trees and buildings;
- generic façades and stylised trees;
- estimated edges under trees and in Saint-Vallier;
- paved shoulders treated as verge.

**Next**:
- tune trees and buildings after the first drive;
- textures from the Panoramax photos;
- glTF export for Unreal;
- automatic QA.
