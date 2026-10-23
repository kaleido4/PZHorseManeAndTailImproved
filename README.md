# Horse Mane & Tail Rework

Project Zomboid (Build 42) add-on for the [Horse mod](https://steamcommunity.com/sharedfiles/filedetails/?id=3661336777) that gives adult horses a new mane and tail.

![Before (top) and after (bottom)](docs/before_after_ThoroughbredBay.png)

- **Mane:** long and full, like an Iberian mane. Flat locks rise over the crest, fall down one side of the neck sweeping back along it in a bow, flick at the tip and end in a sharp point. No two locks are the same width, sweep or length, a low filler under them fills the crest where they turn over and fall, and the bottom is cut in a V: short at the poll, longest at the base of the neck, short again at the withers. A texture adds the fine strands and darkens each lock's edges, and the Horse mod still tints it in each breed's mane colour.
- **Forelock:** three pointed locks that fall from the poll down the forehead and flow onto the side of the face over the eye, lying on the skin, in the breed's tail colour.
- **Tail:** long and full, hanging straight down behind the hindquarters and cut straight across at about mid-cannon (a banged tail). Grooves around it read as strands. When the tail lifts, the lower hair trails and droops instead of swinging up as one stiff piece. It stays in each breed's tail colour, because its UVs land inside the tail area of the breed textures.
- Mares and stallions share the new models. Foals are unchanged.

Nothing is saved, so it can be added to or removed from an existing save.

![Palomino before and after](docs/before_after_AmericanQuarterPalomino.png)

## How it works

The Horse mod builds a horse from a skinned body mesh plus seven mane items attached to bones along the crest (`DEF_ManeStart`, `DEF_ManeMid1`-`5`, `DEF_ManeEnd`). It uses one model for the five middle bones, and foals use the same mane items as adults.

| File | What it does |
|---|---|
| `models_X/HorseMod/Horse/Mare.glb`, `Mare_NoHead.glb`, `Stallion.glb`, `Stallion_NoHead.glb` | replace the Horse mod's bodies (`loadModAfter=\Horse` makes them win). The old tail (36 triangles) is cut out and a new tail (480 triangles) is added, skinned to `DEF_Tail1`-`3`; the bodies with a head also get the forelock (84 triangles). The skeleton, inverse bind matrices and every other vertex are untouched |
| `models_X/HorseManeTailImproved/ManeStart.glb`, `ManeMid1`-`5.glb`, `ManeEnd.glb` | one mane mesh per bone: three or four locks and a filler under them, 166-212 triangles |
| `scripts/HorseManeTailImproved/mane_models.txt` | the models for those meshes, with `textures/HorseManeTailImproved/ManeStrands.png` |
| `lua/client/HorseManeTailImproved/ManeModels.lua` | wraps the Horse mod's `ManeManager.setupMane` so each mane item of an adult horse shows the model for its bone (`InventoryItem:setStaticModel`). Foals keep the Horse mod's manes |

The Horse mod's own mane models are not replaced, so foals look exactly as before.

Things worth knowing:

- Attached models are drawn through the bone's model transform, which carries the skeleton's 0.6 scale, so mane pieces are modelled 1/0.6 larger than they appear.
- The mane bones sit about 19° differently in every animation than in the bind pose, so the mane is shaped against the neck as it is posed in `Horse_Idle`, not the bind pose.
- With one model shared by the five middle bones, a long mane could only be as long everywhere, and the bones near the head tilt differently, so it fanned forwards under the jaw. One model per bone lets every piece fit its own part of the neck, and the V is set as how far each lock's tip hangs below where it roots on the crest (`V_CUT`), solved per lock.
- Horses do not save their attached items; the Horse mod recreates the manes every session. The per-bone model lives in the mane item's mod data, so nothing of it is saved, and removing this mod brings back the original manes.
- Every other lock sits a little further out (`LOCK_LAYER`), so overlapping locks stack instead of cutting through each other.
- The mane and forelock are shaded with the normals of the hair mass, not of each lock (`mass_normals`): up on top of the crest, turning to face the side down the neck, and out from the head for the forelock. With each lock's own rounded normals the game's light shaded every lock like a shiny metal tube; this way they light together as one soft clump. The texture is kept light for the same reason, since the breed tint is multiplied into it.
- The forelock is in the body mesh rather than on the end mane bone, because that bone turns up to 18° against the head while grazing (10° at the gallop) and lifted the forelock off the face. In the body it is skinned to `DEF_Head`, its root blended into the neck like the skin under it, and it uses the tail strips of the breed texture. Each point is wrapped onto the skin around the head's outline (`NeckSections.wrap`), so it lies on the face even where it flows down the side.
- Removing the old tail leaves a small opening in the rump; it is capped with two triangles.
- The tail has only three bones and ends around the hock, so everything below would swing with `DEF_Tail3` like a plank. Its lower part is partly weighted to `DEF_Spine1` (up to 30% at the cut), which makes it trail and droop when the tail lifts.

## Building

The models are generated by Python scripts in `blender/`, run with the Python that ships with Blender (it has numpy). Nothing in the Horse mod folder is ever written to.

```bash
"E:/SteamLibrary/steamapps/common/Blender/5.2/python/bin/python.exe" blender/build.py
```

```bash
"E:/SteamLibrary/steamapps/common/Blender/5.2/python/bin/python.exe" blender/verify.py
```

`build.py` copies the original body meshes into `blender/reference/` the first time (git-ignored), then writes every model into `Contents/`. It also writes the mane texture. `verify.py` checks the result: the skeleton is unchanged, skin weights are normalised, every new tail and forelock UV lands inside the tail strips, the forelock rides on the head and only on bodies with one, no new holes, the mane pieces are closed, the Horse mod's shared mane models are left alone, and every mane mesh has its model and Lua mapping.

The paths to the Horse mod and the game default to this machine's Steam folders; set `HORSE_MOD_DIR` and `PZ_DIR` to override them.

### Previews

`preview.py` assembles a horse the way the game does (posed by one of the Horse mod's animations, mane pieces on their bones, breed texture, mane texture and tint, the models' own normals) and renders it in Blender. Blender's sun casts shadows the game does not, so thin dark lines under the forelock are a preview artefact:

```bash
"E:/SteamLibrary/steamapps/common/Blender/blender.exe" --background --factory-startup --python blender/preview.py -- --models "Contents/mods/PZHorseManeAndTailImproved/common/media/models_X/HorseMod" --breed ThoroughbredBay --anim Horse_Gallop.glb --time 0.3 --out blender/_work/gallop.png
```

Add `--compare <Horse mod models_X/HorseMod folder>` for a before/after sheet. `poster.py` renders `poster.png`, `icon.png` and `preview.png`.

### Tuning

- `blender/mane.py`: `V_CUT` (how far the locks hang along the crest, from poll to withers), `LOCKS`/`MID_LOCKS` (where the locks root), the ranges each lock's shape is drawn from (`LOCK_WIDTH`, `LOCK_SWEEP`, `LOCK_FLICK`, `LOCK_JITTER`; change `SEED` for a different mix), `APEX` (rise over the crest), `CLEARANCE` (gap to the neck), `FILLER_*` (the filler under the bend), `SIDE` (which side the mane falls to), `FORELOCK` (the forelock's locks).
- `blender/mane_texture.py`: the strand texture.
- `blender/tail.py`: `TAIL_BOTTOM` (height of the cut), `PROFILE` (centre, width and depth from the dock down), `STRAND_DEPTH` and `STRAND_BULGE` (the strand grooves), `TRAIL_START` and `TRAIL_MAX` (how much of the lower hair stays with the body when the tail lifts).

## Credits

The body meshes in this mod are the Horse mod's mare and stallion models with the tail replaced and a forelock added. The Horse mod, its models and textures are by the [Horse team](https://github.com/PZ-HorseTeam/HorseMod).
