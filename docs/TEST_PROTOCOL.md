# Real-world test protocol (soil pile)

Goal: real ground truth for the hazard engine, with no special equipment.

Setup
- Build a soil or sand pile about 1–1.5 m tall in a yard, with one face
  that is roughly planar.
- Note the material (dry sand, damp soil, etc.).
- Optional scale: paint two marks on a fixed object (wall or rock) and measure
  the distance between them with a tape. Write it in notes.txt.

Capture rules (every video)
- Walk slowly around or along the pile for 60–120 seconds, phone at chest
  height, camera pointed at the pile, steady speed, no zooming.
- Record the date, time, light conditions and phone model in notes.txt.

Sequence
1. Baseline: film the untouched pile twice.
2. Undercut: dig out the base of one face (record depth and width with a tape).
   Film twice.
3. Steepen: cut another face steeper (record your best angle estimate and
   height). Film twice.
4. Crest crack: draw a crack line with a stick along the top edge of one face
   (record length, width, depth). Film twice.
5. Water at toe: pour water to make a puddle at the foot of a face. Film twice.
6. Stretch (movement): push one section outward by a measured 3–5 cm. Film twice.

Files
- Put the videos in data/raw/<date>_<step>_<n>.mp4
- Write everything you measured in data/raw/notes.txt
- Keep one full sequence aside as a held-out test set. Say which one in notes.txt.
