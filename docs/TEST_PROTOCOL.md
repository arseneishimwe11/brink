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

Optional: record motion sensors with the video
- Why: a normal phone video does not store which way is "down". Without it,
  Brink has to guess "up" from the scene, and every degree of error there is a
  degree of error in the slope angle. With sensor data it reads gravity directly.
- Recommended app (free, open source, Android): OpenCamera Sensors.
  Install it from F-Droid (search "OpenCamera Sensors") or from the releases
  page of github.com/prime-slam/OpenCamera-Sensors. It needs a phone with full
  Camera2 support, which most phones from the last several years have.
- One-time setup in the app: open Settings, turn on the Camera2 API if asked,
  then under "IMU settings" switch on "Enable sync video IMU recording".
- Film exactly as in the capture rules above. For each video the app saves a
  folder of small .csv files (accelerometer, gyroscope, frame timestamps) next
  to the video under DCIM/OpenCamera.
- Copy the video and its .csv files together into data/raw/, keeping the app's
  file names for the .csv files so they stay matched to their video.
- Fallback if that app does not work on your phone: install "Sensor Logger"
  (free on Google Play), start a recording with the accelerometer switched on,
  then film with your normal camera app while it runs in the background, and
  stop the logger after the video. Export as zipped CSV and put the zip in
  data/raw/ with the same name as the video. The timing match is rougher this
  way, but it is good enough for gravity on a slow walk.
- Film at least one sequence both ways (with and without sensors) if you can,
  so the two gravity methods can be compared on the same pile.
- Write in notes.txt which app you used for each video.
