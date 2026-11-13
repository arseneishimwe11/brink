# Brink: project brief

## Name
Brink: the edge of the pit, and the edge of disaster.
Tagline: Brink flags the wall before it falls.
One line: Brink finds dangerous ground from one phone video, before it falls.
Local (Kinyarwanda) name option for the field app: Murinzi ("guardian").
Note: check trademarks before committing (Brink's is a large security company,
in a different industry). Backup name: Plumb.

## What it is
Brink turns an ordinary phone video of a mine, quarry, soil pit or construction
excavation into a 3D model, and checks that model for dangerous shapes:
undercut bases, overhangs, faces that are too steep or too high for their material,
cracks along the top edge, and water at the foot of a wall.
Every day it produces a clear hazard map, a plain warning ("Face B is undercut
by 1.2 m. Do not work below it."), and a dated, location-tagged inspection record.
As a second layer, it compares captures over time to track ground movement.

Vision: Brink becomes the ground-safety layer for every slope, cut and
excavation that can fall on people: mines and quarries first, then construction
pits and trenches, hillside walls, road cuts and drainage channels.

## The problem
- Mining accidents in Rwanda killed at least 429 people and injured 272 in five
  years (RMB report cited by The New Times, 2023).
- July 9, 2026: a licensed quarry collapsed in Bugesera, killing 10 workers. The
  district suspended quarrying in the sector pending inspections.
- August 2026: six people died when a soil pit for brickmaking collapsed in
  Musanze. Workers had reportedly dug into the side of the pit.
- RMB has told operators to inspect sites daily during the rainy seasons, keep
  documented daily inspection reports at their offices, close dangerous areas,
  and identify cracks and unstable ground.
- Those inspections are done by eye. Small operators have no drone, no survey
  team and often no engineer on site. District inspectors cannot visit every site.

## Why now, and what is hard
Large mines already model pit walls in 3D with drones and professional software.
Small operators have nothing. The opportunity is a daily hazard check from the
phone already in the supervisor's pocket: no drone, no pilot, no permits.
The hard part is making a casual handheld video produce a 3D model good enough to
measure angles, overhangs and undercuts reliably, in mud, dust and bad light.

## Solution: what users see
1. The supervisor films a slow walk along the pit or quarry face (about 2 min),
   guided by on-screen tips.
2. The video becomes a 3D model.
3. A hazard map highlights undercuts, overhangs, over-steep faces, crest cracks
   and water at the toe, with measurements and a confidence level.
4. A daily card states what to avoid today, in Kinyarwanda and English.
5. An automatic inspection record (timestamped, location-tagged, tamper-evident).
6. Later: movement between days, rain-aware risk, volumes and stockpiles.

## How it works
- Capture: guided video, plus phone motion sensors and location.
- Frame selection: keep sharp, overlapping frames.
- Masking (AI segmentation): remove people, machines, vegetation, sky.
- Camera solving (Structure from Motion), then dense reconstruction.
- Orientation: find "up" (gravity) from the phone sensors or the scene.
  Angles need no scale. Heights need rough scale (phone AR tracking, one taped
  distance, or painted marks).
- Hazard geometry: slope angle and height per face, overhang and undercut
  detection, crest-crack detection (3D plus image), water detection at the toe.
- Rules: conservative hazard rules by material type, to be approved by a mining
  or geotechnical engineer advisor.
- Report: hazard map, plain warnings, inspection record.
- Phase 2: align captures from different days and measure movement (M3C2-style).

## What Brink never does
- It never says a wall is "safe". It flags hazards and leaves decisions to people.
- It is not engineering certification.
- Movement detection is never sold as an early-warning guarantee.

## Customers
- First: licensed mining companies and quarry cooperatives (monthly fee per site).
- Second: construction firms with deep excavations and trenches (per project).
- Later: insurers (a mining climate-insurance scheme is under discussion in
  Rwanda), mineral buyers and responsible-sourcing programs (hypothesis to test),
  regulators, and neighbouring countries.

## Roadmap
- Phase 0: single-capture hazard engine plus benchmark.
  Exit: reliably flags undercuts and over-steep faces from a casual phone walk.
- Phase 1: Android capture app, processing server, web dashboard, daily report.
  Exit: one site films daily for two weeks.
- Phase 2: movement between days, rain-aware risk, engineer-approved rules,
  volumes. Exit: first paying site.
- Phase 3: excavations, hillside walls, multiple sites, region.
