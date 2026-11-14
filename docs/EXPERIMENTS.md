# Brink experiment log

Newest entries at the bottom. Each entry gives the command that
produced its numbers and the code version it ran on.

## 2026-11-14 00:46 UTC: Undercut method comparison, geometry tier, dev split

Command: `brink bench geometry --split dev --count 40 --methods profile,normals --conditions ideal,typical,poor`  
Code: commit `dabebeb` plus any uncommitted changes

40 synthetic scenes, hazard geometry run on sampled surfaces (no images).
Cloud quality sets depth noise, holes, and the assumed error in gravity and scale:
ideal (0.15% of range, 0 deg, 0%), typical (0.4%, 1 deg, 5%), poor (0.8%, 3 deg, 12%).

Undercuts and overhangs:

| Undercut method | Cloud quality | Captures | Undercut >= 20 cm flagged | 20-30 cm | 30-50 cm | >= 50 cm | Under 20 cm flagged | Controls with a false alarm | False alarms per capture | Depth error, median | Depth error, 90th pct | Depth bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| profile | ideal | 40/40 | 20/20 (100%, 84-100) | 10/10 (100%, 72-100) | 6/6 (100%, 61-100) | 4/4 (100%, 51-100) | 4/4 (100%, 51-100) | 0/14 | 0.00 | 0.007 m | 0.030 m | +0.004 m |
| normals | ideal | 40/40 | 7/20 (35%, 18-57) | 4/10 (40%, 17-69) | 0/6 (0%, 0-39) | 3/4 (75%, 30-95) | 0/4 (0%, 0-49) | 0/14 | 0.00 | 0.033 m | 0.075 m | -0.032 m |
| profile | typical | 40/40 | 20/20 (100%, 84-100) | 10/10 (100%, 72-100) | 6/6 (100%, 61-100) | 4/4 (100%, 51-100) | 4/4 (100%, 51-100) | 0/14 | 0.00 | 0.024 m | 0.084 m | +0.014 m |
| normals | typical | 40/40 | 10/20 (50%, 30-70) | 7/10 (70%, 40-89) | 0/6 (0%, 0-39) | 3/4 (75%, 30-95) | 0/4 (0%, 0-49) | 0/14 | 0.05 | 0.042 m | 0.067 m | +0.014 m |
| profile | poor | 40/40 | 19/20 (95%, 76-99) | 10/10 (100%, 72-100) | 5/6 (83%, 44-97) | 4/4 (100%, 51-100) | 4/4 (100%, 51-100) | 0/14 | 0.00 | 0.051 m | 0.084 m | +0.013 m |
| normals | poor | 40/40 | 10/20 (50%, 30-70) | 7/10 (70%, 40-89) | 0/6 (0%, 0-39) | 3/4 (75%, 30-95) | 0/4 (0%, 0-49) | 0/14 | 0.20 | 0.079 m | 0.162 m | +0.022 m |

Slope and height of the test face:

| Undercut method | Cloud quality | Faces measured | Slope error, median | 90th pct | Worst | Within 5 deg | Height error, median | Height error, 90th pct | Over-steep call correct | Time per capture, median |
|---|---|---|---|---|---|---|---|---|---|---|
| profile | ideal | 40/40 | 0.06 deg | 0.20 deg | 0.46 deg | 40/40 | 1.5% | 2.0% | 38/38 | 2.4 s |
| normals | ideal | 40/40 | 0.06 deg | 0.20 deg | 0.46 deg | 40/40 | 1.5% | 2.0% | 38/38 | 2.4 s |
| profile | typical | 40/40 | 0.63 deg | 1.01 deg | 1.39 deg | 40/40 | 5.3% | 6.8% | 38/38 | 2.4 s |
| normals | typical | 40/40 | 0.63 deg | 1.01 deg | 1.39 deg | 40/40 | 5.3% | 6.8% | 38/38 | 2.5 s |
| profile | poor | 40/40 | 2.20 deg | 2.92 deg | 3.13 deg | 40/40 | 12.3% | 15.7% | 38/38 | 1.2 s |
| normals | poor | 40/40 | 2.20 deg | 2.92 deg | 3.13 deg | 40/40 | 12.3% | 15.7% | 38/38 | 1.2 s |

Profile method (vertical cross-sections) against normals method (downward-facing patches). Tuned on this split: column width, minimum hollow height, noise floor correction, crossing-face exclusion.

## 2026-11-14 00:57 UTC: Undercut method comparison, geometry tier, frozen test split

Command: `brink bench geometry --split test --count 60 --methods profile,normals --conditions ideal,typical,poor`  
Code: commit `dabebeb` plus any uncommitted changes

60 synthetic scenes, hazard geometry run on sampled surfaces (no images).
Cloud quality sets depth noise, holes, and the assumed error in gravity and scale:
ideal (0.15% of range, 0 deg, 0%), typical (0.4%, 1 deg, 5%), poor (0.8%, 3 deg, 12%).

Undercuts and overhangs:

| Undercut method | Cloud quality | Captures | Undercut >= 20 cm flagged | 20-30 cm | 30-50 cm | >= 50 cm | Under 20 cm flagged | Controls with a false alarm | False alarms per capture | Depth error, median | Depth error, 90th pct | Depth bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| profile | ideal | 60/60 | 31/31 (100%, 89-100) | 14/14 (100%, 78-100) | 10/10 (100%, 72-100) | 7/7 (100%, 65-100) | 5/5 (100%, 57-100) | 0/21 | 0.00 | 0.012 m | 0.025 m | +0.010 m |
| normals | ideal | 60/60 | 10/31 (32%, 19-50) | 7/14 (50%, 27-73) | 0/10 (0%, 0-28) | 3/7 (43%, 16-75) | 0/5 (0%, 0-43) | 0/21 | 0.07 | 0.012 m | 0.073 m | -0.026 m |
| profile | typical | 60/60 | 31/31 (100%, 89-100) | 14/14 (100%, 78-100) | 10/10 (100%, 72-100) | 7/7 (100%, 65-100) | 5/5 (100%, 57-100) | 0/21 | 0.00 | 0.025 m | 0.059 m | +0.022 m |
| normals | typical | 60/60 | 12/31 (39%, 24-56) | 9/14 (64%, 39-84) | 0/10 (0%, 0-28) | 3/7 (43%, 16-75) | 0/5 (0%, 0-43) | 0/21 | 0.07 | 0.038 m | 0.068 m | -0.041 m |
| profile | poor | 60/60 | 31/31 (100%, 89-100) | 14/14 (100%, 78-100) | 10/10 (100%, 72-100) | 7/7 (100%, 65-100) | 5/5 (100%, 57-100) | 0/21 | 0.00 | 0.047 m | 0.103 m | -0.006 m |
| normals | poor | 60/60 | 13/31 (42%, 26-59) | 10/14 (71%, 45-88) | 0/10 (0%, 0-28) | 3/7 (43%, 16-75) | 0/5 (0%, 0-43) | 0/21 | 0.13 | 0.094 m | 0.192 m | -0.045 m |

Slope and height of the test face:

| Undercut method | Cloud quality | Faces measured | Slope error, median | 90th pct | Worst | Within 5 deg | Height error, median | Height error, 90th pct | Over-steep call correct | Time per capture, median |
|---|---|---|---|---|---|---|---|---|---|---|
| profile | ideal | 60/60 | 0.04 deg | 0.18 deg | 0.36 deg | 60/60 | 1.6% | 2.0% | 56/56 | 2.3 s |
| normals | ideal | 60/60 | 0.04 deg | 0.18 deg | 0.36 deg | 60/60 | 1.6% | 2.0% | 56/56 | 2.2 s |
| profile | typical | 60/60 | 0.71 deg | 1.00 deg | 1.23 deg | 60/60 | 5.1% | 7.2% | 56/56 | 2.3 s |
| normals | typical | 60/60 | 0.71 deg | 1.00 deg | 1.23 deg | 60/60 | 5.1% | 7.2% | 56/56 | 2.3 s |
| profile | poor | 60/60 | 2.21 deg | 3.01 deg | 3.15 deg | 60/60 | 11.3% | 14.8% | 56/56 | 1.2 s |
| normals | poor | 60/60 | 2.21 deg | 3.01 deg | 3.15 deg | 60/60 | 11.3% | 14.8% | 56/56 | 1.1 s |

First run on the test split. No parameter was changed after seeing these numbers.
