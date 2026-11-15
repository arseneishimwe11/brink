# Licences of dependencies and models

Brink is meant to become a commercial product, so every dependency is listed
here with its licence and what that means for us. Versions are the ones
installed in the project environment when this file was last checked.

**How this was checked:** package metadata and the licence files inside each
installed wheel, plus a scan of the compiled libraries for bundled components.
This is an engineering inventory, not legal advice. Have a lawyer review it
before shipping a product.

## Needs attention

| Item | Issue | What to do |
|---|---|---|
| `pycolmap` wheel from PyPI | COLMAP itself is BSD-3, but the published binary has SuiteSparse compiled in, including two modules that are **GPL** (CHOLMOD Supernodal and SuiteSparseQR). | Fine while Brink runs only on our own servers: the GPL applies when you distribute the binary, not when you use it. **Before giving Brink's software to anyone** (an on-site installation, a customer's server), build COLMAP and Ceres without those modules (Ceres can use Eigen's sparse solver or CHOLMOD without Supernodal), or replace the camera solver. |
| System `ffmpeg` program | The Ubuntu package is a GPL build (it includes x264). | Used only by the benchmark to encode synthetic test videos, as a separate program. It is not part of the engine. If the product ever needs to encode video, use an LGPL build or a different encoder. |
| Hazard rules | Slope limits start from US OSHA 29 CFR 1926 Subpart P Appendix B, a public regulation. | No licence issue. They are provisional until an engineer approves them. |

## Engine dependencies (needed to run `brink run`)

| Package | Version | Licence | Notes |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause (with small permissive parts) | |
| scipy | 1.18.1 | BSD-3-Clause | |
| opencv-python-headless | 5.0.0 | Apache-2.0 | Wheel bundles FFmpeg libraries under **LGPL** (shared libraries, which is the form LGPL allows), plus OpenBLAS (BSD), libvpx and libaom (BSD), libavif (BSD), OpenSSL 1.1.1 (OpenSSL licence). Keep the bundled notice file with any redistribution. |
| pycolmap (COLMAP) | 4.2.1 | BSD-3-Clause | See "Needs attention". Also compiled in: Ceres (BSD-3), FAISS (MIT), PoseLib (BSD-3), METIS (Apache-2.0), VLFeat SIFT (BSD-2), OpenImageIO (Apache-2.0), Boost (BSL-1.0), SQLite (public domain). The SIFT patent expired in 2020. |
| open3d | 0.20.0 | MIT | Bundles Embree and oneTBB (Apache-2.0) and other permissive libraries. Needs the system libraries libEGL and libGL (MIT-style) and libgomp (GPL with the GCC runtime exception, which permits linking from any program). |
| scikit-image | 0.26.0 | BSD-3-Clause | Used for marching cubes in the synthetic scenes. |
| pydantic | 2.14.3 | MIT | |
| typer | 0.29.1 | MIT | |
| PyYAML | 6.0.3 | MIT | |
| reportlab | 5.1.0 | BSD-3-Clause | PDF output. |
| pillow | 13.0.0 | MIT-CMU | Required by reportlab. |
| three.js | 0.147.0 | MIT | Two files are copied into `src/brink/outputs/viewer_assets/` with their licence. |

## Development and benchmark only (not shipped)

| Package | Licence | Use |
|---|---|---|
| pytest, ruff, mypy, hatchling | MIT | Tests, lint, type checks, packaging |
| matplotlib | Matplotlib licence (BSD-compatible) | Debug plots |
| ffmpeg (system) | GPL build | Encoding synthetic videos; see above |

## Models

Phase 0 ships **no machine-learning model**. Everything in the default
pipeline is classical geometry and image processing.

Candidates that would bring model weights, and their status:

| Model | Licence | Status |
|---|---|---|
| ALIKED features + LightGlue matcher | ALIKED: BSD-3. LightGlue: Apache-2.0. | Not used. COLMAP 4 supports them, but the PyPI build of pycolmap was compiled without the ONNX runtime and stops with "ALIKED feature extraction requires ONNX support". Worth building in and testing on real videos if SIFT struggles there. Check the training-data terms of the specific weights before adopting. |
| Depth Anything V2 **Small** | Apache-2.0 (code and weights) | Not used. Candidate for a second dense method. The Base, Large and Giant variants are **non-commercial** and must not be used. |

## Deliberately excluded

| Tool or model | Licence | Why excluded |
|---|---|---|
| OpenMVS | AGPL-3.0 | The usual CPU multi-view stereo. AGPL would oblige us to publish our server code. Replaced by our own stereo stage on OpenCV. |
| COLMAP dense stereo (patch match) | BSD-3 | Needs an NVIDIA GPU. |
| DUSt3R, MASt3R | CC BY-NC-SA 4.0 | Non-commercial. |
| VGGT (original weights) | Non-commercial research licence | Non-commercial. |
| SuperPoint, SuperGlue | Non-commercial research licence | Non-commercial. |
| Ultralytics YOLO | AGPL-3.0 | Would bind the server code. |
| SegFormer pretrained weights | NVIDIA source code licence | Non-commercial. |
| pymeshlab / MeshLab | GPL-3.0 | Copyleft; not needed. |
| Blender as a Python module | GPL | Copyleft; our own renderer is used for synthetic scenes. |

## Tools for people, not linked into Brink

| Tool | Licence | Use |
|---|---|---|
| OpenCamera Sensors (Android app) | GPL-3.0 | Suggested in `docs/TEST_PROTOCOL.md` for recording video with motion sensors. It runs on the tester's phone and only produces files; nothing of it is included in Brink. |
| Sensor Logger (Android app) | Proprietary, free tier | Fallback suggested in the same place. |

## Brink's own code

Proprietary (see `pyproject.toml`). No open-source licence has been chosen.
