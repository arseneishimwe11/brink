"""A self-contained 3D viewer: one folder that opens straight from disk."""

from __future__ import annotations

import base64
import json
import shutil
from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from brink.domain import Report
from brink.hazards.engine import GeometryResult
from brink.outputs.colors import HAZARD_NAME, HAZARD_RGB

_MAX_POINTS = 350_000

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Brink model</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html,body{margin:0;height:100%;background:#15181c;color:#e8e8e8;font:14px/1.4 system-ui,sans-serif}
  #view{position:fixed;inset:0}
  #panel{position:fixed;top:12px;left:12px;max-width:360px;max-height:calc(100% - 24px);overflow:auto;
         background:rgba(21,24,28,.88);border:1px solid #3a4048;border-radius:8px;padding:12px 14px}
  h1{font-size:16px;margin:0 0 6px} .muted{color:#9aa3ad;font-size:12px}
  .hz{border-left:5px solid #888;padding:6px 8px;margin:8px 0;background:#1e2329;cursor:pointer;border-radius:3px}
  .hz b{font-size:12px;letter-spacing:.04em} .none{margin:8px 0}
</style></head><body>
<div id="view"></div>
<div id="panel"><h1>Brink model</h1><div id="meta" class="muted"></div><div id="list"></div>
<p class="muted" id="foot"></p><p class="muted">Drag to turn, scroll to zoom, right-drag to move. Click a hazard to look at it.</p></div>
<script src="three.min.js"></script><script src="OrbitControls.js"></script><script src="data.js"></script>
<script>
const D = window.BRINK_DATA;
function decode(b64, Type){const s=atob(b64);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new Type(u.buffer);}
const pos = decode(D.positions, Float32Array), col = decode(D.colors, Uint8Array);
const el = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(window.devicePixelRatio);
el.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x15181c);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000); camera.up.set(0,0,1);
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
geo.setAttribute('color', new THREE.BufferAttribute(col, 3, true));
geo.computeBoundingSphere();
scene.add(new THREE.Points(geo, new THREE.PointsMaterial({size:D.point_size, vertexColors:true, sizeAttenuation:true})));
const controls = new THREE.OrbitControls(camera, renderer.domElement);
const c = geo.boundingSphere.center, r = geo.boundingSphere.radius;
function lookFrom(target, dir, dist){controls.target.copy(target);camera.position.copy(target).addScaledVector(dir.normalize(), dist);controls.update();}
lookFrom(new THREE.Vector3(D.focus[0],D.focus[1],D.focus[2]), new THREE.Vector3(D.view_dir[0],D.view_dir[1],D.view_dir[2]), D.view_dist);
document.getElementById('meta').textContent = D.meta;
document.getElementById('foot').textContent = D.footer;
const list = document.getElementById('list');
if (!D.hazards.length) { const p=document.createElement('p'); p.className='none'; p.textContent=D.empty_text; list.appendChild(p); }
D.hazards.forEach(h => {
  const d = document.createElement('div'); d.className='hz'; d.style.borderLeftColor = h.color;
  d.innerHTML = '<b>'+h.level+' &middot; '+h.name+'</b><br>'+h.message+'<br><span class="muted">confidence: '+h.confidence+'</span>';
  d.onclick = () => lookFrom(new THREE.Vector3(h.at[0],h.at[1],h.at[2]), new THREE.Vector3(h.from[0],h.from[1],h.from[2]), h.dist);
  list.appendChild(d);
  const m = new THREE.Mesh(new THREE.SphereGeometry(D.marker, 16, 12), new THREE.MeshBasicMaterial({color:h.color, transparent:true, opacity:.55}));
  m.position.set(h.at[0],h.at[1],h.at[2]); scene.add(m);
});
function resize(){renderer.setSize(el.clientWidth, el.clientHeight); camera.aspect = el.clientWidth/el.clientHeight; camera.updateProjectionMatrix();}
window.addEventListener('resize', resize); resize();
(function loop(){requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera);})();
</script></body></html>
"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_viewer(geometry: GeometryResult, colours: NDArray[np.uint8], report: Report, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = resources.files("brink.outputs").joinpath("viewer_assets")
    for name in ("three.min.js", "OrbitControls.js", "THREE_LICENSE"):
        shutil.copyfile(str(assets.joinpath(name)), out_dir / name)

    xyz = geometry.cloud.xyz
    step = max(1, int(np.ceil(len(xyz) / _MAX_POINTS)))
    xyz, rgb = xyz[::step], colours[::step]
    centre = np.median(xyz, axis=0)
    extent = float(np.linalg.norm(np.percentile(xyz, 98, axis=0) - np.percentile(xyz, 2, axis=0)))
    faces = {f.id: f for f in geometry.faces}

    def view_of(face_id: str | None) -> list[float]:
        face = faces.get(face_id) if face_id else None
        direction = np.array(face.outward) if face else np.array([0.0, -1.0, 0.0])
        return (direction + np.array([0.0, 0.0, 0.35])).tolist()

    rank = {"danger": 2, "warning": 1, "note": 0}
    hazards = sorted(report.hazards, key=lambda h: (-rank[h.severity.value], -h.confidence))
    main = geometry.faces[0] if geometry.faces else None
    recon = report.reconstruction
    data = {
        "positions": base64.b64encode(np.ascontiguousarray(xyz - centre, dtype="<f4").tobytes()).decode(),
        "colors": base64.b64encode(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()).decode(),
        "point_size": max(geometry.voxel * 1.6 * np.sqrt(step), 0.005),
        "marker": 0.03 * extent,
        "focus": ((np.array(main.centroid) - centre) if main else np.zeros(3)).tolist(),
        "view_dir": view_of(main.id if main else None),
        "view_dist": 0.9 * extent,
        "meta": f"{Path(report.capture.video_path).name} | {recon.frames_registered} frames | "
                f"up from {recon.gravity.source if recon.gravity else 'n/a'} | "
                f"scale from {recon.scale.source.replace('_', ' ')}",
        "footer": "Automated screening aid, not an engineering assessment. Thresholds are provisional. "
                  "Ground that is not flagged has not been cleared.",
        "empty_text": "No hazard was measured in this capture. That is not a clearance.",
        "hazards": [
            {
                "level": h.severity.value.upper(), "name": HAZARD_NAME[h.type], "message": _escape(h.message),
                "confidence": f"{h.confidence_level.value} ({h.confidence:.2f})",
                "color": "#{:02x}{:02x}{:02x}".format(*HAZARD_RGB[h.type]),
                "at": (np.array(h.location) - centre).tolist(), "from": view_of(h.face_id),
                "dist": max(0.35 * extent, 4 * (h.extent_along_face or 0.5)),
            }
            for h in hazards
        ],
    }
    (out_dir / "data.js").write_text("window.BRINK_DATA = " + json.dumps(data) + ";\n")
    index = out_dir / "index.html"
    index.write_text(_HTML)
    return index
