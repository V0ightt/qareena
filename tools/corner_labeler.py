"""Click the four corners of each document photo (phone-photo tier, DESIGN §9.1).

Serves a local page that shows one photo at a time. Click the document's corners in order:
top-left, top-right, bottom-right, bottom-left (as the document reads, whatever its tilt).
The corners are saved to ``corners.json`` next to the photos; the template's field boxes are
later projected through them to give ground-truth boxes for photos.

Usage:  uv run python tools/corner_labeler.py var/photos/dev
Then open http://127.0.0.1:8770 (this machine only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Corners</title>
<style>body{font-family:sans-serif;margin:12px}#wrap{position:relative;display:inline-block}
img{max-width:95vw;max-height:80vh;cursor:crosshair}.dot{position:absolute;width:12px;height:12px;
border-radius:6px;background:#e33;transform:translate(-6px,-6px);pointer-events:none}</style></head>
<body><p id="info"></p><div id="wrap"><img id="img"></div>
<p><button id="undo">Undo</button> <button id="skip">Skip</button></p>
<script>
let photos = [], i = 0, pts = [];
const img = document.getElementById("img"), wrap = document.getElementById("wrap");
const order = ["top-left", "top-right", "bottom-right", "bottom-left"];
async function load() {
  photos = await (await fetch("/todo")).json(); i = 0; show();
}
function show() {
  pts = []; wrap.querySelectorAll(".dot").forEach(d => d.remove());
  if (i >= photos.length) { document.getElementById("info").textContent = "All done."; img.src = ""; return; }
  img.src = "/photo/" + encodeURIComponent(photos[i]);
  info();
}
function info() {
  document.getElementById("info").textContent =
    `${i + 1}/${photos.length}  ${photos[i]}  -  click the ${order[pts.length] || "..."} corner`;
}
img.addEventListener("click", async (e) => {
  const r = img.getBoundingClientRect();
  const x = (e.clientX - r.left) * img.naturalWidth / r.width, y = (e.clientY - r.top) * img.naturalHeight / r.height;
  pts.push([Math.round(x), Math.round(y)]);
  const d = document.createElement("div"); d.className = "dot";
  d.style.left = (e.clientX - r.left) + "px"; d.style.top = (e.clientY - r.top) + "px"; wrap.appendChild(d);
  if (pts.length === 4) {
    await fetch("/corners", {method: "POST", headers: {"Content-Type": "application/json"},
                             body: JSON.stringify({photo: photos[i], corners: pts})});
    i += 1; show();
  } else { info(); }
});
document.getElementById("undo").onclick = () => { pts.pop(); const ds = wrap.querySelectorAll(".dot"); if (ds.length) ds[ds.length - 1].remove(); info(); };
document.getElementById("skip").onclick = () => { i += 1; show(); };
load();
</script></body></html>"""


class Corners(BaseModel):
    photo: str
    corners: list[list[int]]


def make_app(folder: Path) -> FastAPI:
    store = folder / "corners.json"
    data: dict[str, list[list[int]]] = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
    photos = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE

    @app.get("/todo")
    def todo() -> list[str]:
        return [p for p in photos if p not in data]

    @app.get("/photo/{name}")
    def photo(name: str) -> FileResponse:
        if name not in photos:  # only files listed in this folder: no path tricks
            raise HTTPException(404)
        return FileResponse(folder / name)

    @app.post("/corners")
    def save(c: Corners) -> JSONResponse:
        if c.photo not in photos or len(c.corners) != 4:
            raise HTTPException(400)
        data[c.photo] = c.corners
        store.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return JSONResponse({"saved": len(data)})

    return app


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "var/photos/dev").resolve()
    if not folder.is_dir():
        sys.exit(f"no such folder: {folder}")
    print(f"labelling {folder}  ->  http://127.0.0.1:8770")
    uvicorn.run(make_app(folder), host="127.0.0.1", port=8770, log_level="warning")
