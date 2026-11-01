"""
VicCount — server local (FastAPI). Deschide http://localhost:8765
Din telefon (aceeași rețea): http://IP-ul-PC-ului:8765
"""
import argparse
import io
import json
import os
import socket
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
MAX_SIDE = int(os.environ.get("VICCOUNT_MAX_SIDE", "2400"))

app = FastAPI(title="VicCount")
_counter = None


def counter():
    global _counter
    if _counter is None:
        from engine import Counter

        _counter = Counter()
    return _counter


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/api/health")
def health():
    c = counter()
    return {"ok": True, "device": str(c.device)}


@app.post("/api/count")
async def count(
    image: UploadFile = File(...),
    text: str = Form(""),
    boxes: str = Form("[]"),          # [[x1,y1,x2,y2], ...] în pixeli imaginii trimise
    neg_text: str = Form(""),
    refine: bool = Form(True),
):
    try:
        img = Image.open(io.BytesIO(await image.read()))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, "Imagine invalidă")

    box_list = json.loads(boxes or "[]")
    w, h = img.size
    scale = 1.0
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        box_list = [[v * scale for v in b] for b in box_list]

    r = counter().count(img, text=text, boxes_xyxy=box_list, neg_text=neg_text, refine=refine)
    return JSONResponse(r.__dict__)


app.mount("/static", StaticFiles(directory=WEB), name="static")


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    print("[VicCount] încarc modelul…", flush=True)
    try:
        counter()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    print(f"[VicCount] gata pe {_counter.device}")
    print(f"  PC:      http://localhost:{a.port}")
    print(f"  Telefon: http://{lan_ip()}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
