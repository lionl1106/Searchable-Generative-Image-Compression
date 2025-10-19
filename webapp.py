import os, shutil, subprocess, uuid, zipfile, time, datetime, hashlib, json
from pathlib import Path
from urllib.parse import quote
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

INDEX_DIR = Path(os.getenv("INDEX_DIR", "./IO/faiss")).resolve()
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PREVIEW_CACHE = Path(os.getenv("PREVIEW_CACHE", "./cache/previews")).resolve()
PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)

MEDIA_ROOTS = [
    Path(os.getenv("MEDIA_ROOT", "./")).resolve(),
    Path("./data").resolve(),
    Path("./IO").resolve(),
    INDEX_DIR.resolve(),
    INDEX_DIR.parent.resolve(),
]

def _resolve_media_path(raw: str) -> Path | None:
    try:
        p = Path(raw).expanduser()
    except Exception:
        return None
    if p.exists() and p.is_file():
        return p.resolve()
    name = Path(raw).name
    for root in MEDIA_ROOTS:
        try:
            for cand in root.rglob(name):
                if cand.is_file() and (cand.suffix.lower() in IMAGE_EXTS or cand.suffix.lower() == ".c2df"):
                    return cand.resolve()
        except Exception:
            continue
    return None

def _timing_headers(elapsed_ms: int, stage: str):
    return {
        "X-SIC-Stage": stage,
        "X-SIC-Elapsed-MS": str(int(elapsed_ms)),
        "X-SIC-Elapsed-S": f"{elapsed_ms/1000:.3f}",
        "X-SIC-Server-Clock": datetime.datetime.utcnow().isoformat() + "Z",
        "Access-Control-Expose-Headers": "X-SIC-Stage, X-SIC-Elapsed-MS, X-SIC-Elapsed-S, X-SIC-Server-Clock, Content-Disposition, Content-Type"
    }

def _safe_media_type(p: Path) -> str:
    suf = p.suffix.lower()
    if suf == ".png": return "image/png"
    if suf in {".jpg", ".jpeg"}: return "image/jpeg"
    if suf == ".webp": return "image/webp"
    if suf == ".bmp": return "image/bmp"
    if suf == ".c2df": return "application/octet-stream"
    return "application/octet-stream"

def _hash_path(p: Path) -> str:
    st = p.stat()
    return hashlib.sha1((str(p.resolve()) + f"|{int(st.st_mtime)}|{st.st_size}").encode("utf-8")).hexdigest()

@app.get("/")
def home():
    return FileResponse("static/index.html", media_type="text/html")

@app.get("/file")
def serve_file(path: str):
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if p.suffix.lower() not in IMAGE_EXTS and p.suffix.lower() != ".c2df":
        raise HTTPException(status_code=403, detail="Forbidden file type")
    return FileResponse(str(p), media_type=_safe_media_type(p), filename=p.name)

def _preview_url_for_path(path: str) -> str:
    p = _resolve_media_path(path)
    if not p:
        return ""
    if p.suffix.lower() in IMAGE_EXTS:
        return f"/file?path={quote(str(p))}"
    if p.suffix.lower() == ".c2df":
        key = _hash_path(p)
        out_png = PREVIEW_CACHE / f"{key}.png"
        if not out_png.exists():
            tmp_root = PREVIEW_CACHE / f"tmp_{key}"
            in_dir = tmp_root / "in"; out_dir = tmp_root / "out"
            in_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                dst = in_dir / p.name
                shutil.copy2(p, dst)
                cmd = [
                    "python", "./src/decompress.py",
                    "--base_config", os.getenv("BASE_CONFIG", "./src/config/config_test.yaml"),
                    "--ckpt_path",   os.getenv("CKPT_PATH",   "./checkpoints/model.ckpt"),
                    "--dataset_dir", str(in_dir),
                    "--save_dir",    str(out_dir),
                    "--gpu_idx",     os.getenv("GPU_IDX", "0"),
                ]
                subprocess.run(cmd, check=True)
                outs = [q for q in out_dir.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_EXTS]
                if outs:
                    shutil.copy2(outs[0], out_png)
            except Exception:
                return f"/file?path={quote(str(p))}"
            finally:
                try: shutil.rmtree(tmp_root, ignore_errors=True)
                except Exception: pass
        if out_png.exists():
            return f"/file?path={quote(str(out_png))}"
    return ""

@app.post("/compress")
async def compress(file: UploadFile = File(...), bg: BackgroundTasks = None):
    job = str(uuid.uuid4())
    in_root  = Path(f"./tmp/{job}")
    in_dir   = in_root / "input"
    out_dir  = in_root / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = in_dir / file.filename
    with in_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    cmd = [
        "python", "./src/compress.py",
        "--base_config", os.getenv("BASE_CONFIG", "./src/config/config_test.yaml"),
        "--ckpt_path",   os.getenv("CKPT_PATH",   "./checkpoints/model.ckpt"),
        "--dataset_dir", str(in_dir),
        "--save_dir",    str(out_dir),
        "--gpu_idx",     os.getenv("GPU_IDX", "0"),
    ]
    try:
        _t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        _t1 = time.perf_counter()
        _elapsed_ms = int((_t1 - _t0) * 1000)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    c2df_files = [p for p in out_dir.rglob("*.c2df") if p.is_file()]
    if not c2df_files:
        raise HTTPException(status_code=404, detail="No .c2df outputs found")

    if len(c2df_files) == 1:
        out_path = c2df_files[0]
        if bg: bg.add_task(shutil.rmtree, in_root)
        return FileResponse(
            path=str(out_path),
            filename=out_path.name,
            media_type="application/octet-stream",
            headers=_timing_headers(_elapsed_ms, "compress")
        )
    else:
        zip_path = out_dir / f"{job}_c2df.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in c2df_files:
                zf.write(p, arcname=p.relative_to(out_dir))
        if bg: bg.add_task(shutil.rmtree, in_root)
        return FileResponse(
            path=str(zip_path),
            filename=zip_path.name,
            media_type="application/zip",
            headers=_timing_headers(_elapsed_ms, "compress")
        )
    
@app.post("/decompress")
async def decompress(file: UploadFile = File(...), bg: BackgroundTasks = None):
    job = str(uuid.uuid4())
    in_root  = Path(f"./tmp/{job}")
    in_dir   = in_root / "input"
    out_dir  = in_root / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = in_dir / file.filename
    with in_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    cmd = [
        "python", "./src/decompress.py",
        "--base_config", os.getenv("BASE_CONFIG", "./src/config/config_test.yaml"),
        "--ckpt_path",   os.getenv("CKPT_PATH",   "./checkpoints/model.ckpt"),
        "--dataset_dir", str(in_dir),
        "--save_dir",    str(out_dir),
        "--gpu_idx",     os.getenv("GPU_IDX", "0"),
    ]
    try:
        _t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        _t1 = time.perf_counter()
        _elapsed_ms = int((_t1 - _t0) * 1000)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    img_files = [p for p in out_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    if not img_files:
        raise HTTPException(status_code=404, detail="No image outputs found")

    if len(img_files) == 1:
        out_path = img_files[0]
        if bg: bg.add_task(shutil.rmtree, in_root)
        mime = "image/png" if out_path.suffix.lower()==".png" else \
               "image/jpeg" if out_path.suffix.lower() in (".jpg",".jpeg") else \
               "image/webp" if out_path.suffix.lower()==".webp" else \
               "image/bmp"  if out_path.suffix.lower()==".bmp" else \
               "application/octet-stream"
        return FileResponse(
            path=str(out_path),
            filename=out_path.name,
            media_type=mime,
            headers=_timing_headers(_elapsed_ms, "decompress")
        )
    else:
        zip_path = out_dir / f"{job}_decompressed.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in img_files:
                zf.write(p, arcname=p.relative_to(out_dir))
        if bg: bg.add_task(shutil.rmtree, in_root)
        return FileResponse(
            path=str(zip_path),
            filename=zip_path.name,
            media_type="application/zip",
            headers=_timing_headers(_elapsed_ms, "decompress")
        )

def _yield_ndjson(obj: dict):
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

@app.post("/search/stream/text")
async def stream_search_text(req: Request):
    body = await req.json()
    text = (body.get("text") or "").strip()
    topk = int(body.get("topk") or 10)
    index_dir = Path(body.get("index_dir") or INDEX_DIR).resolve()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    def gen():
        t0 = time.perf_counter()
        yield _yield_ndjson({"type":"meta","stage":"start","query_type":"text","query":text,"topk":topk})
        cmd = ["python", "./src/search.py", "query-text", "--index_dir", str(index_dir), "--text", text, "--topk", str(topk)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
            items = json.loads(out.stdout)
            yield _yield_ndjson({"type":"meta","stage":"searched","count":len(items),"elapsed_ms":int((time.perf_counter()-t0)*1000)})
            for it in items:
                p = it.get("path"); s = float(it.get("score", 0.0))
                url = _preview_url_for_path(p)
                yield _yield_ndjson({"type":"item","path":p,"score":s,"preview_url":url})
            yield _yield_ndjson({"type":"done","elapsed_ms":int((time.perf_counter()-t0)*1000)})
        except subprocess.CalledProcessError as e:
            yield _yield_ndjson({"type":"error","detail": e.stderr or e.stdout or str(e)})
        except Exception as e:
            yield _yield_ndjson({"type":"error","detail": str(e)})

    return StreamingResponse(gen(), media_type="application/x-ndjson")

@app.post("/search/stream/image")
async def stream_search_image(file: UploadFile = File(...), topk: int = 10, index_dir: str | None = None):
    job = str(uuid.uuid4())
    in_root = Path(f"./tmp/{job}"); in_root.mkdir(parents=True, exist_ok=True)
    img_path = in_root / file.filename
    with img_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    def gen():
        t0 = time.perf_counter()
        yield _yield_ndjson({"type":"meta","stage":"start","query_type":"image","filename":file.filename,"topk":int(topk)})
        cmd = ["python", "./src/search.py", "query-image", "--index_dir", str(Path(index_dir or INDEX_DIR).resolve()),
               "--image", str(img_path), "--topk", str(int(topk))]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
            items = json.loads(out.stdout)
            yield _yield_ndjson({"type":"meta","stage":"searched","count":len(items),"elapsed_ms":int((time.perf_counter()-t0)*1000)})
            for it in items:
                p = it.get("path"); s = float(it.get("score", 0.0))
                url = _preview_url_for_path(p)
                yield _yield_ndjson({"type":"item","path":p,"score":s,"preview_url":url})
            yield _yield_ndjson({"type":"done","elapsed_ms":int((time.perf_counter()-t0)*1000)})
        except subprocess.CalledProcessError as e:
            yield _yield_ndjson({"type":"error","detail": e.stderr or e.stdout or str(e)})
        except Exception as e:
            yield _yield_ndjson({"type":"error","detail": str(e)})
        finally:
            try: shutil.rmtree(in_root, ignore_errors=True)
            except Exception: pass

    return StreamingResponse(gen(), media_type="application/x-ndjson")

@app.post("/search/stream/c2df")
async def stream_search_c2df(file: UploadFile = File(...), topk: int = 10, index_dir: str | None = None):
    job = str(uuid.uuid4())
    in_root = Path(f"./tmp/{job}"); in_root.mkdir(parents=True, exist_ok=True)
    c2df_path = in_root / file.filename
    with c2df_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    def gen():
        t0 = time.perf_counter()
        yield _yield_ndjson({"type":"meta","stage":"start","query_type":"c2df","filename":file.filename,"topk":int(topk)})
        cmd = ["python", "./src/search.py", "query-c2df", "--index_dir", str(Path(index_dir or INDEX_DIR).resolve()),
               "--c2df", str(c2df_path), "--topk", str(int(topk))]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
            items = json.loads(out.stdout)
            yield _yield_ndjson({"type":"meta","stage":"searched","count":len(items),"elapsed_ms":int((time.perf_counter()-t0)*1000)})
            for it in items:
                p = it.get("path"); s = float(it.get("score", 0.0))
                url = _preview_url_for_path(p)
                yield _yield_ndjson({"type":"item","path":p,"score":s,"preview_url":url})
            yield _yield_ndjson({"type":"done","elapsed_ms":int((time.perf_counter()-t0)*1000)})
        except subprocess.CalledProcessError as e:
            yield _yield_ndjson({"type":"error","detail": e.stderr or e.stdout or str(e)})
        except Exception as e:
            yield _yield_ndjson({"type":"error","detail": str(e)})
        finally:
            try: shutil.rmtree(in_root, ignore_errors=True)
            except Exception: pass

    return StreamingResponse(gen(), media_type="application/x-ndjson")

