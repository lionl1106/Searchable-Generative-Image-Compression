import os, shutil, subprocess, uuid, zipfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html", media_type="text/html")

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
        subprocess.run(cmd, check=True)
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
            media_type="application/octet-stream"
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
            media_type="application/zip"
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
        subprocess.run(cmd, check=True)
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
            media_type=mime
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
            media_type="application/zip"
        )