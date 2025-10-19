from __future__ import annotations
import argparse, json, sys, traceback
import requests
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import faiss
import zstandard as zstd
from PIL import Image

import torch
import open_clip

from filemaker import unpack_c2df

def l2n(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def dequantize_clip_u8(q: np.ndarray) -> np.ndarray:
    z = (q.astype(np.float32) / 255.0) * 2.0 - 1.0
    return l2n(z.astype(np.float32))

def decode_clip_from_c2df(c2df_path: Path) -> Tuple[np.ndarray, Dict]:
    enc_result, header = unpack_c2df(c2df_path)
    if 'clip_stream' not in enc_result or 'clip_meta' not in enc_result:
        raise ValueError(f"{c2df_path} No 'clip_stream' or 'clip_meta' is found, this file can't be used to search!")

    stream: bytes = enc_result['clip_stream']
    meta: Dict = enc_result['clip_meta'] or {}
    dim = int(meta.get('dim', 0))
    if dim <= 0:
        raise ValueError(f"{c2df_path} invalid clip_meta.dim")

    raw = zstd.ZstdDecompressor().decompress(stream)
    q = np.frombuffer(raw, dtype=np.uint8)
    if q.size != dim:
        raise ValueError(f"{c2df_path} Can't match the dimension: q={q.size}, dim={dim}")

    z = dequantize_clip_u8(q)  # [-1,1] to L2 normalize
    return z.astype('float32'), header


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(model_id: str | None, device: str):
    try:
        if model_id and ":" in model_id:
            model_name, pretrained = model_id.split(":", 1)
        else:
            model_name, pretrained = "ViT-B-32", "laion2b_s34b_b79k"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception as e:
        print(f"[WARN] Can't load the CLIP weight（{model_id}）：{e}")
        print("[WARN] Try fallback to ViT-B-32 / laion2b_s34b_b79k")
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.to(device).eval()
    return model, tokenizer, preprocess


# -----------------------------
# Build and load FAISS index
# -----------------------------
def build_index_from_c2df_dir(c2df_dir: Path, index_dir: Path) -> None:
    
    index_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted([p for p in c2df_dir.glob("**/*.c2df")])
    if not paths:
        raise RuntimeError(f"Empty folder: {c2df_dir}")
    feats: List[np.ndarray] = []
    keep:  List[str] = []
    model_id_from_header: str | None = None
    for p in paths:
        try:
            z, header = decode_clip_from_c2df(p)
            feats.append(z[None, :])
            keep.append(str(p))
            if model_id_from_header is None and isinstance(header, dict):
                model_id_from_header = header.get("model_id")
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
    if not feats:
        raise RuntimeError("No available .c2df")
    X = np.concatenate(feats, axis=0).astype('float32')
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "paths.json").write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding='utf-8')
    (index_dir / "meta.json").write_text(json.dumps({"dim": d, "model_id": model_id_from_header}, ensure_ascii=False, indent=2), encoding='utf-8')
    # (index.faiss + ids.txt)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "ids.txt").write_text("\n".join(keep), encoding='utf-8')
    print(f"[OK] Index process completed!: N={index.ntotal}, dim={d}")
    if model_id_from_header:
        print(f"[INFO] Suggested CLIP model: {model_id_from_header}")


def load_index(index_dir: Path) -> Tuple[faiss.Index, List[str], Dict]:

    new_idx_path = index_dir / "faiss.index"
    old_idx_path = index_dir / "index.faiss"
    if new_idx_path.exists() and (index_dir / "paths.json").exists():
        index = faiss.read_index(str(new_idx_path))
        paths: List[str] = json.loads((index_dir / "paths.json").read_text(encoding='utf-8'))
        try:
            meta: Dict = json.loads((index_dir / "meta.json").read_text(encoding='utf-8'))
        except Exception:
            meta = {}
    elif old_idx_path.exists() and (index_dir / "ids.txt").exists():
        index = faiss.read_index(str(old_idx_path))
        paths = [line.strip() for line in (index_dir / "ids.txt").read_text(encoding='utf-8').splitlines() if line.strip()]
        meta = {"dim": index.d}
        if paths:
            _, header = unpack_c2df(paths[0])
            if isinstance(header, dict) and header.get("model_id"):
                meta["model_id"] = header["model_id"]
    else: raise FileNotFoundError(f"Can't find the FAISS index in {index_dir}")
    return index, paths, meta


# download random images (Piscum)
def _parse_wh(s: str) -> tuple[int,int]:
    if isinstance(s, str) and "x" in s:
        w, h = s.lower().split("x", 1)
        return int(w), int(h)
    v = int(s)
    return v, v

def download_random_picsum(n: int, out_dir: Path, size: str = "512x512", seed: int | None = None, timeout: int = 20) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = _parse_wh(size)
    ok = 0
    import uuid as _uuid
    rng = random.Random(seed)
    for i in range(n):
        seed_str = str(_uuid.UUID(int=rng.getrandbits(128)))
        url = f"https://picsum.photos/seed/{seed_str}/{w}/{h}.jpg?random={i}"
        fn = out_dir / f"picsum_{seed_str}.jpg"
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            with open(fn, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            ok += 1
        except Exception as e:
            print(f"[WARN] Download process failed!: {url} -> {e}")
            continue
    return ok

def ensure_images_count(image_dir: Path, desired: int, auto_download: bool = False, download_dir: Path | None = None, 
                        size: str = "512x512", seed: int | None = None, timeout: int = 20) -> None:

    existing = list_images(image_dir)
    have = len(existing)
    if have >= desired or not auto_download:
        return
    need = desired - have
    dd = download_dir or image_dir
    print(f"[INFO] Not enough images (have {have} < required {desired}); auto-downloading {need} images to {dd}")
    got = download_random_picsum(need, dd, size=size, seed=seed, timeout=timeout)
    print(f"[INFO] Download complete: added {got} images")

def list_images(root: Path, exts=None):
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    exts = {e if e.startswith(".") else "."+e for e in {e.lower() for e in exts}}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files

@torch.no_grad()
def encode_images_in_batches(img_paths, model, preprocess, device: str, batch_size: int = 32):
    feats = []
    cur = []
    for pth in img_paths:
        try:
            im = Image.open(pth).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Can't read the image {pth}: {e}")
            continue
        cur.append(preprocess(im))
        if len(cur) == batch_size:
            x = torch.stack(cur, 0).to(device)
            z = model.encode_image(x).float()
            z = z / z.norm(dim=-1, keepdim=True)
            feats.append(z.detach().cpu().numpy().astype("float32"))
            cur = []
    if cur:
        x = torch.stack(cur, 0).to(device)
        z = model.encode_image(x).float()
        z = z / z.norm(dim=-1, keepdim=True)
        feats.append(z.detach().cpu().numpy().astype("float32"))
    if not feats:
        return None
    return np.concatenate(feats, 0).astype("float32")

def build_index_from_image_dir(image_dir: Path, index_dir: Path, model_id: str | None, device: str, batch_size: int = 32, exts=None, limit: int | None = None, 
                               random_pick: bool = False, seed: int | None = None, desired: int | None = None, auto_download: bool = False, download_dir: Path | None = None, download_size: str = "512x512", timeout: int = 20) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    if desired is not None and auto_download:
        ensure_images_count(image_dir, desired, auto_download=True, download_dir=download_dir, size=download_size, seed=seed, timeout=timeout)

    all_imgs = list_images(image_dir, exts=exts)
    if not all_imgs:
        raise RuntimeError(f"There is no image in {image_dir}")
    target_n = desired if (desired is not None and desired > 0) else limit
    if target_n is not None and target_n > 0 and target_n <= len(all_imgs):
        rng = random.Random(seed)
        if random_pick:
            all_imgs = rng.sample(all_imgs, target_n)
        else:
            all_imgs = all_imgs[:target_n]
    print(f"[INFO] Using {len(all_imgs)} images to build the index")

    model, tokenizer, preprocess = load_clip(model_id, device)
    X = encode_images_in_batches(all_imgs, model, preprocess, device, batch_size=batch_size)
    if X is None:
        raise RuntimeError("Failed to build FAISS index!")
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    paths = [str(p) for p in all_imgs]
    faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "paths.json").write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding="utf-8")
    (index_dir / "meta.json").write_text(json.dumps({"dim": d, "model_id": model_id}, ensure_ascii=False, indent=2), encoding="utf-8")
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "ids.txt").write_text("\n".join(paths), encoding="utf-8")
    print(f"[OK] Index process completed!: N={index.ntotal}, dim={d}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="SIC build tool (download data + build FAISS index; build / build-images / download)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_dl = sub.add_parser("download", help="Download random images to the specified directory (Lorem Picsum)")
    ap_dl.add_argument("--out_dir", type=Path, required=True)
    ap_dl.add_argument("--desired", type=int, required=True, help="Target count (checks existing; only fills the shortfall)")
    ap_dl.add_argument("--size", type=str, default="512x512")
    ap_dl.add_argument("--seed", type=int, default=None)
    ap_dl.add_argument("--timeout", type=int, default=20)

    ap_build = sub.add_parser("build", help="Build index from directory")
    ap_build.add_argument("--c2df_dir", type=Path, required=True)
    ap_build.add_argument("--index_dir", type=Path, required=True)

    ap_bimg = sub.add_parser("build-images", help="Build index from an image directory (can auto-download to reach the desired count)")
    ap_bimg.add_argument("--image_dir", type=Path, required=True)
    ap_bimg.add_argument("--index_dir", type=Path, required=True)
    ap_bimg.add_argument("--model_id", type=str, default=None, help="e.g., ViT-B-32:laion2b_s34b_b79k")
    ap_bimg.add_argument("--batch_size", type=int, default=32)
    ap_bimg.add_argument("--exts", type=str, default="jpg,jpeg,png,webp,bmp")
    ap_bimg.add_argument("--limit", type=int, default=None, help="Use only N images to build the index (ignored if --desired is also given)")
    ap_bimg.add_argument("--desired", type=int, default=None, help="Target number of images; if insufficient and --auto_download is set, auto-download to reach it")
    ap_bimg.add_argument("--random", action="store_true", help="Use random sampling (requires --limit or --desired)")
    ap_bimg.add_argument("--seed", type=int, default=None)
    ap_bimg.add_argument("--auto_download", action="store_true", help="If images are insufficient, automatically download from Picsum to make up the shortfall")
    ap_bimg.add_argument("--download_dir", type=Path, default=None, help="Directory to save downloaded images (default = --image_dir)")
    ap_bimg.add_argument("--download_size", type=str, default="512x512", help="Download image size, e.g., 512x512 or 512")
    ap_bimg.add_argument("--timeout", type=int, default=20, help="Per-image download timeout (seconds)")

    args = ap.parse_args()
    device = pick_device()

    try:
        if args.cmd == "download":
            args.out_dir.mkdir(parents=True, exist_ok=True)
            have = len(list_images(args.out_dir))
            need = max(0, args.desired - have)
            if need <= 0:
                print(f"[INFO] Already have {have} images, download process is not needed")
                return
            got = download_random_picsum(need, args.out_dir, size=args.size, seed=args.seed, timeout=args.timeout)
            print(f"[OK] Download successfully: {got} images (total {have+got})")
            return
        if args.cmd == "build":
            build_index_from_c2df_dir(args.c2df_dir, args.index_dir)
            return

        if args.cmd == "build-images":
            exts = [e.strip() for e in (args.exts.split(",") if isinstance(args.exts, str) else args.exts) if e.strip()]
            build_index_from_image_dir(
                args.image_dir, args.index_dir, args.model_id, device,
                batch_size=args.batch_size, exts=exts, limit=args.limit,
                random_pick=args.random, seed=args.seed,
                desired=args.desired, auto_download=args.auto_download,
                download_dir=args.download_dir, download_size=args.download_size, timeout=args.timeout
            )
            return

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
