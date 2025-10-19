from __future__ import annotations
import argparse, json, sys, traceback
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
        raise ValueError(f"{c2df_path} No 'clip_stream' or 'clip_meta' was found, this file can't be used to search!")

    stream: bytes = enc_result['clip_stream']
    meta: Dict = enc_result['clip_meta'] or {}
    dim = int(meta.get('dim', 0))
    if dim <= 0:
        raise ValueError(f"{c2df_path} Invalid clip_meta.dim")

    raw = zstd.ZstdDecompressor().decompress(stream)
    q = np.frombuffer(raw, dtype=np.uint8)
    if q.size != dim:
        raise ValueError(f"{c2df_path} Dimension didn't match: q={q.size}, dim={dim}")

    z = dequantize_clip_u8(q)
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
        print(f"[WARN] Can't load CLIP weight（{model_id}）：{e}")
        print("[WARN] Try fallback to ViT-B-32 / laion2b_s34b_b79k")
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.to(device).eval()
    return model, tokenizer, preprocess

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
        try:
            if paths:
                _, header = unpack_c2df(paths[0])
                if isinstance(header, dict) and header.get("model_id"):
                    meta["model_id"] = header["model_id"]
        except Exception:
            pass
    else:
        raise FileNotFoundError(f"Can't find FAISS index in {index_dir}")
    return index, paths, meta


# Query
@torch.no_grad()
def encode_text(query: str, model, tokenizer, device: str) -> np.ndarray:
    tok = tokenizer([query]).to(device)
    z = model.encode_text(tok).float()
    z = z / z.norm(dim=-1, keepdim=True)
    return z.detach().cpu().numpy().astype('float32')

@torch.no_grad()
def encode_image(img_path: Path, model, preprocess, device: str) -> np.ndarray:
    im = Image.open(img_path).convert("RGB")
    x = preprocess(im).unsqueeze(0).to(device)
    z = model.encode_image(x).float()
    z = z / z.norm(dim=-1, keepdim=True)
    return z.detach().cpu().numpy().astype('float32')

def encode_c2df_query(c2df_path: Path) -> np.ndarray:
    z, _ = decode_clip_from_c2df(c2df_path)
    return z[None, :].astype('float32')


# Search
def do_search(q: np.ndarray, index: faiss.Index, paths: List[str], topk: int = 10) -> List[Tuple[str, float]]:
    k = max(1, min(topk, index.ntotal))
    sim, ids = index.search(q, k)
    out = []
    for j, i in enumerate(ids[0]):
        if i == -1: continue
        out.append((paths[i], float(sim[0, j])))
    return out


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="query-text / query-image / query-c2df）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_qt = sub.add_parser("query-text", help="searching with text")
    ap_qt.add_argument("--index_dir", type=Path, required=True)
    ap_qt.add_argument("--text", type=str, required=True)
    ap_qt.add_argument("--topk", type=int, default=10)

    ap_qi = sub.add_parser("query-image", help="searching with image")
    ap_qi.add_argument("--index_dir", type=Path, required=True)
    ap_qi.add_argument("--image", type=Path, required=True)
    ap_qi.add_argument("--topk", type=int, default=10)

    ap_qc = sub.add_parser("query-c2df", help="searching with .c2df")
    ap_qc.add_argument("--index_dir", type=Path, required=True)
    ap_qc.add_argument("--c2df", type=Path, required=True)
    ap_qc.add_argument("--topk", type=int, default=10)

    args = ap.parse_args()
    device = pick_device()

    try:
        index, paths, meta = load_index(args.index_dir)
        model_id = meta.get("model_id")
        model, tokenizer, preprocess = load_clip(model_id, device)

        if args.cmd == "query-text":
            q = encode_text(args.text, model, tokenizer, device)
        elif args.cmd == "query-image":
            q = encode_image(args.image, model, preprocess, device)
        elif args.cmd == "query-c2df":
            q = encode_c2df_query(args.c2df)
        else:
            raise ValueError(f"Unknown behavior: {args.cmd}")

        results = do_search(q, index, paths, topk=args.topk)
        print(json.dumps(
            [{"path": p, "score": s} for p, s in results],
            ensure_ascii=False, indent=2
        ))

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
