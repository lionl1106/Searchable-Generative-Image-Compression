# ---------- import packages ----------
import inspect, io, ast
import faiss
import zstandard as zstd
import open_clip
import torch
torch.set_grad_enabled(False)

import argparse
from pathlib import Path
import struct
import yaml
import os, sys, importlib, shutil
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/../")

from glob import glob
from torchvision import transforms
from omegaconf import OmegaConf

from entropy.compression_model import get_padding_size
import io, json

_T_BYTES  = 0
_T_STR    = 1
_T_INT    = 2
_T_FLOAT  = 3
_T_JSON   = 4
_T_NP     = 5
_T_NONE   = 6
_T_BOOL   = 7

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().contiguous().numpy()
    return None

def _dump_entry(key: str, val):
    k = key.encode('utf-8')
    if key in {"z_indeices_shape", "h_indices_shape", "y_shape", "x_shape"} or key.endswith("_shape"):
        arr = np.asarray(val, dtype = np.int32)
        dtype_s = arr.dtype.str.encode('utf-8')
        data_b = arr.tobytes(order='C')
        payload = []
        payload.append(struct.pack("<B", len(dtype_s))); payload.append(dtype_s)
        payload.append(struct.pack("<B", arr.ndim))
        for d in arr.shape:
            payload.append(struct.pack("<I", int(d)))
        payload.append(struct.pack("<I", len(data_b))); payload.append(data_b)
        return k, _T_NP, b"".join(payload)
        
        # return k, _T_NP, pack_numpy(arr)
    if key in {"token_length", "num_tokens", "n_tokens"} or key.endswith("_length"):
        return k, _T_INT, struct.pack("<q", int(val))
    
    # None / bool / int / float / bytes / str
    if val is None:
        return k, _T_NONE, b""
    if isinstance(val, bool):
        return k, _T_BOOL, struct.pack("<B", 1 if val else 0)
    if isinstance(val, int):
        return k, _T_INT, struct.pack("<q", val) 
    if isinstance(val, float):
        return k, _T_FLOAT, struct.pack("<d", val) 
    if isinstance(val, (bytes, bytearray, memoryview)):
        b = bytes(val)
        return k, _T_BYTES, struct.pack("<I", len(b)) + b
    if isinstance(val, str):
        b = val.encode('utf-8')
        return k, _T_STR, struct.pack("<I", len(b)) + b

    # numpy / torch.Tensor
    arr = _to_numpy(val)
    if arr is not None:
        dtype_s = arr.dtype.str.encode('utf-8')
        data_b = arr.tobytes(order='C')
        payload = []
        payload.append(struct.pack("<B", len(dtype_s))); payload.append(dtype_s)
        payload.append(struct.pack("<B", arr.ndim))
        for d in arr.shape:
            payload.append(struct.pack("<I", int(d)))
        payload.append(struct.pack("<I", len(data_b))); payload.append(data_b)
        return k, _T_NP, b"".join(payload)

    # list / dict
    if isinstance(val, (list, dict)):
        jb = json.dumps(val, ensure_ascii=False).encode('utf-8')
        return k, _T_JSON, struct.pack("<I", len(jb)) + jb

    s = str(val).encode('utf-8')
    return k, _T_STR, struct.pack("<I", len(s)) + s

def pack_c2df(enc_result: dict, header: dict) -> bytes:
    
    blob = io.BytesIO()
    # ---- Magic + ver ----
    ver = int(header.get("version", 2))
    blob.write(b"C2DF"); blob.write(struct.pack("<H", ver))

    # ---- Header JSON ----
    hb = json.dumps(header, ensure_ascii=False).encode('utf-8')
    blob.write(struct.pack("<I", len(hb))); blob.write(hb)

    # ---- Enc-Result ----
    items = list(enc_result.items())
    blob.write(struct.pack("<I", len(items)))

    # ---- key_len | key | type | payload_len | payload ----
    for k, v in items:
        k_b, t, payload = _dump_entry(k, v)
        blob.write(struct.pack("<H", len(k_b))); blob.write(k_b)
        blob.write(struct.pack("<B", t))
        if t in (_T_INT, _T_FLOAT, _T_BOOL, _T_NONE):
            blob.write(payload)
        else:
            blob.write(struct.pack("<I", len(payload))); blob.write(payload)

    return blob.getvalue()

# ---------------- DDP helpers ----------------
def is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def get_rank():
    return int(os.environ.get("RANK", "0"))

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))

def setup_device(args):
    """Return (device, local_rank, world_size, using_ddp)"""
    if is_dist():
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return device, local_rank, dist.get_world_size(), True
    else:
        gpu_idx = int(getattr(args, "gpu_idx", 0))
        device = torch.device(f"cuda:{gpu_idx}")
        return device, 0, 1, False

# ---------------- main components ----------------
class ClipCodec:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cuda:0"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.model_name = f"{model_name}:{pretrained}"
        self.zctx = zstd.ZstdCompressor(level=19)

    @torch.no_grad()
    def image_to_unit_vec(self, img_tensor_CHW: torch.Tensor) -> np.ndarray:
        """img_tensor_CHW: [-1,1] 範圍, CxHxW"""
        pil = transforms.ToPILImage()(img_tensor_CHW.clamp(-1,1).mul(0.5).add(0.5))
        x = self.preprocess(pil).unsqueeze(0).to(self.device)  # (1,3,224,224)
        z = self.model.encode_image(x).float()
        z = z / z.norm(dim=-1, keepdim=True)
        return z.squeeze(0).detach().cpu().numpy().astype("float32")  # (D,)

    def quantize_u8_and_compress(self, z_unit: np.ndarray):
        q = np.clip(np.round((z_unit * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
        cbytes = self.zctx.compress(q.tobytes())
        meta = {
            "model_id": self.model_name,
            "dim": int(z_unit.shape[0]),
            "quant": "u8_symmetric_-1_1",
            "codec": "zstd",
            "zstd_level": 19,
        }
        return cbytes, meta


class FaissDB:
    def __init__(self, index_dir: str, dim: int):
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.ids_path = os.path.join(index_dir, "ids.txt")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.ids = []
        if os.path.exists(self.ids_path):
            with open(self.ids_path, "r", encoding="utf-8") as f:
                self.ids = [line.strip() for line in f if line.strip()]

    def add(self, vec_unit: np.ndarray, doc_id: str):
        assert vec_unit.ndim == 1
        v = vec_unit.copy()[None, :]
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        self.index.add(v.astype("float32"))
        self.ids.append(doc_id)

    def persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            for _id in self.ids:
                f.write(_id + "\n")


def filter_kwargs_for_fn(fn, kwargs: dict):
    sig = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in sig}

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


# ---------- define dataloader ----------
class Test_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        image = self.transform(image)
        image = image * 2.0 - 1.0
        image_name = os.path.splitext(os.path.split(image_ori)[1])[0]
        return image, image_name

    def __len__(self):
        return len(self.image_path)


def _as_int_list(x):
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, (np.integer, int)):
        return [int(x)]
    if isinstance(x, str):
        try:
            y = ast.literal_eval(x)
            if isinstance(y, (list, tuple)):
                return [int(v) for v in y]
            return [int(y)]
        except Exception:
            return [int(x)]
    try:
        return [int(x)]
    except Exception:
        raise TypeError(f"cannot cast {type(x)} to int list")

def sanitize_enc_result_types(enc: dict) -> dict:
    KEYS_SHAPE = {"z_indices_shape", "h_indices_shape", "y_shape", "x_shape"}
    KEYS_LEN   = {"token_length", "num_tokens", "n_tokens", "length"}
    out = dict(enc)
    for k, v in list(out.items()):
        if k.endswith("_shape") or k in KEYS_SHAPE:
            out[k] = tuple(_as_int_list(v))
        elif k.endswith("_length") or k in KEYS_LEN:
            out[k] = int(_as_int_list(v)[0])
    return out


@torch.no_grad()
def test(args):
    # ----- setup device / DDP -----
    device, local_rank, world_size, using_ddp = setup_device(args)

    # dataset & sampler
    test_set = Test_Dataset(args.dataset_dir)
    sampler = DistributedSampler(test_set, shuffle=False, drop_last=False) if using_ddp else None
    test_loader = DataLoader(
        test_set, batch_size=1,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=4, pin_memory=True
    )

    # config & output dirs
    cfg = load_config(args.base_config)
    cfg.model.params.ckpt_path = args.ckpt_path
    cfg.model.params.ignore_keys = ['epoch_for_strategy', 'lmbda_idx', 'lmbda_list']

    out_dir = args.save_dir
    img_dir = os.path.join(out_dir, "recon")
    bit_dir = os.path.join(out_dir, "bitstreams")
    index_dir = os.path.join(out_dir, "faiss")
    clip_dir = os.path.join(out_dir, "clip_vecs")

    if (not using_ddp) or (local_rank == 0):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=False)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(bit_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        os.makedirs(clip_dir, exist_ok=True)
    if using_ddp:
        dist.barrier()

    # load model
    model = instantiate_from_config(cfg.model)
    model = model.to(device).eval()
    model.hybrid_codec.quantize_feat.force_zero_thres = 0.12
    model.hybrid_codec.quantize_feat.update(force=True)

    if using_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    clip_codec = ClipCodec(device=device)

    pbar = tqdm(total=len(test_loader), disable=(using_ddp and local_rank != 0))

    for i, (img, img_name) in enumerate(test_loader):
        if using_ddp:
            test_loader.sampler.set_epoch(0) if hasattr(test_loader.sampler, "set_epoch") else None

        img = img.to(device, non_blocking=True)
        img_name = img_name[0]

        im_H, im_W = img.shape[2], img.shape[3]

        # padding to 256
        padding_l, padding_r, padding_t, padding_b = get_padding_size(im_H, im_W, p=256)
        img_padded = torch.nn.functional.pad(
            img, (padding_l, padding_r, padding_t, padding_b), mode="replicate",
        )

        enc_result = model.module.encode_only(img_padded) if using_ddp else model.encode_only(img_padded)

        # -------- CLIP encode and quantize --------
        clip_vec = clip_codec.image_to_unit_vec(img[0])
        clip_stream, clip_meta = clip_codec.quantize_u8_and_compress(clip_vec)

        enc_result["clip_stream"] = clip_stream
        enc_result["clip_meta"]   = clip_meta

        header = {
            "version": 2,
            "model_id": clip_meta.get("model_id", ""),
            "embed_dim": int(clip_meta.get("dim", 0)),
            "quant_type": clip_meta.get("quant", "u8_symmetric_-1_1"),
            "image_hw": [int(im_H), int(im_W)],
            "padding":  [int(padding_l), int(padding_r), int(padding_t), int(padding_b)],
        }

        c2df = pack_c2df(enc_result, header)
        out_path = os.path.join(bit_dir, f"{img_name}.c2df")
        with open(out_path, "wb") as f:
            f.write(c2df)

        np.save(os.path.join(clip_dir, f"{img_name}.npy"), clip_vec)

        torch.cuda.empty_cache()
        pbar.update(1)

    pbar.close()

    if using_ddp:
        dist.barrier()
    if (not using_ddp) or (get_rank() == 0):
        npy_list = sorted(glob(os.path.join(clip_dir, "*.npy")))
        if len(npy_list) > 0:
            sample = np.load(npy_list[0])
            faiss_db = FaissDB(index_dir=index_dir, dim=int(sample.shape[0]))
            for npy_path in npy_list:
                img_stem = os.path.splitext(os.path.basename(npy_path))[0]
                vec = np.load(npy_path)
                doc_id = os.path.join(bit_dir, f"{img_stem}.c2df")
                if os.path.exists(doc_id):
                    faiss_db.add(vec, doc_id=doc_id)
            faiss_db.persist()

    if using_ddp:
        dist.destroy_process_group()

    return None


def init_func():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark  = False
    torch.manual_seed(0)
    torch.set_num_threads(4)
    np.random.seed(seed=0)


if __name__ == "__main__":
    init_func()
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, help='path to base config')
    parser.add_argument('--ckpt_path', type=str, help='path to checkpoint')
    parser.add_argument('--dataset_dir', type=str, help='path to dataset')
    parser.add_argument('--save_dir', type=str, help='path to save results')
    parser.add_argument('--gpu_idx', type=int, default=0, help='gpu index (單卡時使用；多卡由 LOCAL_RANK 決定)')
    parser.add_argument('--fid_patch_size', type=int, default=256, help='patch size for fid calculation')
    parser.add_argument('--fid_split_patch_num', type=int, default=2, help='split patch number for fid calculation')
    args = parser.parse_args()
    test(args)
