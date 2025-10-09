# ---------- import packages ----------
import inspect
import torch
torch.set_grad_enabled(False)

import argparse
from pathlib import Path
import yaml
import os, sys, importlib, shutil
import numpy as np
from tqdm import tqdm
import ast

sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/../")

from torchvision.utils import save_image

from omegaconf import OmegaConf

from filemaker import unpack_c2df

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

def _as_int_list(x):
    # turn x into a list of int
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
    # scan all .c2df files
    c2df_dir = Path(args.dataset_dir)
    files = sorted(c2df_dir.glob("*.c2df"))
    if not files:
        raise FileNotFoundError(f"No .c2df files under: {c2df_dir}")

    # load the model
    DEVICE = f"cuda:{args.gpu_idx}"
    cfg = load_config(args.base_config)
    cfg.model.params.ckpt_path = args.ckpt_path
    cfg.model.params.ignore_keys = ['epoch_for_strategy', 'lmbda_idx', 'lmbda_list']
    out_dir = Path(args.save_dir)
    img_dir = out_dir / "recon"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_dir.mkdir(parents=True, exist_ok=False)

    # set the model
    model = instantiate_from_config(cfg.model)
    model = model.to(DEVICE).eval()
    model.hybrid_codec.quantize_feat.force_zero_thres = 0.12
    model.hybrid_codec.quantize_feat.update(force=True)

    # decode each file
    for fp in tqdm(files, total=len(files)):
        enc_result, header = unpack_c2df(fp)
        
        enc_result = sanitize_enc_result_types(enc_result)
        enc_result = filter_kwargs_for_fn(model.decode_only, enc_result)

        img_rec_padded = model.decode_only(**enc_result)
        img_rec = torch.nn.functional.pad(
            img_rec_padded, (-header["padding"][0], -header["padding"][1], -header["padding"][2], -header["padding"][3])
        )
        img_rec = img_rec.clamp(-1.0, 1.0).mul(0.5).add(0.5)
        save_image(img_rec, str(img_dir / (fp.stem + ".png")))

        torch.cuda.empty_cache()
    
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
    parser.add_argument('--gpu_idx', type=int, default=0, help='gpu index')
    parser.add_argument('--fid_patch_size', type=int, default=256, help='patch size for fid calculation')
    parser.add_argument('--fid_split_patch_num', type=int, default=2, help='split patch number for fid calculation')
    args = parser.parse_args()
    test(args)


