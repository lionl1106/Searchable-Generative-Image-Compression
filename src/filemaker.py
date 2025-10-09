import io, json, struct, numpy as np, torch
from pathlib import Path

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

def _load_entry(t: int, payload: bytes):
    off = 0
    if t == _T_NONE:
        return None
    if t == _T_BOOL:
        return bool(struct.unpack_from("<B", payload, 0)[0])
    if t == _T_INT:
        return struct.unpack_from("<q", payload, 0)[0]
    if t == _T_FLOAT:
        return struct.unpack_from("<d", payload, 0)[0]
    if t == _T_BYTES:
        L, = struct.unpack_from("<I", payload, 0)
        return payload[4:4+L]
    if t == _T_STR:
        L, = struct.unpack_from("<I", payload, 0)
        return payload[4:4+L].decode('utf-8')
    if t == _T_JSON:
        L, = struct.unpack_from("<I", payload, 0)
        jb = payload[4:4+L]
        return json.loads(jb.decode('utf-8'))

    if t == _T_NP:
        dt_len = struct.unpack_from("<B", payload, off)[0]; off += 1
        dt = payload[off:off+dt_len].decode('utf-8'); off += dt_len
        ndim = struct.unpack_from("<B", payload, off)[0]; off += 1
        shape = []
        for _ in range(ndim):
            d, = struct.unpack_from("<I", payload, off); off += 4
            shape.append(int(d))
        data_len, = struct.unpack_from("<I", payload, off); off += 4
        data = payload[off:off+data_len]
        arr = np.frombuffer(data, dtype=np.dtype(dt))
        return arr.reshape(shape)
    raise ValueError(f"unknown type code: {t}")

def unpack_c2df(src) -> (dict, dict):
    
    if isinstance(src, (str, Path)):
        data = Path(src).read_bytes()
    else:
        data = bytes(src)

    off = 0
    assert data[:4] == b"C2DF", "bad magic"
    off = 4
    ver, = struct.unpack_from("<H", data, off); off += 2
    # Header JSON
    hlen, = struct.unpack_from("<I", data, off); off += 4
    header = json.loads(data[off:off+hlen].decode('utf-8')) if hlen > 0 else {}
    off += hlen

    n_items, = struct.unpack_from("<I", data, off); off += 4
    enc_result = {}
    for _ in range(n_items):
        klen, = struct.unpack_from("<H", data, off); off += 2
        key = data[off:off+klen].decode('utf-8'); off += klen
        tcode = struct.unpack_from("<B", data, off)[0]; off += 1
        if tcode in (_T_INT, _T_FLOAT, _T_BOOL, _T_NONE):
            if tcode == _T_INT:
                payload = data[off:off+8]; off += 8
            elif tcode == _T_FLOAT:
                payload = data[off:off+8]; off += 8
            elif tcode == _T_BOOL:
                payload = data[off:off+1]; off += 1
            else:
                payload = b""
        else:
            L, = struct.unpack_from("<I", data, off); off += 4
            payload = data[off:off+L]; off += L
        enc_result[key] = _load_entry(tcode, payload)

    return enc_result, header