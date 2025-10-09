# Searchable-Generative-Image-Compression
A neural image compression codec that makes bitstreams searchable (by CLIP) and reconstructs with a generative decoder.

---

## Highlights
- **Searchable bitstreams**: while compressing, a CLIP embedding will be extracted and packed into the `.c2df` bitstream so that the compressed image(s) can be indexed & retrieved by semantics.
- **Generative reconstruction**: a learned (neural) decoder reconstructs perceptually high‑quality images.

## Quickstart

0) Prerequisites

Download [the model weight](https://huggingface.co/Hsuan-Wei/model/tree/main)

```bash
conda create -n test python=3.12
```

1) Clone this repo then

```bash
python -m pip install -U pip setuptools wheel
python -m pip install "pip<24.1"
cd [path/to/this/repo]
pip install -r requirements.txt
```

2) Build the entropy coder for detail branch

```bash
sudo apt-get install cmake g++
cd src
mkdir build
cd build
cmake ../cpp -DCMAKE_BUILD_TYPE=Release[Debug]
make -j
```

3) Compress

```bash
cd src
python compress.py \
    --base_config ./config/config_test.yaml \
    --ckpt_path ../checkpoints/model.ckpt \
    --dataset_dir "../IO/images" \
    --save_dir "../IO" \
    --gpu_idx 0
```

## License
- This project is released under the Apache-2.0 License.

## Acknowledgements
This project builds upon open-source components such as PyTorch, OpenCLIP, Dual-generative-Latent-Fusion, FAISS, and FastAPI.

---