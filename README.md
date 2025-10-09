# Searchable-Generative-Image-Compression
A neural image compression codec that makes bitstreams searchable (by CLIP) and reconstructs with a generative decoder.

---

## Highlights
- **Searchable bitstreams**: while compressing, a CLIP embedding will be extracted and packed into the `.c2df` bitstream so that the compressed image(s) can be indexed & retrieved by semantics.
- **Generative reconstruction**: a learned (neural) decoder reconstructs perceptually high‑quality images.

---