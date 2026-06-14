# embedding-large-corpi

Embed large text corpora, project them into a **single shared 3D latent space**,
and explore similarity in an interactive, rotatable plot.

**Live visualization:** https://jesseguerrero.github.io/embedding-large-corpi/

The first corpus is the complete **Philip Schaff** set from [CCEL](https://www.ccel.org/fathers):
the Ante-Nicene Fathers (10 vols) and Nicene & Post-Nicene Fathers Series I & II
(28 vols) — 38 plain-text volumes, ~19.6M words. The pipeline is corpus-agnostic:
drop another folder of `.txt` files in and it lands in the *same* space, so clouds
are directly comparable.

## Pipeline

```
corpora/<name>/*.txt
   │  src/chunk_corpus.py      clean CCEL boilerplate, pack ~400-word passages   (CPU, anywhere)
   ▼
data/chunks/<name>.jsonl
   │  src/embed.py             SentenceTransformer -> L2-normalized vectors       (GPU host)
   ▼
data/embeddings/<name>.npy
   │  src/reduce_visualize.py  PCA(50) -> UMAP(3, cosine) on the COMBINED set     (CPU)
   ▼
docs/index.html  +  docs/similarity.json
```

### Model
`Alibaba-NLP/gte-large-en-v1.5` — 1024-dim, 8192-token context, MTEB-strong.
Swap with `--model` (e.g. `Qwen/Qwen3-Embedding-8B`) for higher fidelity.

### Why this design
- **Chunking is the biggest lever** on representation quality — whole volumes
  would be truncated, so text is cleaned of CCEL apparatus and packed into
  ~400-word passages with overlap.
- **One shared projection.** PCA→UMAP is fit on *all* corpora together; fitting
  each separately would make coordinates incomparable.
- **Similarity is computed in full 1024-dim space** (centroid cosine, see
  `docs/similarity.json`). The 3D plot is the lossy, illustrative view.

## Run

```bash
# 1. chunk (local)
python3 src/chunk_corpus.py --src corpora/schaff --corpus schaff

# 2. embed (GPU host; needs sentence-transformers + a GPU)
python3 src/embed.py --chunks data/chunks/schaff.jsonl

# 3. reduce + visualize -> docs/index.html
python3 src/reduce_visualize.py
```

Add a corpus: `corpora/<name>/*.txt` → rerun steps 1–3 (with `--corpus <name>`).

## Notes
- Source texts and embeddings are git-ignored (size). Only code + the published
  `docs/` visualization are committed.
- The browser plot is stratified-subsampled (`--viz-cap`, default 24k points)
  for smooth rendering; similarity uses every chunk.
