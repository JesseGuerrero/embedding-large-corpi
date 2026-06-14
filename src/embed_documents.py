#!/usr/bin/env python3
"""One-point-per-document embedding + concept words, projected to 3D with PCA.

Each .txt file -> chunked -> chunk embeddings mean-pooled -> one L2-normalized
document vector (labeled by filename). Each concept phrase -> one vector. All
vectors share the same Qwen3-Embedding space and are projected to 3D via PCA
(appropriate for a small number of points).

Document vectors are cached per-file in an .npz (the slow part: a big corpus is
tens of thousands of chunks). Concept words are cheap and always re-embedded, so
adding/removing concept points is near-instant once the document cache exists.

The interactive distance tool is injected separately by src/add_distance_tool.py
so the visualization logic lives in one place.
"""
import argparse, glob, json, os, sys, time
import numpy as np

def chunk_words(text, size):
    w = text.split()
    return [" ".join(w[i:i+size]) for i in range(0, len(w), size)] or [""]

def embed_all(args):
    import torch
    from sentence_transformers import SentenceTransformer

    cache = {}
    if os.path.exists(args.cache):
        d = np.load(args.cache, allow_pickle=True)
        cache = {str(l): np.asarray(v, dtype=np.float32) for l, v in zip(d["labels"], d["vecs"])}
        print(f"[cache] loaded {len(cache)} document vectors from {args.cache}", file=sys.stderr)

    files = sorted(glob.glob(os.path.join(args.src, "*.txt")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    need = [n for n in names if n not in cache]

    model = None
    if need or concepts:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mk = {"torch_dtype": torch.bfloat16} if dev == "cuda" else {}
        model = SentenceTransformer(args.model, trust_remote_code=True, device=dev, model_kwargs=mk)
        model.max_seq_length = args.max_seq

    # documents (use cache where possible)
    for f, name in zip(files, names):
        if name in cache:
            print(f"[doc] {name}: cached", file=sys.stderr); continue
        text = open(f, encoding="utf-8", errors="replace").read()
        chunks = chunk_words(text, args.chunk_words)
        t = time.time()
        emb = model.encode(chunks, batch_size=args.batch, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
        doc = emb.mean(0); doc = (doc / (np.linalg.norm(doc) + 1e-12)).astype(np.float32)
        cache[name] = doc
        print(f"[doc] {name}: {len(chunks)} chunks -> 1 vector ({time.time()-t:.0f}s)", file=sys.stderr)

    # persist the (possibly grown) document cache
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    clabels = list(cache.keys())
    np.savez(args.cache, labels=np.array(clabels), vecs=np.vstack([cache[k] for k in clabels]))
    print(f"[cache] saved {len(clabels)} document vectors -> {args.cache}", file=sys.stderr)

    # assemble: documents (file order) then concepts
    labels, kinds, vecs = [], [], []
    for name in names:
        labels.append(name); kinds.append("document"); vecs.append(cache[name])
    if concepts:
        cemb = model.encode(concepts, normalize_embeddings=True, convert_to_numpy=True)
        for w, v in zip(concepts, cemb):
            labels.append(w); kinds.append("concept"); vecs.append(v.astype(np.float32))
            print(f"[concept] {w}", file=sys.stderr)

    return np.vstack(vecs).astype(np.float32), np.array(labels), np.array(kinds)

def build_html(X, labels, kinds, out):
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go

    SIM = (X @ X.T).astype(float)   # vectors are L2-normalized => dot = cosine
    Y = PCA(n_components=3, random_state=42).fit_transform(X)

    fig = go.Figure()
    for kind, color, size, sym in [("document", "#4C78A8", 7, "circle"),
                                   ("concept", "#E45756", 6, "diamond")]:
        sel = np.where(kinds == kind)[0]
        fig.add_trace(go.Scatter3d(
            x=Y[sel, 0], y=Y[sel, 1], z=Y[sel, 2],
            mode="markers+text", name=kind,
            marker=dict(size=size, color=color, symbol=sym, opacity=0.9,
                        line=dict(width=0.5, color="#222")),
            text=labels[sel], textposition="top center",
            textfont=dict(size=11, color="#ddd"),
            hovertext=labels[sel], hoverinfo="text",
        ))
    fig.update_layout(
        title="Corpus documents & concept words — shared embedding space "
              "(Qwen3-Embedding-4B, PCA→3D) · click two points for distance",
        template="plotly_dark",
        scene=dict(xaxis_title="PC-1", yaxis_title="PC-2", zaxis_title="PC-3"),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0),
    )

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True, config={"responsive": True})
    open(os.path.join(os.path.dirname(out), ".nojekyll"), "w").close()

    docs_i = np.where(kinds == "document")[0]; con_i = np.where(kinds == "concept")[0]
    sim = {labels[c]: {labels[d]: round(float(SIM[c, d]), 4) for d in docs_i} for c in con_i}
    json.dump(sim, open(os.path.join(os.path.dirname(out), "concept_document_similarity.json"), "w"), indent=2)
    print(f"[reduce] {X.shape[0]} points ({len(docs_i)} docs + {len(con_i)} concepts) -> {out}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--concepts", default="")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--cache", default="data/doc_vectors.npz", help="per-document vector cache")
    ap.add_argument("--chunk-words", type=int, default=300)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    X, labels, kinds = embed_all(args)
    build_html(X, labels, kinds, args.out)

if __name__ == "__main__":
    main()
