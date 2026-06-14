#!/usr/bin/env python3
"""One-point-per-document embedding + concept words, projected to 3D with PCA.

Each .txt file -> chunked -> chunk embeddings mean-pooled -> one L2-normalized
document vector (labeled by filename). Each concept phrase -> one vector.
All vectors live in the same Qwen3-Embedding space, projected to 3D via PCA
(appropriate for a small number of points), rendered as an interactive,
labeled Plotly scatter -> docs/index.html.
"""
import argparse, glob, json, os, sys, time
import numpy as np

def chunk_words(text, size):
    w = text.split()
    return [" ".join(w[i:i+size]) for i in range(0, len(w), size)] or [""]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of .txt files")
    ap.add_argument("--concepts", required=True, help="comma-separated concept phrases")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--chunk-words", type=int, default=300)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mk = {"torch_dtype": torch.bfloat16} if dev == "cuda" else {}
    m = SentenceTransformer(args.model, trust_remote_code=True, device=dev, model_kwargs=mk)
    m.max_seq_length = args.max_seq

    labels, kinds, vecs = [], [], []

    # --- documents: chunk -> embed -> mean-pool ---
    files = sorted(glob.glob(os.path.join(args.src, "*.txt")))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        text = open(f, encoding="utf-8", errors="replace").read()
        chunks = chunk_words(text, args.chunk_words)
        t = time.time()
        emb = m.encode(chunks, batch_size=args.batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
        doc = emb.mean(0)
        doc = doc / (np.linalg.norm(doc) + 1e-12)
        labels.append(name); kinds.append("document"); vecs.append(doc)
        print(f"[doc] {name}: {len(chunks)} chunks -> 1 vector ({time.time()-t:.0f}s)", file=sys.stderr)

    # --- concept words: one vector each ---
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    cemb = m.encode(concepts, normalize_embeddings=True, convert_to_numpy=True)
    for w, v in zip(concepts, cemb):
        labels.append(w); kinds.append("concept"); vecs.append(v)
        print(f"[concept] {w}", file=sys.stderr)

    X = np.vstack(vecs).astype(np.float32)
    kinds = np.array(kinds); labels = np.array(labels)
    print(f"[reduce] {X.shape[0]} points, dim {X.shape[1]} -> PCA(3)", file=sys.stderr)

    # --- full-dim concept<->document cosine similarity (the quantitative signal) ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    docs_i = np.where(kinds == "document")[0]
    con_i = np.where(kinds == "concept")[0]
    sim = {labels[c]: {labels[d]: round(float(X[c] @ X[d]), 4) for d in docs_i} for c in con_i}
    json.dump(sim, open(os.path.join(os.path.dirname(args.out), "concept_document_similarity.json"), "w"), indent=2)
    print("[reduce] concept -> nearest document (full-dim cosine):", file=sys.stderr)
    for c in con_i:
        order = sorted(docs_i, key=lambda d: -float(X[c] @ X[d]))
        print(f"   {labels[c]:20s} -> {labels[order[0]]} ({float(X[c]@X[order[0]]):.3f})", file=sys.stderr)

    # --- PCA to 3D ---
    Y = PCA(n_components=3, random_state=42).fit_transform(X)

    # --- plot: labeled markers, documents vs concepts distinguished ---
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
        title="Corpus documents & concept words — shared embedding space (Qwen3-Embedding-4B, PCA→3D)",
        template="plotly_dark",
        scene=dict(xaxis_title="PC-1", yaxis_title="PC-2", zaxis_title="PC-3"),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(args.out, include_plotlyjs="cdn", full_html=True, config={"responsive": True})
    open(os.path.join(os.path.dirname(args.out), ".nojekyll"), "w").close()
    print(f"[reduce] wrote {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
