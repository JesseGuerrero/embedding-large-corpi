#!/usr/bin/env python3
"""Reduce all embedded corpora into ONE shared 3D space and render an
interactive, GitHub-Pages-ready Plotly scatter.

Fits PCA->UMAP on the COMBINED set of every corpus found in data/embeddings,
so coordinates are directly comparable across corpora. Colors by corpus when
more than one is present, otherwise by sub-group (series, for the Schaff set).

Outputs:
  docs/index.html        -- rotatable 3D scatter (self-contained, CDN plotly)
  docs/similarity.json   -- centroid cosine-similarity matrix between groups
"""
import argparse, glob, json, os, sys
import numpy as np

def stable_subsample(n, cap, seed=0):
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))

def load_all(emb_dir, chunk_dir):
    rows = []  # (corpus, vectors, meta_list)
    for npy in sorted(glob.glob(os.path.join(emb_dir, "*.npy"))):
        corpus = os.path.splitext(os.path.basename(npy))[0]
        cj = os.path.join(chunk_dir, f"{corpus}.jsonl")
        if not os.path.exists(cj):
            print(f"[warn] no chunks for {corpus}, skipping", file=sys.stderr); continue
        vecs = np.load(npy)
        meta = [json.loads(l) for l in open(cj, encoding="utf-8")]
        assert len(meta) == len(vecs), f"{corpus}: {len(meta)} meta vs {len(vecs)} vecs"
        rows.append((corpus, vecs, meta))
        print(f"[load] {corpus}: {vecs.shape}", file=sys.stderr)
    if not rows:
        sys.exit("No embeddings found. Run embed.py first.")
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--chunk-dir", default="data/chunks")
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--viz-cap", type=int, default=24000, help="max points drawn")
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.decomposition import PCA
    import umap
    import plotly.graph_objects as go

    rows = load_all(args.emb_dir, args.chunk_dir)
    multi = len(rows) > 1

    X = np.vstack([v for _, v, _ in rows])
    corpora, groups, hover = [], [], []
    for corpus, vecs, meta in rows:
        for r in meta:
            corpora.append(corpus)
            groups.append(corpus if multi else r.get("series", corpus))
            snip = r["text"][:160].replace("\n", " ")
            hover.append(f"<b>{r.get('title', r['volume'])}</b><br>{snip}…")
    corpora = np.array(corpora); groups = np.array(groups)
    print(f"[reduce] total {X.shape[0]} vectors, dim {X.shape[1]}", file=sys.stderr)

    # --- centroid cosine-similarity between groups (computed on FULL set) ---
    sim = {}
    uniq = sorted(set(groups.tolist()))
    cents = {}
    for g in uniq:
        c = X[groups == g].mean(0)
        c = c / (np.linalg.norm(c) + 1e-12)
        cents[g] = c
    for a in uniq:
        sim[a] = {b: round(float(cents[a] @ cents[b]), 4) for b in uniq}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(sim, open(os.path.join(os.path.dirname(args.out), "similarity.json"), "w"), indent=2)
    print("[reduce] centroid cosine similarity:", file=sys.stderr)
    for a in uniq:
        print("   ", a, sim[a], file=sys.stderr)

    # --- PCA -> UMAP to 3D on the combined, comparable space ---
    pca_dim = min(args.pca_dim, X.shape[1], X.shape[0])
    Xp = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(X)
    print(f"[reduce] PCA -> {Xp.shape}; running UMAP to 3D...", file=sys.stderr)
    reducer = umap.UMAP(n_components=3, n_neighbors=args.umap_neighbors,
                        min_dist=args.umap_min_dist, metric="cosine",
                        random_state=args.seed, verbose=True)
    Y = reducer.fit_transform(Xp)

    # --- subsample for the browser, stratified by group ---
    keep = []
    for g in uniq:
        idx = np.where(groups == g)[0]
        share = max(1, int(round(args.viz_cap * len(idx) / len(groups))))
        keep.append(idx[stable_subsample(len(idx), share, seed=args.seed)])
    keep = np.sort(np.concatenate(keep))
    print(f"[reduce] plotting {len(keep)}/{len(groups)} points", file=sys.stderr)

    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
               "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
    fig = go.Figure()
    for i, g in enumerate(uniq):
        sel = keep[groups[keep] == g]
        fig.add_trace(go.Scatter3d(
            x=Y[sel, 0], y=Y[sel, 1], z=Y[sel, 2],
            mode="markers", name=f"{g} ({len(sel)})",
            marker=dict(size=1.8, color=palette[i % len(palette)], opacity=0.7),
            text=[hover[k] for k in sel], hoverinfo="text",
        ))
    title = ("Embedding latent space — multiple corpora"
             if multi else "Schaff corpus — embedding latent space (by series)")
    fig.update_layout(
        title=title, template="plotly_dark",
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(args.out, include_plotlyjs="cdn", full_html=True,
                   config={"responsive": True})
    open(os.path.join(os.path.dirname(args.out), ".nojekyll"), "w").close()
    print(f"[reduce] wrote {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
