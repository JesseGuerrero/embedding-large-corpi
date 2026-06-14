#!/usr/bin/env python3
"""One-point-per-document embedding + concept words, projected to 3D with PCA,
rendered as an interactive Plotly graph where you can click two points to read
the distance between them.

Each .txt file -> chunked -> chunk embeddings mean-pooled -> one L2-normalized
document vector (labeled by filename). Each concept phrase -> one vector. All
vectors live in the same Qwen3-Embedding space and are projected to 3D via PCA
(appropriate for a small number of points).

Vectors are cached to an .npz so the HTML can be regenerated instantly with
--from-cache (no GPU / no re-embedding).
"""
import argparse, glob, json, os, sys, time
import numpy as np

def chunk_words(text, size):
    w = text.split()
    return [" ".join(w[i:i+size]) for i in range(0, len(w), size)] or [""]

def embed_all(args):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mk = {"torch_dtype": torch.bfloat16} if dev == "cuda" else {}
    m = SentenceTransformer(args.model, trust_remote_code=True, device=dev, model_kwargs=mk)
    m.max_seq_length = args.max_seq

    labels, kinds, vecs = [], [], []
    for f in sorted(glob.glob(os.path.join(args.src, "*.txt"))):
        name = os.path.splitext(os.path.basename(f))[0]
        text = open(f, encoding="utf-8", errors="replace").read()
        chunks = chunk_words(text, args.chunk_words)
        t = time.time()
        emb = m.encode(chunks, batch_size=args.batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
        doc = emb.mean(0); doc = doc / (np.linalg.norm(doc) + 1e-12)
        labels.append(name); kinds.append("document"); vecs.append(doc)
        print(f"[doc] {name}: {len(chunks)} chunks -> 1 vector ({time.time()-t:.0f}s)", file=sys.stderr)

    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    cemb = m.encode(concepts, normalize_embeddings=True, convert_to_numpy=True)
    for w, v in zip(concepts, cemb):
        labels.append(w); kinds.append("concept"); vecs.append(v)
        print(f"[concept] {w}", file=sys.stderr)

    return np.vstack(vecs).astype(np.float32), np.array(labels), np.array(kinds)

# Interactive distance panel injected after the Plotly figure renders.
# {plot_id} is substituted by Plotly; data placeholders are substituted below.
POST_JS = r"""
var gd = document.getElementById('{plot_id}');
var LABELS = __LABELS__, KINDS = __KINDS__, COORDS = __COORDS__, SIM = __SIM__, T2G = __T2G__;
var sel = [], lineIdx = null, baseTraces = gd.data.length;

var panel = document.createElement('div');
panel.id = 'distpanel';
panel.style.cssText = 'position:fixed;top:12px;left:12px;z-index:1000;background:rgba(20,20,28,0.92);'
  + 'color:#eee;font:13px/1.5 system-ui,sans-serif;padding:12px 14px;border:1px solid #444;'
  + 'border-radius:8px;max-width:320px;box-shadow:0 2px 10px rgba(0,0,0,.5)';
document.body.appendChild(panel);

function render(){
  if(sel.length===0){ panel.innerHTML = '<b>Distance tool</b><br>Click any two points to compare them.'; return; }
  if(sel.length===1){ panel.innerHTML = '<b>Distance tool</b><br>A: <b>'+LABELS[sel[0]]+'</b><br>Now click a second point…'; return; }
  var a=sel[0], b=sel[1], cos=SIM[a][b];
  var dx=COORDS[a][0]-COORDS[b][0], dy=COORDS[a][1]-COORDS[b][1], dz=COORDS[a][2]-COORDS[b][2];
  var e3=Math.sqrt(dx*dx+dy*dy+dz*dz);
  panel.innerHTML = '<b>Distance tool</b>'
    + '<br>A: <b>'+LABELS[a]+'</b>'
    + '<br>B: <b>'+LABELS[b]+'</b><hr style="border-color:#444">'
    + 'Cosine similarity <span style="color:#9bd">(full 2560-D)</span>: <b>'+cos.toFixed(4)+'</b>'
    + '<br>Cosine distance (1−cos): <b>'+(1-cos).toFixed(4)+'</b>'
    + '<br>Angle: <b>'+(Math.acos(Math.max(-1,Math.min(1,cos)))*180/Math.PI).toFixed(1)+'°</b>'
    + '<br><span style="color:#888">3-D plot distance (PCA, illustrative): '+e3.toFixed(3)+'</span>'
    + '<br><br><span style="color:#888;font-size:11px">Click a third point to start over.</span>';
}

function clearLine(){ if(lineIdx!==null){ Plotly.deleteTraces(gd,[lineIdx]); lineIdx=null; } }
function drawLine(){
  var a=sel[0], b=sel[1];
  Plotly.addTraces(gd, {
    x:[COORDS[a][0],COORDS[b][0]], y:[COORDS[a][1],COORDS[b][1]], z:[COORDS[a][2],COORDS[b][2]],
    type:'scatter3d', mode:'lines+markers', name:'selection', showlegend:false, hoverinfo:'skip',
    line:{color:'#FFD700',width:5}, marker:{size:6,color:'#FFD700',symbol:'circle'}
  }).then(function(){ lineIdx = gd.data.length-1; });
}

gd.on('plotly_click', function(d){
  var pt=d.points[0]; if(pt.curveNumber>=baseTraces) return;   // ignore clicks on the line
  var g=T2G[pt.curveNumber][pt.pointNumber];
  if(sel.length===2){ sel=[]; clearLine(); }
  if(sel.indexOf(g)===-1) sel.push(g);
  render();
  if(sel.length===2) drawLine();
});
render();
"""

def build_html(X, labels, kinds, out):
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go

    # full-dim cosine matrix (vectors already L2-normalized => dot = cosine)
    SIM = (X @ X.T).astype(float)
    Y = PCA(n_components=3, random_state=42).fit_transform(X)

    fig = go.Figure()
    trace_global = []
    for kind, color, size, sym in [("document", "#4C78A8", 7, "circle"),
                                   ("concept", "#E45756", 6, "diamond")]:
        sel = np.where(kinds == kind)[0]
        trace_global.append([int(i) for i in sel])
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

    js = (POST_JS
          .replace("__LABELS__", json.dumps(list(map(str, labels))))
          .replace("__KINDS__", json.dumps(list(map(str, kinds))))
          .replace("__COORDS__", json.dumps([[round(float(v), 5) for v in row] for row in Y]))
          .replace("__SIM__", json.dumps([[round(float(v), 5) for v in row] for row in SIM]))
          .replace("__T2G__", json.dumps(trace_global)))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"responsive": True}, post_script=js)
    open(os.path.join(os.path.dirname(out), ".nojekyll"), "w").close()

    # also dump concept->document similarity for the record
    docs_i = np.where(kinds == "document")[0]; con_i = np.where(kinds == "concept")[0]
    sim = {labels[c]: {labels[d]: round(float(SIM[c, d]), 4) for d in docs_i} for c in con_i}
    json.dump(sim, open(os.path.join(os.path.dirname(out), "concept_document_similarity.json"), "w"), indent=2)
    print(f"[reduce] {X.shape[0]} points -> {out}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--concepts", default="")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--cache", default="data/doc_vectors.npz")
    ap.add_argument("--from-cache", action="store_true", help="skip embedding; rebuild HTML from cache")
    ap.add_argument("--chunk-words", type=int, default=300)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    if args.from_cache:
        d = np.load(args.cache, allow_pickle=True)
        X, labels, kinds = d["X"], d["labels"], d["kinds"]
        print(f"[cache] loaded {X.shape} from {args.cache}", file=sys.stderr)
    else:
        X, labels, kinds = embed_all(args)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        np.savez(args.cache, X=X, labels=labels, kinds=kinds)
        print(f"[cache] saved -> {args.cache}", file=sys.stderr)

    build_html(X, labels, kinds, args.out)

if __name__ == "__main__":
    main()
