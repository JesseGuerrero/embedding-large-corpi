#!/usr/bin/env python3
"""One-point-per-document embedding + concept words, projected to 3D with PCA,
rendered as an interactive Plotly graph with Distance and Analogy tools and
per-point show/hide toggles (persisted to browser localStorage).

Each .txt file -> chunked -> chunk embeddings mean-pooled -> one L2-normalized
document vector (labeled by filename). Each concept phrase -> one vector. All
vectors share the same Qwen3-Embedding space, projected to 3D via PCA.

Document vectors are cached per-file in an .npz (the slow part). Concept words
are cheap and always re-embedded, so adding/removing words is near-instant once
the document cache exists.
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

    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    clabels = list(cache.keys())
    np.savez(args.cache, labels=np.array(clabels), vecs=np.vstack([cache[k] for k in clabels]))
    print(f"[cache] saved {len(clabels)} document vectors -> {args.cache}", file=sys.stderr)

    labels, kinds, vecs = [], [], []
    for name in names:
        labels.append(name); kinds.append("document"); vecs.append(cache[name])
    if concepts:
        cemb = model.encode(concepts, normalize_embeddings=True, convert_to_numpy=True)
        for w, v in zip(concepts, cemb):
            labels.append(w); kinds.append("concept"); vecs.append(v.astype(np.float32))
            print(f"[concept] {w}", file=sys.stderr)

    return np.vstack(vecs).astype(np.float32), np.array(labels), np.array(kinds)

# Injected after the figure renders. {plot_id} -> Plotly; __DATA__ -> build_html.
# Tools: Distance (two points -> cosine distance) and Analogy (A,B,C -> D).
# Per-point show/hide toggles persist to localStorage. Clicks map to the global
# index via each point's customdata, so filtering never breaks the tools.
# Only one-time addTraces + restyle is used (no per-click trace add/remove).
TOOL_JS = r"""
var gd = document.getElementById('{plot_id}');
var LABELS=__LABELS__, KINDS=__KINDS__, COORDS=__COORDS__, VECS=__VECS__;
var base = gd.data.length, IDX={}, busy=false, mode='distance', sel=[];

var STORE='emb_excluded_v1', excluded={};
try{ var raw=localStorage.getItem(STORE); if(raw) JSON.parse(raw).forEach(function(l){excluded[l]=1;}); }catch(e){}
function saveVis(){ try{ localStorage.setItem(STORE, JSON.stringify(Object.keys(excluded))); }catch(e){} }
function isVis(i){ return !excluded[LABELS[i]]; }

var zmin=Infinity, zmax=-Infinity;
for(var i=0;i<COORDS.length;i++){ var z=COORDS[i][2]; if(z<zmin)zmin=z; if(z>zmax)zmax=z; }
var zoff=(isFinite(zmin)&&isFinite(zmax))?(zmax-zmin)*0.05:0.1;

function dot(a,b){ var s=0; for(var i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
function cos(i,j){ return dot(VECS[i],VECS[j]); }
function addv(a,b){ var r=new Array(a.length); for(var i=0;i<a.length;i++) r[i]=a[i]+b[i]; return r; }
function subv(a,b){ var r=new Array(a.length); for(var i=0;i<a.length;i++) r[i]=a[i]-b[i]; return r; }
function normv(a){ var n=Math.sqrt(dot(a,a))||1, r=new Array(a.length); for(var i=0;i<a.length;i++) r[i]=a[i]/n; return r; }
function kindIdx(k){ var a=[]; for(var i=0;i<LABELS.length;i++) if(KINDS[i]===k) a.push(i); return a; }
var DOCS=kindIdx('document'), CONS=kindIdx('concept');

function redraw(){
  function pack(idxs){ var x=[],y=[],z=[],t=[],cd=[];
    for(var k=0;k<idxs.length;k++){ var i=idxs[k]; if(!isVis(i))continue;
      x.push(COORDS[i][0]); y.push(COORDS[i][1]); z.push(COORDS[i][2]); t.push(LABELS[i]); cd.push(i); }
    return [x,y,z,t,cd]; }
  var d=pack(DOCS), c=pack(CONS);
  Plotly.restyle(gd,{x:[d[0],c[0]],y:[d[1],c[1]],z:[d[2],c[2]],text:[d[3],c[3]],hovertext:[d[3],c[3]],customdata:[d[4],c[4]]},[0,1]);
}

Plotly.addTraces(gd, [
  {name:'distline', type:'scatter3d', mode:'lines', x:[],y:[],z:[], line:{color:'#FFD700',width:5}, hoverinfo:'skip', showlegend:false},
  {name:'distlabel', type:'scatter3d', mode:'text', x:[],y:[],z:[], text:[], textposition:'top center', textfont:{color:'#FFD700',size:36}, hoverinfo:'skip', showlegend:false},
  {name:'A→B', type:'scatter3d', mode:'lines', x:[],y:[],z:[], line:{color:'#2ee6a6',width:7}, hoverinfo:'skip', showlegend:false},
  {name:'C→D', type:'scatter3d', mode:'lines', x:[],y:[],z:[], line:{color:'#ff8c42',width:7}, hoverinfo:'skip', showlegend:false},
  {name:'sel', type:'scatter3d', mode:'markers', x:[],y:[],z:[], marker:{size:16,color:'rgba(255,255,255,0.45)'}, hoverinfo:'skip', showlegend:false}
]).then(function(){ var n=gd.data.length; IDX.sel=n-1; IDX.cd=n-2; IDX.ab=n-3; IDX.dlabel=n-4; IDX.dline=n-5; redraw(); render(); });

// ---- panels ----
function mkbtn(t){ var b=document.createElement('button'); b.textContent=t;
  b.style.cssText='margin-right:6px;padding:4px 11px;border-radius:6px;border:1px solid #666;background:#2a2a36;color:#eee;cursor:pointer;font:12px system-ui'; return b; }

var panel=document.createElement('div');
panel.style.cssText='position:fixed;top:12px;left:12px;z-index:1000;background:rgba(20,20,28,.93);color:#eee;'
  +'font:13px/1.5 system-ui,sans-serif;padding:12px 14px;border:1px solid #444;border-radius:8px;max-width:350px;box-shadow:0 2px 12px rgba(0,0,0,.55)';
document.body.appendChild(panel);
var bD=mkbtn('Distance'), bA=mkbtn('Analogy'), bar=document.createElement('div');
bar.style.marginBottom='8px'; bar.appendChild(bD); bar.appendChild(bA); panel.appendChild(bar);
var info=document.createElement('div'); panel.appendChild(info);
bD.onclick=function(){ setMode('distance'); }; bA.onclick=function(){ setMode('analogy'); };

var tp=document.createElement('div');
tp.style.cssText='position:fixed;top:12px;right:12px;z-index:1000;background:rgba(20,20,28,.93);color:#eee;'
  +'font:12px/1.45 system-ui,sans-serif;padding:10px 12px;border:1px solid #444;border-radius:8px;'
  +'max-height:86vh;overflow:auto;min-width:210px;box-shadow:0 2px 12px rgba(0,0,0,.55)';
document.body.appendChild(tp);
var th=document.createElement('div'); th.innerHTML='<b>Show / hide points</b>'; th.style.marginBottom='6px'; tp.appendChild(th);
var ctl=document.createElement('div'); ctl.style.marginBottom='4px';
var bAll=mkbtn('All'), bNone=mkbtn('None'); ctl.appendChild(bAll); ctl.appendChild(bNone); tp.appendChild(ctl);
bAll.onclick=function(){ excluded={}; afterToggle(); };
bNone.onclick=function(){ excluded={}; for(var i=0;i<LABELS.length;i++) excluded[LABELS[i]]=1; afterToggle(); };
function addGroup(title, idxs){
  var h=document.createElement('div'); h.textContent=title; h.style.cssText='margin:7px 0 2px;color:#9bd;font-weight:bold'; tp.appendChild(h);
  idxs.forEach(function(i){
    var row=document.createElement('label'); row.style.cssText='display:block;cursor:pointer;white-space:nowrap';
    var cb=document.createElement('input'); cb.type='checkbox'; cb.checked=isVis(i); cb.style.marginRight='6px'; cb.setAttribute('data-i', i);
    cb.onchange=function(){ if(cb.checked) delete excluded[LABELS[i]]; else excluded[LABELS[i]]=1; afterToggle(); };
    row.appendChild(cb); row.appendChild(document.createTextNode(LABELS[i])); tp.appendChild(row);
  });
}
addGroup('Documents', DOCS); addGroup('Concepts', CONS);
function syncChecks(){ var cbs=tp.querySelectorAll('input[type=checkbox]');
  for(var k=0;k<cbs.length;k++){ var i=+cbs[k].getAttribute('data-i'); cbs[k].checked=isVis(i); } }
function afterToggle(){ saveVis(); sel=[]; window.__D=null; syncChecks(); clearAll(); redraw(); render(); }

function setMode(m){ mode=m; sel=[]; window.__D=null; clearAll();
  bD.style.background=m=='distance'?'#4356c0':'#2a2a36'; bA.style.background=m=='analogy'?'#4356c0':'#2a2a36'; render(); }

function coordsOf(arr){ var x=[],y=[],z=[]; for(var i=0;i<arr.length;i++){ x.push(COORDS[arr[i]][0]); y.push(COORDS[arr[i]][1]); z.push(COORDS[arr[i]][2]); } return [x,y,z]; }
function clearAll(){ if(IDX.sel===undefined||busy) return; busy=true;
  Plotly.restyle(gd,{x:[[],[],[],[],[]],y:[[],[],[],[],[]],z:[[],[],[],[],[]],text:[[],[],[],[],[]]},
    [IDX.dline,IDX.dlabel,IDX.ab,IDX.cd,IDX.sel]).then(function(){busy=false;}); }

function predict(A,B,C){
  var t=normv(addv(VECS[C],subv(VECS[B],VECS[A]))), ex={}; ex[A]=ex[B]=ex[C]=1;
  var r=[]; for(var i=0;i<VECS.length;i++){ if(ex[i]||!isVis(i))continue; r.push([dot(t,VECS[i]),i]); }
  r.sort(function(a,b){return b[0]-a[0];}); return r;
}

function drawDistance(){
  if(IDX.sel===undefined||busy) return; busy=true;
  var hc=coordsOf(sel), lx=[],ly=[],lz=[],tx=[],ty=[],tz=[],tt=[];
  if(sel.length===2){ var a=sel[0],b=sel[1],ca=COORDS[a],cb=COORDS[b];
    lx=[ca[0],cb[0]]; ly=[ca[1],cb[1]]; lz=[ca[2],cb[2]];
    tx=[(ca[0]+cb[0])/2]; ty=[(ca[1]+cb[1])/2]; tz=[(ca[2]+cb[2])/2+zoff]; tt=[(1-cos(a,b)).toFixed(3)]; }
  Plotly.restyle(gd,{x:[lx,tx,hc[0]],y:[ly,ty,hc[1]],z:[lz,tz,hc[2]],text:[[],tt,[]]},
    [IDX.dline,IDX.dlabel,IDX.sel]).then(function(){busy=false;});
}
function drawAnalogy(){
  if(IDX.sel===undefined||busy) return; busy=true;
  var abx=[],aby=[],abz=[],cdx=[],cdy=[],cdz=[],hi=sel.slice();
  if(sel.length>=2){ var A=sel[0],B=sel[1]; abx=[COORDS[A][0],COORDS[B][0]]; aby=[COORDS[A][1],COORDS[B][1]]; abz=[COORDS[A][2],COORDS[B][2]]; }
  if(sel.length===3 && window.__D!=null){ var C=sel[2], D=window.__D;
    cdx=[COORDS[C][0],COORDS[D][0]]; cdy=[COORDS[C][1],COORDS[D][1]]; cdz=[COORDS[C][2],COORDS[D][2]]; hi=[sel[0],sel[1],sel[2],D]; }
  var hc=coordsOf(hi);
  Plotly.restyle(gd,{x:[abx,cdx,hc[0]],y:[aby,cdy,hc[1]],z:[abz,cdz,hc[2]]},[IDX.ab,IDX.cd,IDX.sel]).then(function(){busy=false;});
}

function render(){
  if(mode==='distance'){
    if(sel.length===0) info.innerHTML='<b>Distance</b><br>Click any two points to measure them.';
    else if(sel.length===1) info.innerHTML='<b>Distance</b><br>A: <b>'+LABELS[sel[0]]+'</b><br>Click a second point…';
    else{ var a=sel[0],b=sel[1],cs=cos(a,b),dx=COORDS[a][0]-COORDS[b][0],dy=COORDS[a][1]-COORDS[b][1],dz=COORDS[a][2]-COORDS[b][2];
      info.innerHTML='<b>Distance</b><br>A: <b>'+LABELS[a]+'</b><br>B: <b>'+LABELS[b]+'</b><hr style="border-color:#444">'
        +'Cosine distance (1−cos, full 2560-D): <b style="color:#FFD700">'+(1-cs).toFixed(3)+'</b><br>'
        +'Cosine similarity: <b>'+cs.toFixed(3)+'</b> &nbsp; Angle: <b>'+(Math.acos(Math.max(-1,Math.min(1,cs)))*180/Math.PI).toFixed(1)+'°</b><br>'
        +'<span style="color:#888">3-D plot distance: '+Math.sqrt(dx*dx+dy*dy+dz*dz).toFixed(3)+'</span>'
        +'<br><span style="color:#888;font-size:11px">Click a third point to reset.</span>'; }
  } else {
    if(sel.length===0) info.innerHTML='<b>Analogy</b> &nbsp;<span style="color:#2ee6a6">A→B</span> : <span style="color:#ff8c42">C→D</span><br>Click <b>A</b>, then <b>B</b> (defines the relation), then <b>C</b>.';
    else if(sel.length===1) info.innerHTML='<b>Analogy</b><br>A: <b>'+LABELS[sel[0]]+'</b><br>Click <b>B</b> (e.g. A=king → B=queen).';
    else if(sel.length===2) info.innerHTML='<b>Analogy</b><br><span style="color:#2ee6a6">'+LABELS[sel[0]]+' → '+LABELS[sel[1]]+'</span><br>Now click <b>C</b> to find its match.';
    else{ var R=predict(sel[0],sel[1],sel[2]);
      if(!R.length){ window.__D=null; info.innerHTML='<b>Analogy</b><br>No other visible points to match against.'; return; }
      window.__D=R[0][1];
      var top=R.slice(0,3).map(function(p){return LABELS[p[1]]+' ('+p[0].toFixed(3)+')';}).join(', ');
      info.innerHTML='<b>Analogy</b><br><span style="color:#2ee6a6">'+LABELS[sel[0]]+' : '+LABELS[sel[1]]+'</span> :: '
        +'<span style="color:#ff8c42">'+LABELS[sel[2]]+' : <b>'+LABELS[R[0][1]]+'</b></span> ('+R[0][0].toFixed(3)+')'
        +'<hr style="border-color:#444"><span style="color:#888;font-size:12px">best matches: '+top+'</span>'
        +'<br><span style="color:#888;font-size:11px">Click a fourth point to reset.</span>'; }
  }
}

gd.on('plotly_click', function(e){
  var p=e.points[0]; if(p.curveNumber>=base) return;       // ignore tool traces
  var g=p.customdata; if(g==null) return;
  var cap = mode==='distance'?2:3;
  if(sel.length>=cap){ sel=[]; window.__D=null; clearAll(); }
  if(sel.indexOf(g)===-1) sel.push(g);
  render();
  if(mode==='distance') drawDistance();
  else { if(sel.length===3) render(); drawAnalogy(); }   // ensure __D set before drawing
});
setMode('distance');

// ---- analogy click list (bottom-left): up to 4 clicked point names, analogy mode only ----
var alist=[], aidx=[], alistEl=document.createElement('div');
alistEl.style.cssText='position:fixed;bottom:12px;left:12px;z-index:1000;background:rgba(20,20,28,.93);color:#eee;'
  +'font:13px/1.5 system-ui,sans-serif;padding:10px 12px;border:1px solid #444;border-radius:8px;'
  +'white-space:nowrap;max-width:none;box-shadow:0 2px 12px rgba(0,0,0,.55);display:none';
document.body.appendChild(alistEl);
function renderAList(){
  if(mode!=='analogy'){ alistEl.style.display='none'; return; }
  alistEl.style.display='block';
  var ph=['[POINT_1]','[POINT_2]','[POINT_3]','[POINT_4]'];
  var p=ph.map(function(d,i){ return alist[i]!=null ? alist[i] : d; });
  alistEl.innerHTML='<b>'+p[0]+'</b> is a <b>'+p[1]+'</b> as a <b>'+p[2]+'</b> is a <b>'+p[3]+'</b>.';
}
var _alastG=null;
gd.on('plotly_click', function(e){
  if(mode!=='analogy') return;
  var p=e.points[0]; if(p.curveNumber>=base) return;
  var g=p.customdata; if(g==null) return;
  if(g===_alastG) return;   // ignore gl3d double-fire (same point repeated), timing-independent
  _alastG=g;
  if(alist.length>=4){ alist=[]; aidx=[]; }
  alist.push(LABELS[g]); aidx.push(g);
  if(aidx.length===3){ var R=predict(aidx[0],aidx[1],aidx[2]); if(R.length) alist.push(LABELS[R[0][1]]); }  // P4 = analogy result
  renderAList();
});
bD.addEventListener('click', function(){ _alastG=null; renderAList(); });
bA.addEventListener('click', function(){ alist=[]; aidx=[]; _alastG=null; renderAList(); });
renderAList();

// ---- pad axis ranges so point text labels aren't clipped at the scene edges ----
(function(){
  var mn=[Infinity,Infinity,Infinity], mx=[-Infinity,-Infinity,-Infinity];
  for(var i=0;i<COORDS.length;i++) for(var a=0;a<3;a++){ var v=COORDS[i][a]; if(v<mn[a])mn[a]=v; if(v>mx[a])mx[a]=v; }
  var maxsp=Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])||1, pad=maxsp*0.35, rng={};
  ['x','y','z'].forEach(function(ax,a){ rng['scene.'+ax+'axis.range']=[mn[a]-pad, mx[a]+pad]; });
  rng['scene.aspectmode']='data';
  Plotly.relayout(gd, rng);
})();
"""

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
            customdata=[int(i) for i in sel],
            hovertext=labels[sel], hoverinfo="text",
        ))
    fig.update_layout(
        title="Corpus documents & concept words — shared embedding space "
              "(Qwen3-Embedding-4B, PCA→3D) · Distance & Analogy tools",
        template="plotly_dark",
        scene=dict(xaxis_title="PC-1", yaxis_title="PC-2", zaxis_title="PC-3"),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0),
    )

    js = (TOOL_JS
          .replace("__LABELS__", json.dumps(list(map(str, labels))))
          .replace("__KINDS__", json.dumps(list(map(str, kinds))))
          .replace("__COORDS__", json.dumps([[round(float(v), 5) for v in row] for row in Y]))
          .replace("__VECS__", json.dumps([[round(float(v), 5) for v in row] for row in X])))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"responsive": True}, post_script=js)
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
    ap.add_argument("--include", default="",
                    help="comma-separated labels to keep (others dropped); tolerant match")
    args = ap.parse_args()

    X, labels, kinds = embed_all(args)

    if args.include:
        import re
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        keep = {norm(w) for w in args.include.split(",") if w.strip()}
        mask = np.array([norm(l) in keep for l in labels])
        X, labels, kinds = X[mask], labels[mask], kinds[mask]
        print(f"[filter] kept {int(mask.sum())} points: {list(labels)}", file=sys.stderr)

    build_html(X, labels, kinds, args.out)

if __name__ == "__main__":
    main()
