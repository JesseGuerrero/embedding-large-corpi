#!/usr/bin/env python3
"""Inject a client-side distance tool into an existing Plotly 3D HTML.

No re-embedding: reads point coordinates straight from Plotly click events.
Click two points -> draws a line between them and labels its midpoint with the
3-D (PCA-projected) distance, plus a small info panel.
"""
import sys

JS = r"""
<script>
(function(){
  function init(){
    var gd = document.querySelector('.plotly-graph-div');
    if(!gd || !gd.on){ return setTimeout(init, 150); }
    var base = gd.data.length, sel = [];
    var panel = document.createElement('div');
    panel.style.cssText = 'position:fixed;top:12px;left:12px;z-index:1000;background:rgba(20,20,28,.92);'
      + 'color:#eee;font:13px/1.5 system-ui,sans-serif;padding:10px 13px;border:1px solid #444;'
      + 'border-radius:8px;max-width:330px;box-shadow:0 2px 10px rgba(0,0,0,.5)';
    document.body.appendChild(panel);

    function lbl(p){ return (p.data.text && p.data.text[p.pointNumber]) || p.data.name; }
    function dist(a,b){ var dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z; return Math.sqrt(dx*dx+dy*dy+dz*dz); }
    function reset(){
      var idx=[]; for(var i=base;i<gd.data.length;i++) idx.push(i);
      if(idx.length) Plotly.deleteTraces(gd, idx);
    }
    function render(){
      if(sel.length===0){ panel.innerHTML='<b>Distance tool</b><br>Click any two points.'; }
      else if(sel.length===1){ panel.innerHTML='<b>Distance tool</b><br>A: <b>'+sel[0].label+'</b><br>Click a second point…'; }
      else{ var d=dist(sel[0],sel[1]);
        panel.innerHTML='<b>Distance tool</b><br>A: <b>'+sel[0].label+'</b><br>B: <b>'+sel[1].label+'</b>'
          +'<hr style="border-color:#444">3-D plot distance: <b style="color:#FFD700">'+d.toFixed(3)+'</b>'
          +'<br><span style="color:#888;font-size:11px">(PCA projection · click a third point to reset)</span>'; }
    }
    function draw(){
      var a=sel[0], b=sel[1], d=dist(a,b);
      var mx=(a.x+b.x)/2, my=(a.y+b.y)/2, mz=(a.z+b.z)/2;
      Plotly.addTraces(gd, [
        {type:'scatter3d', mode:'lines', x:[a.x,b.x], y:[a.y,b.y], z:[a.z,b.z],
         line:{color:'#FFD700', width:5}, showlegend:false, hoverinfo:'skip', name:'distance'},
        {type:'scatter3d', mode:'markers+text', x:[mx], y:[my], z:[mz], text:[d.toFixed(3)],
         textposition:'middle center', textfont:{color:'#FFD700', size:15},
         marker:{size:2, color:'#FFD700'}, showlegend:false, hoverinfo:'skip', name:'distance'}
      ]);
    }
    gd.on('plotly_click', function(e){
      var p=e.points[0]; if(p.curveNumber>=base) return;   // ignore clicks on drawn line/label
      var it={label:lbl(p), x:p.x, y:p.y, z:p.z};
      if(sel.length===2){ sel=[]; reset(); }
      sel.push(it); render(); if(sel.length===2) draw();
    });
    render();
  }
  init();
})();
</script>
"""

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/index.html"
    html = open(path, encoding="utf-8").read()
    if "Distance tool" in html:
        print("[skip] distance tool already present"); return
    html = html.replace("</body>", JS + "\n</body>", 1)
    open(path, "w", encoding="utf-8").write(html)
    print(f"[ok] injected distance tool into {path}")

if __name__ == "__main__":
    main()
