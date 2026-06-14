#!/usr/bin/env python3
"""Inject a client-side distance tool into an existing Plotly 3D HTML.

No re-embedding: reads point coordinates straight from Plotly click events.
Click two points -> a line is drawn between them with the (PCA-projected)
distance labeled at its midpoint, plus a small info panel.

Implementation note: uses ONE persistent selection trace added at init and
updates it with Plotly.restyle. It does NOT add/remove traces per click —
doing that on a WebGL (scatter3d) plot triggers a re-render loop that
overflows the stack ("Maximum call stack size exceeded").
"""
import re, sys

START = "<!--distance-tool-start-->"
END = "<!--distance-tool-end-->"

JS = START + r"""
<script>
(function(){
  function init(){
    var gd = document.querySelector('.plotly-graph-div');
    if(!gd || !gd.on || !window.Plotly){ return setTimeout(init, 150); }
    var base = gd.data.length, sel = [], selLine = -1, selLbl = -1, busy = false, zoff = 0;

    // vertical offset for the label, scaled to the plot's z-range, so the
    // number floats above the line instead of overlapping it.
    (function(){
      var zmin=Infinity, zmax=-Infinity;
      for(var i=0;i<base;i++){ var z=gd.data[i].z||[]; for(var j=0;j<z.length;j++){ if(z[j]<zmin)zmin=z[j]; if(z[j]>zmax)zmax=z[j]; } }
      if(isFinite(zmin) && isFinite(zmax)) zoff=(zmax-zmin)*0.05;
    })();

    // Two persistent traces: the selection line, and a midpoint distance label.
    Plotly.addTraces(gd, [
      {type:'scatter3d', mode:'lines', x:[], y:[], z:[],
       line:{color:'#FFD700', width:5}, hoverinfo:'skip', showlegend:false, name:'selection'},
      {type:'scatter3d', mode:'text', x:[], y:[], z:[], text:[],
       textposition:'top center', textfont:{color:'#FFD700', size:16},
       hoverinfo:'skip', showlegend:false, name:'distance'}
    ]).then(function(){ selLbl = gd.data.length - 1; selLine = gd.data.length - 2; });

    var panel = document.createElement('div');
    panel.style.cssText = 'position:fixed;top:12px;left:12px;z-index:1000;background:rgba(20,20,28,.92);'
      + 'color:#eee;font:13px/1.5 system-ui,sans-serif;padding:10px 13px;border:1px solid #444;'
      + 'border-radius:8px;max-width:330px;box-shadow:0 2px 10px rgba(0,0,0,.5)';
    document.body.appendChild(panel);

    function lbl(p){ return (p.data.text && p.data.text[p.pointNumber]) || p.data.name; }
    function dist(a,b){ var dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z; return Math.sqrt(dx*dx+dy*dy+dz*dz); }

    function clearLine(){
      if(selLine<0 || busy) return; busy=true;
      Plotly.restyle(gd, {x:[[],[]], y:[[],[]], z:[[],[]], text:[[],[]]}, [selLine, selLbl]).then(function(){ busy=false; });
    }
    function showLine(){
      if(selLine<0) return;
      var a=sel[0], b=sel[1], d=dist(a,b);
      var mx=(a.x+b.x)/2, my=(a.y+b.y)/2, mz=(a.z+b.z)/2;
      busy=true;
      Plotly.restyle(gd, {x:[[a.x,b.x],[mx]], y:[[a.y,b.y],[my]], z:[[a.z,b.z],[mz+zoff]],
        text:[[], [d.toFixed(3)]]}, [selLine, selLbl]).then(function(){ busy=false; });
    }
    function render(){
      if(sel.length===0){ panel.innerHTML='<b>Distance tool</b><br>Click any two points.'; }
      else if(sel.length===1){ panel.innerHTML='<b>Distance tool</b><br>A: <b>'+sel[0].label+'</b><br>Click a second point…'; }
      else{ var d=dist(sel[0],sel[1]);
        panel.innerHTML='<b>Distance tool</b><br>A: <b>'+sel[0].label+'</b><br>B: <b>'+sel[1].label+'</b>'
          +'<hr style="border-color:#444">3-D plot distance: <b style="color:#FFD700">'+d.toFixed(3)+'</b>'
          +'<br><span style="color:#888;font-size:11px">(PCA projection · click a third point to reset)</span>'; }
    }

    gd.on('plotly_click', function(e){
      var p = e.points[0];
      if(p.curveNumber === selLine || p.curveNumber === selLbl) return;   // ignore selection traces
      var it = {label:lbl(p), x:p.x, y:p.y, z:p.z};
      if(sel.length===2){ sel=[]; clearLine(); }
      sel.push(it); render();
      if(sel.length===2) showLine();
    });
    render();
  }
  init();
})();
</script>
""" + END

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/index.html"
    html = open(path, encoding="utf-8").read()
    # remove any prior injected block (markers, or the older unmarked version)
    html = re.sub(re.escape(START) + r".*?" + re.escape(END), "", html, flags=re.S)
    html = re.sub(r"<script>\s*\(function\(\)\{\s*function init\(\).*?\}\)\(\);\s*</script>", "", html, flags=re.S)
    html = html.replace("</body>", JS + "\n</body>", 1)
    open(path, "w", encoding="utf-8").write(html)
    print(f"[ok] injected distance tool into {path}")

if __name__ == "__main__":
    main()
