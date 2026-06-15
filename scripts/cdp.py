#!/usr/bin/env python3
"""CDP JS-eval helper for the Digital Twin browser test rig (scripts/dt_browser.sh).

Runs a JavaScript expression in the localhost:3036 tab over the Chrome DevTools
Protocol and prints the JSON-serialized result (or {"error": ...}). Used for
reading/driving the ArcGIS app (e.g. window.app.selectService('buildingView'),
reading view.camera / ground.navigationConstraint). Input/screenshots are done
with xdotool/ffmpeg in the shell wrapper, not here.

Invoke via the wrapper (it provides websocket-client through uv):
    scripts/dt_browser.sh js '<expr or .js file>'
"""
import argparse, json, sys, urllib.request
import websocket  # provided by `uv run --with websocket-client`


def tabs(port):
    with urllib.request.urlopen(f"http://localhost:{port}/json/list", timeout=10) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9222)
    ap.add_argument("--match", default="localhost:3036", help="substring of the target tab URL")
    ap.add_argument("--timeout", type=int, default=120, help="JS awaitPromise timeout (s)")
    ap.add_argument("expr", help="JS expression, or a path ending in .js to read it from")
    a = ap.parse_args()

    expr = open(a.expr).read() if a.expr.endswith(".js") else a.expr

    target = next((t for t in tabs(a.port)
                   if t.get("type") == "page" and a.match in t.get("url", "")), None)
    if not target:
        print(json.dumps({"error": f"no page tab matching {a.match!r} on :{a.port}"}))
        sys.exit(1)

    ws = websocket.create_connection(target["webSocketDebuggerUrl"],
                                     max_size=200 * 1024 * 1024, timeout=a.timeout + 15)
    ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
        "expression": expr, "awaitPromise": True, "returnByValue": True,
        "timeout": a.timeout * 1000,
    }}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") != 1:
            continue
        res = msg.get("result", {})
        if "exceptionDetails" in res:
            exc = res["exceptionDetails"].get("exception", {})
            print(json.dumps({"error": exc.get("description") or exc.get("value")
                              or res["exceptionDetails"]}))
        else:
            print(json.dumps(res.get("result", {}).get("value")))
        break


if __name__ == "__main__":
    main()
