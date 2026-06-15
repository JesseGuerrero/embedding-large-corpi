#!/usr/bin/env bash
# Browser test rig for the embedding 3D graph — virtual X display + Brave
# (software WebGL) + instant screenshots + REAL mouse input + CDP JS eval.
#
# Why real input: Plotly's scatter3d point picking only engages on genuine
# pointer events (synthetic JS MouseEvents do NOT trigger the gl pick), so the
# selection lag / click issues are only reproducible with a real xdotool click.
#
# Requires: Xvfb, brave, ffmpeg, xdotool, uv, python3.
#
# Usage:
#   scripts/browser.sh start                 # http server (docs/) + Xvfb :97 + Brave -> :8099
#   scripts/browser.sh shot [out.png]        # instant screenshot (default /tmp/emb-shot.png)
#   scripts/browser.sh js '<expr|file.js>'   # run JS in the page via CDP, print JSON
#   scripts/browser.sh click X Y             # real left click at screen px
#   scripts/browser.sh move X Y              # move pointer (hover)
#   scripts/browser.sh drag X1 Y1 X2 Y2 [N]  # real left-drag (orbit) in N steps
#   scripts/browser.sh key <keysym> ; type '<text>'
#   scripts/browser.sh stop
set -euo pipefail

DISP="${EMB_DISP:-:97}"
GEOM="${EMB_GEOM:-1600x900}"
PORT="${EMB_CDP_PORT:-9223}"
HTTP="${EMB_HTTP_PORT:-8099}"
URL="${EMB_URL:-http://localhost:$HTTP/index.html}"
PROFILE="${EMB_PROFILE:-/tmp/brave-emb}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DOCS="$(cd "$HERE/../docs" && pwd)"
export DISPLAY="$DISP"

start() {
  if ! pgrep -f "http.server $HTTP" >/dev/null 2>&1; then
    ( cd "$DOCS" && setsid python3 -m http.server "$HTTP" </dev/null >/tmp/emb-http.log 2>&1 & )
    sleep 1
  fi
  if ! pgrep -f "Xvfb $DISP" >/dev/null 2>&1; then
    setsid Xvfb "$DISP" -screen 0 "${GEOM}x24" </dev/null >/tmp/emb-xvfb.log 2>&1 &
    sleep 1
  fi
  if pgrep -f "user-data-dir=$PROFILE" >/dev/null 2>&1; then
    echo "brave already running on $DISP (CDP :$PORT) — reusing; '$0 stop' to reset"; exit 0
  fi
  setsid brave --user-data-dir="$PROFILE" --no-first-run --no-default-browser-check \
    --window-size="${GEOM/x/,}" --window-position=0,0 \
    --use-angle=swiftshader --enable-unsafe-swiftshader \
    --remote-debugging-port="$PORT" --remote-allow-origins=* --app="$URL" \
    </dev/null >>/tmp/emb-brave.log 2>&1 &
  echo "started http :$HTTP (docs/) + Xvfb $DISP + brave (CDP :$PORT) -> $URL"
  echo "give it ~3-6s to load plotly + render; then: $0 shot"
}

stop() {
  pkill -f "user-data-dir=$PROFILE" 2>/dev/null || true
  pkill -f "Xvfb $DISP" 2>/dev/null || true
  pkill -f "http.server $HTTP" 2>/dev/null || true
  echo "stopped"
}

shot() {
  local out="${1:-/tmp/emb-shot.png}"
  local g; g="$(xdotool getdisplaygeometry 2>/dev/null | tr ' ' x)"; g="${g:-$GEOM}"
  ffmpeg -y -loglevel error -f x11grab -video_size "$g" -i "$DISP" -frames:v 1 "$out"
  echo "$out"
}

drag() {
  local x1=$1 y1=$2 x2=$3 y2=$4 steps=${5:-14} i xi yi
  xdotool mousemove "$x1" "$y1" mousedown 1
  for i in $(seq 1 "$steps"); do
    xi=$(( x1 + (x2 - x1) * i / steps )); yi=$(( y1 + (y2 - y1) * i / steps ))
    xdotool mousemove "$xi" "$yi"
  done
  xdotool mouseup 1
}

js() { uv run --quiet --with websocket-client python3 "$HERE/cdp.py" --port "$PORT" --match "localhost:$HTTP" "$@"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  start)  start ;;
  stop)   stop ;;
  shot)   shot "$@" ;;
  click)  xdotool mousemove "$1" "$2" click 1 ;;
  move)   xdotool mousemove "$1" "$2" ;;
  drag)   drag "$@" ;;
  key)    xdotool key "$1" ;;
  type)   xdotool type -- "$1" ;;
  js)     js "$@" ;;
  *) echo "usage: $0 {start|stop|shot|click X Y|move X Y|drag X1 Y1 X2 Y2 [steps]|key|type|js}"; exit 1 ;;
esac
