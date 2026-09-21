#!/usr/bin/env bash
# One command to put the app on screen.  ./demo.sh
#
# Boot takes ~40s: two checkpoints load into memory before the first page will answer.
# The browser is opened only once the app actually responds, so nobody watches a spinner.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
PORT="${PORT:-5002}"

for w in results/rop/weights.pth results/rop_staging/weights.pth; do
  if [ ! -f "$w" ]; then
    echo "!! missing $w"
    echo "   fetch both with:  $PY scripts/get_weights.py"
    exit 1
  fi
done

lsof -ti:"$PORT" | xargs kill 2>/dev/null || true
sleep 1

echo "starting RetinAI on :$PORT (loading models, ~40s)..."
PORT="$PORT" $PY webapp/app.py > /tmp/retinai-demo.log 2>&1 &
APP=$!
trap 'kill $APP 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 90); do
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/"; then
    echo "ready -> http://127.0.0.1:$PORT"
    command -v open >/dev/null && open "http://127.0.0.1:$PORT/screen"
    echo
    echo "  On the Screen page, set the patient context to 'Preterm infant (NICU"
    echo "  screening)' first — it is required, and the app returns 400 without it."
    echo
    echo "  No fundus images ship with this repository. The evaluation set is private"
    echo "  infant patient photographs and stays out of a public repo; supply your own."
    echo
    echo "  DEMO.md is the presenter runbook: what to say, and the numbers to quote."
    echo
    echo "  Ctrl-C to stop."
    wait $APP
    exit 0
  fi
  sleep 1
done

echo "!! did not come up — last lines of /tmp/retinai-demo.log:"; tail -20 /tmp/retinai-demo.log
exit 1
