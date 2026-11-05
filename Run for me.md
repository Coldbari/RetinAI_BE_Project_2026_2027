# Running the app

From the root of this repository:

```bash
python scripts/get_weights.py   # once after cloning — ~180 MB, both checkpoints
./demo.sh                       # boots the app and opens the browser (~40s)
```

`demo.sh` refuses to start if either checkpoint is missing, rather than booting an app whose
every result reads "model not loaded".

These two are the older checkpoints. The served model, five DINOv2 checkpoints, is not on the
model repository yet. Without it the verdict comes from the retired ResNet50, so use a run like
that to check the software works, not to read a result. `demo.sh` prints the same warning.

## If something goes wrong

```bash
tail -20 /tmp/retinai-demo.log     # what it said while dying
python scripts/get_weights.py      # re-pull both checkpoints
```

Port 5002 by default; `PORT=5003 ./demo.sh` to change it.
