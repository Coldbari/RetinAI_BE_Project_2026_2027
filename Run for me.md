# Running the app

From the root of this repository:

```bash
python scripts/get_weights.py   # once after cloning — ~180 MB, both checkpoints
./demo.sh                       # boots the app and opens the browser (~40s)
```

`demo.sh` refuses to start if either checkpoint is missing, rather than booting an app whose
every result reads "model not loaded".

## If something goes wrong

```bash
tail -20 /tmp/retinai-demo.log     # what it said while dying
python scripts/get_weights.py      # re-pull both checkpoints
```

Port 5002 by default; `PORT=5003 ./demo.sh` to change it.
