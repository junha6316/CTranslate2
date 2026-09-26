"""Replay bench_batch_flash.py's cell order (small, one ctype) and time the ts=0 b5 n1
target rep by rep, to see transients after the preceding cells.
Usage: seq.py LABEL [--ctype float16] [--mode quick|full] [--target-reps 12]
       [--only-prev CELL]  (CELL like ts0b1n8: run only that preceding cell before target)
"""
import argparse, ctypes, json, os, statistics, sys, time
import numpy as np, ctranslate2
sys.path.insert(0, "/opt")
from bench_batch_windows import windows
ap = argparse.ArgumentParser()
ap.add_argument("label"); ap.add_argument("--ctype", default="float16")
ap.add_argument("--mode", default="quick"); ap.add_argument("--target-reps", type=int, default=12)
ap.add_argument("--only-prev", default=None)
a = ap.parse_args()
wins = windows(80)
model = ctranslate2.models.Whisper("/opt/models/faster-whisper-small", device="cuda",
                                   compute_type=a.ctype, flash_attention=True)
_cu = ctypes.CDLL("libcuda.so.1"); _cu.cuInit(0); _pool = ctypes.c_void_p(); _d = ctypes.c_int()
_cu.cuDeviceGet(ctypes.byref(_d), 0); _cu.cuDeviceGetMemPool(ctypes.byref(_pool), _d)
if os.environ.get("R6_POOL_THRESHOLD"):
    v = ctypes.c_uint64(int(os.environ["R6_POOL_THRESHOLD"])); _cu.cuMemPoolSetAttribute(_pool, 4, ctypes.byref(v))
def reserved_mb():
    v = ctypes.c_uint64(); _cu.cuMemPoolGetAttribute(_pool, 5, ctypes.byref(v)); return round(v.value / 2**20)
def prompt(ts):
    p = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
    return p if ts else p + ["<|notimestamps|>"]
def sv(n): return ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(wins[:n])))
batches = [1, 8] if a.mode == "quick" else [1, 4, 8]
runs = 3 if a.mode == "quick" else 9
cells = [(ts, b, n) for ts in [True, False] for b in [1, 5] for n in batches]
target = (False, 5, 1)
prev = cells[:cells.index(target)]
if a.only_prev:
    prev = [c for c in prev if "ts%db%dn%d" % (int(c[0]), c[1], c[2]) == a.only_prev]
for ts, b, n in prev:
    for _ in range(runs):
        model.generate(sv(n), [prompt(ts)] * n, beam_size=b)
t = []
for _ in range(a.target_reps):
    t0 = time.perf_counter()
    r = model.generate(sv(1), [prompt(False)], beam_size=5)
    t.append(round((time.perf_counter() - t0) * 1000, 2))
print(json.dumps(dict(label=a.label, ctype=a.ctype, mode=a.mode, only_prev=a.only_prev,
                      prev=["ts%db%dn%d" % (int(c[0]), c[1], c[2]) for c in prev],
                      target_ms=t, median_after3=statistics.median(t[3:]),
                      ntok=len(r[0].sequences_ids[0]), pool_reserved_mb=reserved_mb(),
                      thr=os.environ.get("R6_POOL_THRESHOLD"))), flush=True)
