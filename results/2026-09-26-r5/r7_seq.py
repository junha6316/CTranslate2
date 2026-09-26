"""Replay bench_batch(_flash).py's cell order for one (model, ctype) and time the
ts=0 b5 n1 target rep by rep, to expose the transient after the preceding cells.
Usage: r7_seq.py LABEL [--model small|large-v3] [--ctype float16] [--mode quick|full]
       [--target-reps 12] [--only-prev CELL] [--flash 1]
The pool release threshold is whatever CT2 set (CT2_CUDA_POOL_RELEASE_THRESHOLD);
it is read back from the device mempool and reported as thr.
"""
import argparse, ctypes, json, os, statistics, sys, time
import numpy as np, ctranslate2
sys.path.insert(0, "/opt")
from bench_batch_windows import windows
ap = argparse.ArgumentParser()
ap.add_argument("label"); ap.add_argument("--model", default="small")
ap.add_argument("--ctype", default="float16")
ap.add_argument("--mode", default="quick"); ap.add_argument("--target-reps", type=int, default=12)
ap.add_argument("--only-prev", default=None); ap.add_argument("--flash", type=int, default=1)
a = ap.parse_args()
MODELS = {"small": ("/opt/models/faster-whisper-small", 80),
          "large-v3": ("/opt/models/faster-whisper-large-v3", 128)}
mpath, n_mels = MODELS[a.model]
wins = windows(n_mels)
model = ctranslate2.models.Whisper(mpath, device="cuda", compute_type=a.ctype,
                                   flash_attention=bool(a.flash))
_cu = ctypes.CDLL("libcuda.so.1"); _cu.cuInit(0); _pool = ctypes.c_void_p(); _d = ctypes.c_int()
_cu.cuDeviceGet(ctypes.byref(_d), 0); _cu.cuDeviceGetMemPool(ctypes.byref(_pool), _d)
def attr(i):
    v = ctypes.c_uint64(); _cu.cuMemPoolGetAttribute(_pool, i, ctypes.byref(v)); return v.value
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
res_mb_before = round(attr(5) / 2**20)
t = []
for _ in range(a.target_reps):
    t0 = time.perf_counter()
    r = model.generate(sv(1), [prompt(False)], beam_size=5)
    t.append(round((time.perf_counter() - t0) * 1000, 2))
print(json.dumps(dict(label=a.label, model=a.model, ctype=a.ctype, mode=a.mode,
                      only_prev=a.only_prev, flash=a.flash,
                      prev=["ts%db%dn%d" % (int(c[0]), c[1], c[2]) for c in prev],
                      target_ms=t, median_after3=statistics.median(t[3:]),
                      first3_excess_ms=round(sum(t[:3]) - 3 * statistics.median(t[3:]), 2),
                      ntok=len(r[0].sequences_ids[0]), tok_ids=list(r[0].sequences_ids[0]),
                      pool_reserved_before_mb=res_mb_before,
                      pool_reserved_mb=round(attr(5) / 2**20),
                      pool_reserved_high_mb=round(attr(6) / 2**20),
                      thr=attr(4), env=os.environ.get("CT2_CUDA_POOL_RELEASE_THRESHOLD"))),
      flush=True)
