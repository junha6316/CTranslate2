"""Round-6 flash-on beam5 ts-off regression probe (one JSON line on stdout).

Target cell = bench_batch.py "ts=0 b5 n=1" with flash_attention=True: prompt ends in
<|notimestamps|>, beam 5, batch 1 = first window of /opt/bench_batch_windows.py.

Two decoder states, same model object:
  COLD      fresh model, only batch-1 decodes before/while timing.
  POISONED  after PRIMER_RUNS x the cell bench_batch.py runs right before the target in
            its fixed order (ts=1 b5 n=8), which leaves any cross-decode state sized for
            40 rows. bench_batch measures the target in this state.
For each state: WARMUP + median/IQR of REPS on window 0, then a per-window pass
(1 warmup + median of WREPS on each window, b5 n1) and its total.
Usage: probe.py LABEL [--model small|large-v3] [--ctype float16|int8_float16]
                      [--states cold,poisoned] [--no-windows]
"""
import argparse, ctypes, json, os, statistics, sys, time
import numpy as np, ctranslate2
sys.path.insert(0, "/opt")
from bench_batch_windows import windows

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--model", default="small")
ap.add_argument("--ctype", default="float16")
ap.add_argument("--states", default="cold,poisoned")
ap.add_argument("--reps", type=int, default=11)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--wreps", type=int, default=5)
ap.add_argument("--primer-runs", type=int, default=2)
ap.add_argument("--no-windows", action="store_true")
ap.add_argument("--flash", type=int, default=1)
a = ap.parse_args()

MODELS = {"small": ("/opt/models/faster-whisper-small", 80),
          "large-v3": ("/opt/models/faster-whisper-large-v3", 128)}
mpath, n_mels = MODELS[a.model]
wins = windows(n_mels)
TS_ON = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
TS_OFF = TS_ON + ["<|notimestamps|>"]
model = ctranslate2.models.Whisper(mpath, device="cuda", compute_type=a.ctype,
                                   flash_attention=bool(a.flash))

# Device mempool (the one CT2's cudaMallocAsync draws from) via the driver API:
# R6_POOL_THRESHOLD=<bytes> sets CU_MEMPOOL_ATTR_RELEASE_THRESHOLD (CT2 leaves it at the
# default 0 = trim unused memory back to the OS at every sync). Stats are always read.
_cu = ctypes.CDLL("libcuda.so.1")
_pool = ctypes.c_void_p()
try:
    _cu.cuInit(0)
    _dev = ctypes.c_int()
    _cu.cuDeviceGet(ctypes.byref(_dev), 0)
    if _cu.cuDeviceGetMemPool(ctypes.byref(_pool), _dev) != 0:
        _pool = None
except Exception:
    _pool = None

def pool_attr(attr):
    if not _pool:
        return None
    v = ctypes.c_uint64()
    _cu.cuMemPoolGetAttribute(_pool, attr, ctypes.byref(v))
    return v.value

if _pool and os.environ.get("R6_POOL_THRESHOLD"):
    _v = ctypes.c_uint64(int(os.environ["R6_POOL_THRESHOLD"]))
    rc = _cu.cuMemPoolSetAttribute(_pool, 4, ctypes.byref(_v))
    print(f"# pool release threshold -> {_v.value} rc={rc} readback={pool_attr(4)}",
          file=sys.stderr, flush=True)

def pool_stats():
    # 4 RELEASE_THRESHOLD, 5 RESERVED_MEM_CURRENT, 6 RESERVED_MEM_HIGH, 8 USED_MEM_HIGH
    mb = lambda x: None if x is None else round(x / 2**20, 1)
    return dict(thr=pool_attr(4), reserved_cur_mb=mb(pool_attr(5)),
                reserved_high_mb=mb(pool_attr(6)), used_high_mb=mb(pool_attr(8)))

def sv(ws):
    return ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(ws)))

# R6_PROFILE_RANGE=1: bracket only the timed window-0 reps with cudaProfilerStart/Stop
# so `nsys profile -c cudaProfilerApi` records the target decodes, not the primer.
_cudart = None
if os.environ.get("R6_PROFILE_RANGE") == "1":
    _cudart = ctypes.CDLL("/usr/local/cuda-12.8/lib64/libcudart.so.12")

def timed(feats, prompts, reps, warmup, beam=5, profile=False):
    for _ in range(warmup):
        model.generate(feats, prompts, beam_size=beam)
    if profile and _cudart is not None:
        _cudart.cudaProfilerStart()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = model.generate(feats, prompts, beam_size=beam)
        t.append((time.perf_counter() - t0) * 1000)
    if profile and _cudart is not None:
        _cudart.cudaProfilerStop()
    return t, r

def stats(t):
    q = statistics.quantiles(t, n=4) if len(t) >= 2 else [t[0]] * 3
    return dict(median_ms=round(statistics.median(t), 2), iqr_ms=round(q[2] - q[0], 2),
                q1=round(q[0], 2), q3=round(q[2], 2), min_ms=round(min(t), 2))

def measure(state):
    t, r = timed(sv(wins[:1]), [TS_OFF], a.reps, a.warmup, profile=True)
    out = dict(stats(t), ntok=len(r[0].sequences_ids[0]), tok_ids=list(r[0].sequences_ids[0]),
               all_ms=[round(x, 2) for x in t])
    if not a.no_windows:
        per, toks = [], []
        for w in wins:
            tw, rw = timed(sv([w]), [TS_OFF], a.wreps, 1)
            per.append(round(statistics.median(tw), 2))
            toks.append(list(rw[0].sequences_ids[0]))
        out.update(per_window_ms=per, per_window_total_ms=round(sum(per), 2),
                   per_window_ntok=[len(x) for x in toks], per_window_tok_ids=toks)
    return out

res = dict(label=a.label, model=a.model, ctype=a.ctype, file=ctranslate2.__file__,
           version=ctranslate2.__version__, n_windows=len(wins),
           pool_threshold_env=os.environ.get("R6_POOL_THRESHOLD"), flash=a.flash)
for state in a.states.split(","):
    if state == "poisoned":
        feats8 = sv(wins[:8])
        for _ in range(a.primer_runs):
            model.generate(feats8, [TS_ON] * 8, beam_size=5)
    res[state] = measure(state)
    res[state]["pool"] = pool_stats()
    print(f"# {a.label} {a.model} {a.ctype} {state}: median {res[state]['median_ms']} ms "
          f"IQR {res[state]['iqr_ms']} ntok {res[state]['ntok']}"
          + (f" wtotal {res[state].get('per_window_total_ms')}" if not a.no_windows else ""),
          file=sys.stderr, flush=True)
print(json.dumps(res), flush=True)
