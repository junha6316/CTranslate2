"""clear_cache check: after a large decode, does unload_model() hand the pool's kept
memory back to the device, and does the reloaded model still decode identically?
Usage: r7_unload.py LABEL   (threshold from CT2_CUDA_POOL_RELEASE_THRESHOLD)
"""
import ctypes, json, os, subprocess, sys, time
import numpy as np, ctranslate2
sys.path.insert(0, "/opt")
from bench_batch_windows import windows
wins = windows(80)
_cu = ctypes.CDLL("libcuda.so.1"); _cu.cuInit(0); _pool = ctypes.c_void_p(); _d = ctypes.c_int()
_cu.cuDeviceGet(ctypes.byref(_d), 0); _cu.cuDeviceGetMemPool(ctypes.byref(_pool), _d)
def attr(i):
    v = ctypes.c_uint64(); _cu.cuMemPoolGetAttribute(_pool, i, ctypes.byref(v)); return v.value
def smi():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"]).split()[0])
def snap(tag):
    time.sleep(0.3)
    return dict(tag=tag, pool_reserved_mb=round(attr(5) / 2**20), pool_used_mb=round(attr(7) / 2**20),
                smi_used_mb=smi())
TS_ON = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
TS_OFF = TS_ON + ["<|notimestamps|>"]
sv = lambda n: ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(wins[:n])))
out = dict(label=sys.argv[1], env=os.environ.get("CT2_CUDA_POOL_RELEASE_THRESHOLD"), snaps=[])
model = ctranslate2.models.Whisper("/opt/models/faster-whisper-small", device="cuda",
                                   compute_type="float16", flash_attention=True)
out["thr"] = attr(4)
out["snaps"].append(snap("loaded"))
r1 = model.generate(sv(1), [TS_OFF], beam_size=5)
model.generate(sv(8), [TS_ON] * 8, beam_size=5)
out["snaps"].append(snap("after_b5n8"))
model.unload_model()
out["snaps"].append(snap("unloaded"))
model.load_model()
out["snaps"].append(snap("reloaded"))
r2 = model.generate(sv(1), [TS_OFF], beam_size=5)
out["snaps"].append(snap("after_b5n1"))
out["tokens_equal_after_reload"] = r1[0].sequences_ids == r2[0].sequences_ids
del model
out["snaps"].append(snap("deleted"))
print(json.dumps(out), flush=True)
