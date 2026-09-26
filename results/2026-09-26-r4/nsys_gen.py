import sys, numpy as np, ctranslate2
sys.path.insert(0,"/opt/real")
from bench_batch_windows import windows
w=windows(80); sv=ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(w[:1])))
m=ctranslate2.models.Whisper("/opt/models/faster-whisper-small",device="cuda",compute_type="float16")
prompt=["<|startoftranscript|>","<|en|>","<|transcribe|>"]  # ts-on
# warmup (captures graph build)
for _ in range(2):
    m.generate(sv,[prompt],beam_size=5,return_scores=True)
ctranslate2.set_random_seed(0)
import ctypes
# measured region: exactly 3 generates
for _ in range(3):
    r=m.generate(sv,[prompt],beam_size=5,return_scores=True)
print("done ntok",len(r[0].sequences_ids[0]),flush=True)
