import os, subprocess, sys, numpy as np, ctranslate2
sys.path.insert(0, "/opt")
from bench_batch_windows import windows
def used_mb():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
m = ctranslate2.models.Whisper("/opt/models/faster-whisper-small", device="cuda", compute_type="float16")
w = windows(80)
p = lambda n: [["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]] * n
sv8 = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(w[:8])))
r = m.generate(sv8, p(8), beam_size=5)
print("MODE", os.environ.get("CT2_CUDA_ALLOCATOR", "cuda_malloc_async"), "THR", os.environ.get("CT2_CUDA_POOL_RELEASE_THRESHOLD"))
print("tok0", r[0].sequences_ids[0][:8])
before = used_mb()
m.unload_model()
after = used_mb()
print(f"used MB after b5xn8: {before}  after unload_model: {after}  released: {before - after}")
