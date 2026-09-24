"""Workload for nsys: one model, fp16, beam 5, timestamps on, batch N, 2 warm-up + 5 runs.
Prints the wall time of the 5 timed runs so kernel time can be divided by it.
Usage: prof_batch.py MODEL N
"""
import sys, time, numpy as np, ctranslate2
sys.argv += []
mname, n = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, "/opt")
from bench_batch_windows import windows
n_mels = 128 if mname == "large-v3" else 80
m = ctranslate2.models.Whisper(f"/opt/models/faster-whisper-{mname}", device="cuda", compute_type="float16")
sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(windows(n_mels)[:n])))
p = [["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]] * n
for _ in range(2): m.generate(sv, p, beam_size=5)
t0 = time.perf_counter()
for _ in range(5): m.generate(sv, p, beam_size=5)
print(f"WALL {mname} n={n} {(time.perf_counter()-t0)*1000:.1f} ms for 5 runs")
