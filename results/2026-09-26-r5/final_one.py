"""Single-config recheck (flash-on, small int8_float16, ts-off). Usage: final_one.py LABEL OUT.json"""
import json, statistics, sys, time
import numpy as np, ctranslate2
sys.path.insert(0, "/opt/real")
from bench_batch_windows import windows
print("ctranslate2.__file__ =", ctranslate2.__file__, flush=True)
wins = windows(80)
m = ctranslate2.models.Whisper("/opt/models/faster-whisper-small", device="cuda", compute_type="int8_float16", flash_attention=True)
prompt = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
rows = []
for beam, n in [(1, 4), (5, 1)]:
    sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(wins[:n])))
    for _ in range(3): m.generate(sv, [prompt] * n, beam_size=beam)
    t = []
    for _ in range(21):
        t0 = time.perf_counter(); r = m.generate(sv, [prompt] * n, beam_size=beam); t.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(t)
    rows.append(dict(label=sys.argv[1], beam=beam, batch=n, median_ms=med, tok=[list(x.sequences_ids[0]) for x in r]))
    print(f"{sys.argv[1]} b{beam} n={n}: {med:.2f} ms (min {min(t):.2f})", flush=True)
json.dump(rows, open(sys.argv[2], "w"))
