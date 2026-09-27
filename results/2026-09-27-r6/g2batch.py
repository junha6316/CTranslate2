"""Batch-4 beam5 (std windows 0..3, ts-on) under the current CT2_* flags: tokens, scores,
median of 7 (2 warmup). Windows finish at different steps, so the beam search shrinks the
batch mid-decode. Usage: g2batch.py LABEL OUT.json [--flash]"""
import json, statistics, sys, time
import numpy as np, ctranslate2
sys.path.insert(0, "/opt/real")
from bench_batch_windows import windows
print("ctranslate2.__file__ =", ctranslate2.__file__, flush=True)
flash = "--flash" in sys.argv
m = ctranslate2.models.Whisper("/opt/models/faster-whisper-small", device="cuda", compute_type="float16", flash_attention=flash)
wins = windows(80)
prompt = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(wins[:4])))
for _ in range(2): m.generate(sv, [prompt] * 4, beam_size=5)
t = []
for _ in range(7):
    t0 = time.perf_counter(); r = m.generate(sv, [prompt] * 4, beam_size=5, return_scores=True); t.append((time.perf_counter() - t0) * 1000)
row = dict(label=sys.argv[1], median_ms=round(statistics.median(t), 2), tok_ids=[list(x.sequences_ids[0]) for x in r],
           scores=[float(x.scores[0]) for x in r], ntok=[len(x.sequences_ids[0]) for x in r])
print(sys.argv[1], row["median_ms"], row["ntok"], flush=True)
json.dump([row], open(sys.argv[2], "w"))
