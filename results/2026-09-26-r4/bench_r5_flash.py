"""Round-4 paired bench: whisper-small fp16, batch1, first 30s window.
beam1 AND beam5, ts on AND off. median-of-7, 2 warmup, total generate() wall.
Flags come from env (recorded per row); dumps tok_ids + score for parity.
Usage: bench_r5.py LABEL OUT.json
"""
import json, os, statistics, sys, time
import numpy as np, ctranslate2
sys.path.insert(0, "/opt/real")
from bench_batch_windows import windows

LABEL, OUT = sys.argv[1], sys.argv[2]
REPS, WARMUP = 7, 2
ENV_KEYS = ["CT2_CUDA_GRAPHS", "CT2_CUDA_PAD_KV", "CT2_CUDA_GRAPHS_RESERVE",
            "CT2_CUDA_PREALLOC_KV", "CT2_CUDA_GRAPHS_CHECK", "CT2_CUDA_ALLOC_DEBUG"]
env = {k: os.environ.get(k) for k in ENV_KEYS}

wins = windows(80)
sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(wins[:1])))
model = ctranslate2.models.Whisper("/opt/models/faster-whisper-small",
                                   device="cuda", compute_type="float16", flash_attention=True)
base = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]

rows = []
for ts in [True, False]:
    prompt = base if ts else base + ["<|notimestamps|>"]
    for beam in [1, 5]:
        for _ in range(WARMUP):
            model.generate(sv, [prompt], beam_size=beam, return_scores=True)
        t = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            r = model.generate(sv, [prompt], beam_size=beam, return_scores=True)
            t.append((time.perf_counter() - t0) * 1000)
        med = statistics.median(t)
        toks = [int(x) for x in r[0].sequences_ids[0]]
        score = float(r[0].scores[0]) if r[0].scores else None
        row = dict(label=LABEL, ts=ts, beam=beam, median_ms=round(med, 2),
                   min_ms=round(min(t), 2), max_ms=round(max(t), 2),
                   ntok=len(toks), score=score, env=env, tok_ids=toks)
        rows.append(row)
        g=env["CT2_CUDA_GRAPHS"]; rr=env["CT2_CUDA_GRAPHS_RESERVE"]; pad=env["CT2_CUDA_PAD_KV"]
        print(f"{LABEL} ts={ts} b{beam} G={g} R={rr} PAD={pad}: {med:.1f} ms (min {min(t):.1f} max {max(t):.1f}) {len(toks)} tok", flush=True)
json.dump(rows, open(OUT, "w"))
print("wrote", OUT, flush=True)
