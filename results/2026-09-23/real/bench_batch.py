"""Throughput of Whisper generate over batches of 30 s windows, one build per run.

Windows: physicsworks.wav (203 s -> 7 windows, last one padded) then jfk_x5.flac
(55 s -> 2). Batch N decodes the first N windows in one generate call, so different
N see different content; compare builds at the same N, not N against N.
Usage: bench_batch.py LABEL OUT.json [quick]
"""
import json, statistics, sys, time
import numpy as np, ctranslate2
from faster_whisper import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor

LABEL, OUT = sys.argv[1], sys.argv[2]
QUICK = len(sys.argv) > 3
REPS, WARMUP = (2, 1) if QUICK else (5, 1)
MODELS = {"small": ("/opt/models/faster-whisper-small", 80),
          "large-v3": ("/opt/models/faster-whisper-large-v3", 128)}
BATCHES = [1, 8] if QUICK else [1, 4, 8]

def windows(n_mels):
    fe = FeatureExtractor(feature_size=n_mels)
    out = []
    for p in ["/opt/audio/physicsworks.wav", "/opt/audio/jfk_x5.flac"]:
        f = fe(decode_audio(p, sampling_rate=16000))
        for s in range(0, f.shape[-1], 3000):
            w = f[..., s:s + 3000]
            if w.shape[-1] < 1000:   # drop tails shorter than 10 s
                continue
            if w.shape[-1] < 3000:
                w = np.pad(w, ((0, 0), (0, 3000 - w.shape[-1])))
            out.append(w.astype(np.float32))
    return out

rows = []
for mname, (mpath, n_mels) in MODELS.items():
    if QUICK and mname != "small":
        continue
    wins = windows(n_mels)
    for ctype in ["float16", "int8_float16"]:
        model = ctranslate2.models.Whisper(mpath, device="cuda", compute_type=ctype)
        for ts in [True, False]:
            prompt = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
            if not ts:
                prompt = prompt + ["<|notimestamps|>"]
            for beam in [1, 5]:
                for n in BATCHES:
                    sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(wins[:n])))
                    for _ in range(WARMUP):
                        model.generate(sv, [prompt] * n, beam_size=beam)
                    t = []
                    for _ in range(REPS):
                        t0 = time.perf_counter()
                        r = model.generate(sv, [prompt] * n, beam_size=beam)
                        t.append((time.perf_counter() - t0) * 1000)
                    med = statistics.median(t)
                    ntok = sum(len(x.sequences_ids[0]) for x in r)
                    row = dict(label=LABEL, model=mname, ctype=ctype, timestamps=ts, beam=beam,
                               batch=n, median_ms=round(med, 2), min_ms=round(min(t), 2),
                               ntok=ntok, ms_per_window=round(med / n, 2),
                               tok_ids=[list(x.sequences_ids[0]) for x in r])
                    rows.append(row)
                    print(f"{LABEL} {mname} {ctype} ts={ts} b{beam} n={n}: {med:.1f} ms, "
                          f"{med/n:.1f} ms/window, {ntok} tok", flush=True)
        del model
json.dump(rows, open(OUT, "w"))
print("wrote", OUT, len(rows))
