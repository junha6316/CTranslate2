"""probe_ab.py TAG OUT.jsonl: small int8_float16 flash-off cells (ts1 b1 n1, ts1 b1 n4, ts1 b5 n1), REPS,WARMUP=(7,2), same windows as bench_batch.py."""
import json, statistics, sys, time
import numpy as np, ctranslate2
from faster_whisper import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor
TAG, OUT = sys.argv[1], sys.argv[2]
lib = [l.split()[-1] for l in open('/proc/self/maps') if 'libctranslate2' in l][0]
fe = FeatureExtractor(feature_size=80); wins = []
for p in ["/opt/audio/physicsworks.wav", "/opt/audio/jfk_x5.flac"]:
    f = fe(decode_audio(p, sampling_rate=16000))
    for s in range(0, f.shape[-1], 3000):
        w = f[..., s:s + 3000]
        if w.shape[-1] < 1000: continue
        if w.shape[-1] < 3000: w = np.pad(w, ((0, 0), (0, 3000 - w.shape[-1])))
        wins.append(w.astype(np.float32))
model = ctranslate2.models.Whisper("/opt/models/faster-whisper-small", device="cuda", compute_type="int8_float16")
prompt = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
res = dict(tag=TAG, pkg=ctranslate2.__file__, lib=lib)
for beam, n in [(1, 1), (1, 4), (5, 1)]:
    sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(wins[:n])))
    for _ in range(2): model.generate(sv, [prompt] * n, beam_size=beam)
    t = []
    for _ in range(7):
        t0 = time.perf_counter(); model.generate(sv, [prompt] * n, beam_size=beam); t.append((time.perf_counter() - t0) * 1000)
    res[f"b{beam}n{n}"] = round(statistics.median(t), 2)
print(json.dumps(res), flush=True)
open(OUT, "a").write(json.dumps(res) + "\n")
