"""W_dense: 55-60 s of speech time-compressed to 30 s with ffmpeg atempo (1.8-2.0), through
the same feature path as bench_batch_windows (FeatureExtractor(n_mels), 3000 frames).
Keeps the windows whose eager whisper-small fp16 beam5 ts-on decode has 150..220 tokens
(target 5). Run with all CT2_CUDA_* flags unset. Writes dense_80.npy, dense_128.npy and
dense_meta.json."""
import json, subprocess
import numpy as np
import ctranslate2
from faster_whisper import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor
from common import R5T, SMALL, BASE_PROMPT, print_lib_info, to_sv, env_snapshot

print_lib_info()
assert not env_snapshot().get("CT2_CUDA_GRAPHS"), "run eager"
PW, JFK = "/opt/audio/physicsworks.wav", "/opt/audio/jfk_x5.flac"
# (label, [(file, start_s, dur_s), ...], atempo); total speech 55-60 s -> ~30 s.
CANDIDATES = [
    ("pw000-060@2.0", [(PW, 0, 60)], 2.0),
    ("pw060-120@2.0", [(PW, 60, 60)], 2.0),
    ("pw120-180@2.0", [(PW, 120, 60)], 2.0),
    ("pw030-090@2.0", [(PW, 30, 60)], 2.0),
    ("pw090-150@2.0", [(PW, 90, 60)], 2.0),
    ("pw143-203@2.0", [(PW, 143, 60)], 2.0),
    ("jfk000-055@1.84", [(JFK, 0, 55)], 1.84),
    ("pw000-056@1.87", [(PW, 0, 56)], 1.87),
    ("pw100-157@1.9", [(PW, 100, 57)], 1.9),
    ("pw180-203+jfk000-033@1.87", [(PW, 180, 23), (JFK, 0, 33)], 1.87),
    ("pw045-100@1.84", [(PW, 45, 55)], 1.84),
    ("pw150-208@1.94", [(PW, 145, 58)], 1.94),
]


def build_audio(label, parts, tempo):
    out = f"{R5T}/dense_{label}.wav"
    inputs, filt = [], []
    for i, (f, s, d) in enumerate(parts):
        inputs += ["-ss", str(s), "-t", str(d), "-i", f]
        filt.append(f"[{i}:a]aresample=16000,aformat=channel_layouts=mono[a{i}]")
    chain = ";".join(filt) + ";" + "".join(f"[a{i}]" for i in range(len(parts)))
    chain += f"concat=n={len(parts)}:v=0:a=1,atempo={tempo}[out]"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + inputs
                   + ["-filter_complex", chain, "-map", "[out]", "-ar", "16000", "-ac", "1", out],
                   check=True)
    return out


def features(path, n_mels):
    f = FeatureExtractor(feature_size=n_mels)(decode_audio(path, sampling_rate=16000))
    w = f[..., :3000]
    if w.shape[-1] < 3000:
        w = np.pad(w, ((0, 0), (0, 3000 - w.shape[-1])))
    return w.astype(np.float32)


model = ctranslate2.models.Whisper(SMALL, device="cuda", compute_type="float16")
kept, meta = [], []
for label, parts, tempo in CANDIDATES:
    path = build_audio(label, parts, tempo)
    dur = len(decode_audio(path, sampling_rate=16000)) / 16000
    w80 = features(path, 80)
    r = model.generate(to_sv(w80), [BASE_PROMPT], beam_size=5, return_scores=True)
    ntok = len(r[0].sequences_ids[0])
    ok = 150 <= ntok <= 220 and dur <= 30.05
    print(f"{label}: {dur:.2f} s, eager beam5 ntok={ntok} {'KEEP' if ok else 'drop'}", flush=True)
    meta.append(dict(label=label, parts=parts, tempo=tempo, dur=dur, ntok=ntok, kept=ok))
    if ok and len(kept) < 5:
        kept.append((label, path, ntok))
print("kept", [(k[0], k[2]) for k in kept], flush=True)
np.save(f"{R5T}/dense_80.npy", np.stack([features(p, 80) for _, p, _ in kept]))
np.save(f"{R5T}/dense_128.npy", np.stack([features(p, 128) for _, p, _ in kept]))
json.dump(dict(candidates=meta, kept=[dict(label=l, ntok=n) for l, _, n in kept]),
          open(f"{R5T}/dense_meta.json", "w"), indent=1)
print("wrote dense_80.npy / dense_128.npy:", len(kept), "windows", flush=True)
