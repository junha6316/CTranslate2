"""G1: the bench_batch.py 48-config matrix (small/large-v3 x fp16/int8_fp16 x ts on/off x
beam 1/5 x batch 1/4/8), identity only: 1 warmup + 1 decode per config, with scores.
Usage: matrix_ids.py LABEL OUT.json [flash]"""
import json, sys
import numpy as np, ctranslate2
from faster_whisper import decode_audio
from faster_whisper.feature_extractor import FeatureExtractor
from common import print_lib_info, env_snapshot

LABEL, OUT = sys.argv[1], sys.argv[2]
FLASH = len(sys.argv) > 3 and sys.argv[3] == "flash"
MODELS = {"small": ("/opt/models/faster-whisper-small", 80),
          "large-v3": ("/opt/models/faster-whisper-large-v3", 128)}


def windows(n_mels):
    fe = FeatureExtractor(feature_size=n_mels)
    out = []
    for p in ["/opt/audio/physicsworks.wav", "/opt/audio/jfk_x5.flac"]:
        f = fe(decode_audio(p, sampling_rate=16000))
        for s in range(0, f.shape[-1], 3000):
            w = f[..., s:s + 3000]
            if w.shape[-1] < 1000:
                continue
            if w.shape[-1] < 3000:
                w = np.pad(w, ((0, 0), (0, 3000 - w.shape[-1])))
            out.append(w.astype(np.float32))
    return out


info = print_lib_info()
env = env_snapshot()
print("env", env, "flash", FLASH, flush=True)
rows = []
for mname, (mpath, n_mels) in MODELS.items():
    wins = windows(n_mels)
    for ctype in ["float16", "int8_float16"]:
        kw = dict(flash_attention=True) if FLASH else {}
        model = ctranslate2.models.Whisper(mpath, device="cuda", compute_type=ctype, **kw)
        for ts in [True, False]:
            prompt = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
            if not ts:
                prompt = prompt + ["<|notimestamps|>"]
            for beam in [1, 5]:
                for n in [1, 4, 8]:
                    sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(wins[:n])))
                    model.generate(sv, [prompt] * n, beam_size=beam, return_scores=True)
                    r = model.generate(sv, [prompt] * n, beam_size=beam, return_scores=True)
                    rows.append(dict(label=LABEL, model=mname, ctype=ctype, timestamps=ts,
                                     beam=beam, batch=n, flash=FLASH, env=env, lib=info["lib"],
                                     tok_ids=[list(map(int, x.sequences_ids[0])) for x in r],
                                     scores=[float(x.scores[0]) for x in r]))
                    print(f"{LABEL} {mname} {ctype} ts={ts} b{beam} n={n}: "
                          f"{sum(len(x.sequences_ids[0]) for x in r)} tok", flush=True)
        del model
json.dump(rows, open(OUT, "w"))
print("wrote", OUT, len(rows), flush=True)
