"""Tokens + scores of the 48 bench_batch configs, one decode each (return_scores=True).
Usage: scores.py LABEL OUT.json FLASH(0|1)"""
import json, sys
import numpy as np, ctranslate2
sys.path.insert(0, "/opt/real")
from bench_batch_windows import windows
print("ctranslate2.__file__ =", ctranslate2.__file__, flush=True)
LABEL, OUT, FLASH = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
MODELS = {"small": ("/opt/models/faster-whisper-small", 80),
          "large-v3": ("/opt/models/faster-whisper-large-v3", 128)}
rows = []
for mname, (mpath, n_mels) in MODELS.items():
    wins = windows(n_mels)
    for ctype in ["float16", "int8_float16"]:
        model = ctranslate2.models.Whisper(mpath, device="cuda", compute_type=ctype, flash_attention=FLASH)
        for ts in [True, False]:
            prompt = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"] + ([] if ts else ["<|notimestamps|>"])
            for beam in [1, 5]:
                for n in [1, 4, 8]:
                    sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(wins[:n])))
                    r = model.generate(sv, [prompt] * n, beam_size=beam, return_scores=True)
                    rows.append(dict(model=mname, ctype=ctype, timestamps=ts, beam=beam, batch=n,
                                     tok_ids=[list(x.sequences_ids[0]) for x in r],
                                     scores=[float(x.scores[0]) for x in r]))
        del model
json.dump(rows, open(OUT, "w"))
print("wrote", OUT, len(rows), flush=True)
