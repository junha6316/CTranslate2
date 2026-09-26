"""Runs N decodes of each workload, sequentially in one process (flags from env; run with
CT2_VERBOSE=2 to get the runner's transition/capture/summary lines on stderr).
Usage: onedecode.py SPEC [SPEC ...] [--beam B] [--n N] [--model small|large-v3]
                    [--ctype C] [--batch2 SPEC2] [--json OUT]
--batch2 decodes [SPEC, SPEC2] as one batch of 2 instead.
Prints per decode: DECODE/NTOK/SCORE and TOKS lines (per batch entry)."""
import argparse, json, sys
import numpy as np, ctranslate2
from common import SMALL, LARGE, print_lib_info, env_snapshot
from workloads import resolve

ap = argparse.ArgumentParser()
ap.add_argument("specs", nargs="+")
ap.add_argument("--beam", type=int, default=5)
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--model", default="small")
ap.add_argument("--ctype", default="float16")
ap.add_argument("--batch2", default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args()

info = print_lib_info()
path, n_mels = (SMALL, 80) if a.model == "small" else (LARGE, 128)
model = ctranslate2.models.Whisper(path, device="cuda", compute_type=a.ctype)
batches = [[a.specs[0], a.batch2]] if a.batch2 else [[s] for s in a.specs]
rows = []
for specs in batches:
    items = [resolve(s, n_mels) for s in specs]
    sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack([w for w, _ in items])))
    prompts = [p for _, p in items]
    for d in range(a.n):
        print(f"BEGIN {'+'.join(specs)} decode {d}", flush=True)
        sys.stderr.flush()
        res = model.generate(sv, prompts, beam_size=a.beam, return_scores=True)
        for i, r in enumerate(res):
            toks = [int(t) for t in r.sequences_ids[0]]
            score = float(r.scores[0])
            print(f"DECODE {d} ENTRY {i} ({specs[i]}) P={len(prompts[i]) - 1} NTOK {len(toks)}"
                  f" SCORE {score:.8g}", flush=True)
            print(f"TOKS {toks}", flush=True)
            rows.append(dict(decode=d, entry=i, workload=specs[i], beam=a.beam, ntok=len(toks),
                             score=score, tok_ids=toks, env=env_snapshot(), lib=info["lib"]))
if a.json:
    json.dump(rows, open(a.json, "w"))
