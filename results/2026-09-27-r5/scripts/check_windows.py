"""Tokens + scores over a list of workloads for beam 1/5 (identity only: 1 warmup + 1
decode each), whisper-small fp16 batch1. Flags from env. Also probes the runner's
per-decode summary (transitions/captures/replays) for every row.
Usage: check_windows.py LABEL OUT.json [SPEC ...]   (default: the 9 standard windows,
ts on and off)"""
import json, sys
import ctranslate2
from common import SMALL, print_lib_info, env_snapshot, to_sv, probe_decode_summaries
from workloads import resolve

LABEL, OUT = sys.argv[1], sys.argv[2]
specs = sys.argv[3:] or ([f"std:{i}" for i in range(9)] + [f"std:{i}:nots" for i in range(9)])
info = print_lib_info()
env = env_snapshot()
model = ctranslate2.models.Whisper(SMALL, device="cuda", compute_type="float16")
rows = []
for spec in specs:
    w, prompt = resolve(spec, 80)
    sv = to_sv(w)
    for beam in [1, 5]:
        gen = lambda: model.generate(sv, [prompt], beam_size=beam, return_scores=True)
        gen()
        r = gen()
        summ, _ = probe_decode_summaries(gen, n=1) if env.get("CT2_CUDA_GRAPHS") else ([], "")
        toks = [int(t) for t in r[0].sequences_ids[0]]
        rows.append(dict(label=LABEL, workload=spec, beam=beam, ntok=len(toks),
                         score=float(r[0].scores[0]), tok_ids=toks, env=env, lib=info["lib"],
                         summaries=summ))
        s = summ[-1] if summ else {}
        print(f"{LABEL} {spec} b{beam}: {len(toks)} tok score={float(r[0].scores[0]):.8g}"
              f" transitions={s.get('transitions')} captures={s.get('captures')}"
              f" replays={s.get('replays')} seg={s.get('segment_replays')}"
              f" disabled={s.get('disabled')}", flush=True)
json.dump(rows, open(OUT, "w"))
print("wrote", OUT, len(rows), flush=True)
