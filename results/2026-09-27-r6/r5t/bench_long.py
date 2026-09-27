"""Paired wall bench: every workload x beam, median of 7 (2 warmup) of the total
generate() wall, one config per process (flags from env). After each timed row, a probe
at debug level records the runner's per-decode summary (tier transitions, captures,
replays per tier segment, disable reason) -- the probe decodes are not timed.
Usage: bench_long.py LABEL OUT.json [--model small|large-v3] [--beams 5,1] SPEC ..."""
import argparse, json
import ctranslate2
from common import SMALL, LARGE, print_lib_info, env_snapshot, to_sv, timed, probe_decode_summaries
from workloads import resolve

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("out")
ap.add_argument("specs", nargs="+")
ap.add_argument("--model", default="small")
ap.add_argument("--beams", default="5,1")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--warmup", type=int, default=2)
a = ap.parse_args()

info = print_lib_info()
env = env_snapshot()
path, n_mels = (SMALL, 80) if a.model == "small" else (LARGE, 128)
model = ctranslate2.models.Whisper(path, device="cuda", compute_type="float16")
graphs = bool(env.get("CT2_CUDA_GRAPHS"))
rows = []
for beam in [int(b) for b in a.beams.split(",")]:
    for spec in a.specs:
        w, prompt = resolve(spec, n_mels)
        sv = to_sv(w)
        gen = lambda: model.generate(sv, [prompt], beam_size=beam, return_scores=True)
        med, mn, mx, r = timed(gen, a.reps, a.warmup)
        summ = probe_decode_summaries(gen, n=1)[0] if graphs else []
        toks = [int(t) for t in r[0].sequences_ids[0]]
        s = summ[-1] if summ else {}
        row = dict(label=a.label, model=a.model, workload=spec, beam=beam,
                   median_ms=round(med, 2), min_ms=round(mn, 2), max_ms=round(mx, 2),
                   ntok=len(toks), score=float(r[0].scores[0]), tok_ids=toks,
                   prompt_len=len(prompt) - 1, env=env, lib=info["lib"], summaries=summ)
        rows.append(row)
        print(f"{a.label} {a.model} {spec} b{beam}: {med:.1f} ms (min {mn:.1f} max {mx:.1f})"
              f" {len(toks)} tok score={row['score']:.6g} | tr={s.get('transitions')}"
              f" cap={s.get('captures')} rep={s.get('replays')} seg={s.get('segment_replays')}"
              f" tiers=[{s.get('tiers', '')}] dis={s.get('disabled')}", flush=True)
json.dump(rows, open(a.out, "w"))
print("wrote", a.out, len(rows), flush=True)
