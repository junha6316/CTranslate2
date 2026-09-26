"""W_prev source: eager whisper-small fp16 beam5 ts-on transcripts of the standard windows
1..8 (the neighbours of window 0), text tokens only (id < eot), concatenated in order.
A W_prev prompt is [<|startofprev|>] + last k of these + [sot, en, transcribe] (P = k+3).
Run with all CT2_CUDA_* flags unset (eager default path). Writes prev_prompts.json."""
import json
import ctranslate2
from common import R5T, SMALL, BASE_PROMPT, print_lib_info, standard_windows, to_sv, env_snapshot

print_lib_info()
assert not env_snapshot().get("CT2_CUDA_GRAPHS"), "run eager"
vocab = [l.rstrip("\n") for l in open(f"{SMALL}/vocabulary.txt", encoding="utf-8")]
EOT = vocab.index("<|endoftext|>")
model = ctranslate2.models.Whisper(SMALL, device="cuda", compute_type="float16")
wins = standard_windows(80)
src_ids = []
per_window = []
for i in range(1, len(wins)):
    r = model.generate(to_sv(wins[i]), [BASE_PROMPT], beam_size=5, return_scores=True)
    ids = [int(t) for t in r[0].sequences_ids[0]]
    text = [t for t in ids if t < EOT]
    per_window.append(dict(window=i, ntok=len(ids), ntext=len(text)))
    src_ids += text
    print(f"window {i}: {len(ids)} tok, {len(text)} text tok", flush=True)
assert len(src_ids) >= 223, len(src_ids)
out = dict(source_ids=src_ids, source_tokens=[vocab[t] for t in src_ids], windows=per_window,
           k_values=[64, 128, 192, 223])
json.dump(out, open(f"{R5T}/prev_prompts.json", "w"))
print("wrote prev_prompts.json with", len(src_ids), "text tokens", flush=True)
