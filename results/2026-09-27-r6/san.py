"""One decode of a G6 cell (flags from env). Usage: san.py CELL
  flash_b5  flash, beam5, std:0 ts-on        flash_b4  flash, beam5, batch4 (shrink)
  b5        eager/graphs beam5 std:0 ts-on   b4        beam5 batch4 (shrink)
  dense1    beam5 dense:1 (tiers crossings)"""
import sys
import numpy as np, ctranslate2
sys.path.insert(0, "/opt/real"); sys.path.insert(0, "/opt/real/r6/r5t")
from bench_batch_windows import windows
cell = sys.argv[1]
flash = cell.startswith("flash")
m = ctranslate2.models.Whisper("/opt/models/faster-whisper-small", device="cuda", compute_type="float16", flash_attention=flash)
prompt = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
if cell == "dense1":
    ws = [np.load("/opt/real/r6/r5t/dense_80.npy")[1]]
else:
    w = windows(80); ws = w[:4] if cell.endswith("b4") else w[:1]
sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(ws)))
r = m.generate(sv, [prompt] * len(ws), beam_size=5, return_scores=True)
print("SANCELL", cell, [len(x.sequences_ids[0]) for x in r], [round(float(x.scores[0]), 6) for x in r], flush=True)
