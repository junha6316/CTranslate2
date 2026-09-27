"""nsys workload: whisper-small fp16 beam5 ts-on batch1, window 0, exactly N decodes.
Usage: nsys_gen.py [N] [SPEC] [BEAM]"""
import sys
import ctranslate2
from common import SMALL, print_lib_info, to_sv
from workloads import resolve

N = int(sys.argv[1]) if len(sys.argv) > 1 else 7
spec = sys.argv[2] if len(sys.argv) > 2 else "std:0"
beam = int(sys.argv[3]) if len(sys.argv) > 3 else 5
print_lib_info()
m = ctranslate2.models.Whisper(SMALL, device="cuda", compute_type="float16")
w, prompt = resolve(spec, 80)
sv = to_sv(w)
for _ in range(N):
    r = m.generate(sv, [prompt], beam_size=beam, return_scores=True)
print("done ntok", len(r[0].sequences_ids[0]), flush=True)
