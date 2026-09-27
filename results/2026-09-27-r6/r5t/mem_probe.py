"""Peak process GPU memory (NVML, sampled every 2 ms) over a set of workloads, beam5 then
greedy, 2 decodes each. Flags from env.
Usage: mem_probe.py LABEL [--model small|large-v3] SPEC ..."""
import argparse, os, threading, time
import pynvml
import ctranslate2
from common import SMALL, LARGE, print_lib_info, env_snapshot, to_sv
from workloads import resolve

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("specs", nargs="+")
ap.add_argument("--model", default="small")
a = ap.parse_args()

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
pid = os.getpid()
peak = [0]
stop = [False]


def used_mb():
    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        if p.pid == pid and p.usedGpuMemory is not None:
            return p.usedGpuMemory / 2**20
    return 0


def poll():
    while not stop[0]:
        peak[0] = max(peak[0], used_mb())
        time.sleep(0.002)


print_lib_info()
path, n_mels = (SMALL, 80) if a.model == "small" else (LARGE, 128)
model = ctranslate2.models.Whisper(path, device="cuda", compute_type="float16")
t = threading.Thread(target=poll, daemon=True)
t.start()
per = {}
for beam in [5, 1]:
    for spec in a.specs:
        w, prompt = resolve(spec, n_mels)
        sv = to_sv(w)
        before = peak[0]
        for _ in range(2):
            model.generate(sv, [prompt], beam_size=beam, return_scores=True)
        time.sleep(0.02)
        per[(spec, beam)] = peak[0]
stop[0] = True
t.join()
env = env_snapshot()
print(f"MEM {a.label} model={a.model} env={env} peak_used={peak[0]:.0f} MB", flush=True)
for (spec, beam), v in per.items():
    print(f"  running peak after {spec} b{beam}: {v:.0f} MB", flush=True)
