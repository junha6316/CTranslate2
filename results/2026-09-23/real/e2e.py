"""End-to-end faster-whisper transcription of physicsworks.wav (203 s), one build per run.
Sequential WhisperModel.transcribe (library defaults: beam 5, timestamps on, conditioned
on previous text) and BatchedInferencePipeline batch_size=8 (defaults: timestamps off), cut into fixed 30 s clips.
Usage: e2e.py LABEL OUT.json [quick]
"""
import json, statistics, sys, time
from faster_whisper import WhisperModel, BatchedInferencePipeline, decode_audio
LABEL, OUT = sys.argv[1], sys.argv[2]
QUICK = len(sys.argv) > 3
REPS = 1 if QUICK else 3
audio = decode_audio("/opt/audio/physicsworks.wav", sampling_rate=16000)
# Fixed 30 s clips in seconds, instead of VAD (needs onnxruntime, not installed).
dur = len(audio) / 16000
clips = [{"start": float(s), "end": min(s + 30.0, dur)} for s in range(0, int(dur), 30)]
rows = []
for mname in (["small"] if QUICK else ["small", "large-v3"]):
    for ctype in ["float16", "int8_float16"]:
        m = WhisperModel(f"/opt/models/faster-whisper-{mname}", device="cuda", compute_type=ctype)
        bp = BatchedInferencePipeline(m)
        for mode in ["sequential", "batched8"]:
            def run():
                if mode == "sequential":
                    segs, _ = m.transcribe(audio, language="en", vad_filter=False)
                else:
                    segs, _ = bp.transcribe(audio, language="en", batch_size=8, clip_timestamps=clips)
                return " ".join(s.text.strip() for s in segs)
            run()  # warm-up
            t, text = [], ""
            for _ in range(REPS):
                t0 = time.perf_counter(); text = run(); t.append(time.perf_counter() - t0)
            med = statistics.median(t)
            rows.append(dict(label=LABEL, model=mname, ctype=ctype, mode=mode,
                             median_s=round(med, 3), rtf=round(203.3 / med, 1), text=text))
            print(f"{LABEL} {mname} {ctype} {mode}: {med:.2f} s ({203.3/med:.1f}x realtime), {len(text)} chars", flush=True)
        del bp, m
json.dump(rows, open(OUT, "w"), indent=1)
print("wrote", OUT)
