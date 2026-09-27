"""Shared helpers for the round-5 capacity-tier gates (run on the A10G box)."""
import contextlib, json, os, re, statistics, sys, tempfile, time
import numpy as np

sys.path.insert(0, "/opt/real")
R5T = "/opt/real/r6/r5t"
SMALL = "/opt/models/faster-whisper-small"
LARGE = "/opt/models/faster-whisper-large-v3"
BASE_PROMPT = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]

ENV_KEYS = ["CT2_CUDA_GRAPHS", "CT2_CUDA_PAD_KV", "CT2_CUDA_PREALLOC_KV",
            "CT2_CUDA_GRAPHS_RESERVE", "CT2_CUDA_GRAPHS_TIERS",
            "CT2_CUDA_GRAPHS_TIERS_REENTRY", "CT2_CUDA_GRAPHS_CHECK",
            "CT2_CUDA_GRAPHS_FAULT", "CT2_CUDA_POOL_RELEASE_THRESHOLD",
            "CT2_CUDA_ALLOC_DEBUG", "CT2_VERBOSE", "CT2_CUDA_ALLOCATOR"]


def env_snapshot():
    return {k: os.environ.get(k) for k in ENV_KEYS if os.environ.get(k) is not None}


def lib_info():
    import ctranslate2
    lib = None
    with open("/proc/self/maps") as f:
        for line in f:
            if "libctranslate2.so" in line:
                lib = os.path.realpath(line.split()[-1])
                break
    return {"pkg": ctranslate2.__file__, "lib": lib}


def print_lib_info():
    info = lib_info()
    print(f"ctranslate2.__file__ = {info['pkg']}  lib = {info['lib']}", flush=True)
    return info


def standard_windows(n_mels=80):
    from bench_batch_windows import windows
    return windows(n_mels)


def dense_windows(n_mels=80):
    path = f"{R5T}/dense_{n_mels}.npy"
    return [w for w in np.load(path)]


def prev_prompts():
    return json.load(open(f"{R5T}/prev_prompts.json"))


def to_sv(window):
    import ctranslate2
    return ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack([window])))


SUMMARY_RE = re.compile(
    r"decode summary \(steps=(\d+), captures=(\d+), replays=(\d+), transitions=(\d+),"
    r" segment_replays=\[([\d,]*)\], tiers=\[([^\]]*)\], phase=(\w+), disabled=(.*)\)")


def parse_summaries(text):
    out = []
    for m in SUMMARY_RE.finditer(text):
        out.append(dict(steps=int(m.group(1)), captures=int(m.group(2)),
                        replays=int(m.group(3)), transitions=int(m.group(4)),
                        segment_replays=[int(x) for x in m.group(5).split(",") if x],
                        tiers=m.group(6), phase=m.group(7), disabled=m.group(8)))
    return out


@contextlib.contextmanager
def capture_stderr():
    """Redirects fd 2 (spdlog's sink, all threads) into a temp file; yields a getter."""
    sys.stderr.flush()
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    box = {}
    try:
        yield box
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        tmp.seek(0)
        box["text"] = tmp.read().decode("utf-8", "replace")
        tmp.close()


def probe_decode_summaries(gen, n=2):
    """Runs gen() n+1 times at debug level and returns the runner's per-decode
    summaries (a decode's summary is logged at the next decode boundary, i.e. at the
    start of the next generate), plus the raw log text."""
    import ctranslate2, logging
    old = ctranslate2.get_log_level()
    with capture_stderr() as box:
        ctranslate2.set_log_level(logging.DEBUG)
        try:
            for _ in range(n + 1):
                gen()
        finally:
            ctranslate2.set_log_level(old)
    text = box["text"]
    summaries = parse_summaries(text)
    for s in summaries:
        s["probe_mismatch_lines"] = text.count("logits mismatch")
    return summaries, text


def timed(gen, reps=7, warmup=2):
    for _ in range(warmup):
        gen()
    t = []
    r = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = gen()
        t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t), min(t), max(t), r
