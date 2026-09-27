"""Workload specs shared by the gate scripts.
  std:I[:nots]  standard window I (bench_batch_windows order), ts-on unless :nots
  dense:I       dense (time-compressed) window I, ts-on
  prev:K        window 0 with [<|startofprev|>] + last K eager text tokens + base prompt
  pp:P          same as prev with K = P - 3 (P = forwarded prompt length)
"""
from common import BASE_PROMPT, standard_windows, dense_windows, prev_prompts

_cache = {}


def _std(n_mels):
    key = ("std", n_mels)
    if key not in _cache:
        _cache[key] = standard_windows(n_mels)
    return _cache[key]


def _dense(n_mels):
    key = ("dense", n_mels)
    if key not in _cache:
        _cache[key] = dense_windows(n_mels)
    return _cache[key]


def resolve(spec, n_mels=80):
    """Returns (window_array, prompt) for a workload spec. Prompts are token strings:
    small and large-v3 share the text tokens (ids < 50257) but not the special ids."""
    parts = spec.split(":")
    kind = parts[0]
    if kind == "std":
        w = _std(n_mels)[int(parts[1])]
        prompt = list(BASE_PROMPT)
        if len(parts) > 2 and parts[2] == "nots":
            prompt.append("<|notimestamps|>")
        return w, prompt
    if kind == "dense":
        return _dense(n_mels)[int(parts[1])], list(BASE_PROMPT)
    if kind in ("prev", "pp"):
        pp = prev_prompts()
        k = int(parts[1]) if kind == "prev" else int(parts[1]) - 3
        src = pp["source_tokens"]
        return _std(n_mels)[0], ["<|startofprev|>"] + src[-k:] + list(BASE_PROMPT)
    raise ValueError(spec)
