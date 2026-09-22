# Where the timestamp rule actually spends GPU time

Measured on an A10G with CUDA 12.8, `whisper-small` and `large-v3`, float16.

## The question

Commit `fe8a314f` removed a full-vocabulary `LogSoftMax` from `ApplyTimestampRules`
(`src/models/whisper.cc:809-845`). The expectation was that this would help on GPU,
where it was also thought to force host synchronization. Measured against upstream
v4.8.2 it changes nothing: 32 paired cases, median -0.3% with timestamps on, all
inside a +/-3% noise band.

## What the code costs, per call

| operation | us/call |
| --- | --- |
| `LogSoftMax([1, 51866])` — what the commit removed | 19.82 |
| `primitives<CUDA>::max(50257 elements)` | 197.26 |
| `primitives<CUDA>::max(1501 elements)` | 190.99 |
| `primitives<CUDA>::max(16 elements)` | 189.59 |
| `primitives<CUDA>::logsumexp(1501 elements)` | 378.79 |
| `primitives<CUDA>::logsumexp(16 elements)` | 379.10 |

## Reading it

**Reducing 16 elements costs the same as reducing 50257.** The work is not the
arithmetic, it is the device-to-host round trip: these reductions return a `float` to
the host, so each one blocks. `logsumexp` costs about twice `max` because it
synchronizes twice internally.

**The kernel the commit removed never synchronized.** `LogSoftMax` writes its result
back to device memory and returns immediately, so 51,866 elements cost 19.82us — ten
times less than reducing 16 elements through a host round trip.

So the commit removed the one piece of that function that was already cheap, and left
the two blocking reductions, which cost about 576us per call against its 20us. It
addressed roughly 3% of the cost of the code it touched. Over a 27-step decode the
removed kernel is about 0.5ms against a 98-418ms total, which is exactly why it
disappears into the noise.

This also lines up with the end-to-end numbers: turning timestamps off saves 9ms on
`small`/beam1 and 44ms on `large-v3`/beam5, and 576us per step per checked batch row
lands in that range.

## Where the time actually is

`should_sample_timestamp` compares a max against a logsumexp and returns a `bool` to
host code, once per batch row per step. The comparison itself is trivial; the cost is
entirely in shipping two scalars back to the CPU and waiting.

Keeping the decision on the device is the fix, and it is now implemented: see below.
The `LogSoftMax` removal is still correct and still worth keeping on CPU, where there
is no round trip and the arithmetic is the cost. It just should not be described as a
GPU optimization.

## The fix: `ops::TimestampGate`

One CUDA kernel, one block per row, that reduces the text tokens to a max, reduces the
timestamp tokens to a logsumexp, compares them in shared memory, and masks the text
tokens when the timestamp side wins. Nothing is read back to the host. The CPU path
keeps the old per-row logic, which is the right shape there.

### It produces identical tokens

Token ids were dumped from this build and from upstream v4.8.2 across 32 cases
(`small` and `large-v3`, float16 and int8_float16, four audio inputs, beam 1 and 5,
timestamps on). **All 32 match exactly.** The CUDA test suite is 175 passed, 3 skipped,
0 failed.

### It is worth 0.7% to 20%

Against upstream v4.8.2, Flash Attention off in both:

| axis | cases | median | range |
| --- | --- | --- | --- |
| timestamps on | 16 | **-4.5%** | -20.2% .. -0.7% |
| timestamps off | 16 | +0.2% | -2.7% .. +1.9% |

Timestamps off is the control: the rule never runs there, and that axis does not move,
which is what attributes the gain to this change rather than to drift.

The gain tracks how many rows get checked per step, so it grows with beam size and with
the number of decode steps. Total latency, timestamps on, beam 5:

| config | upstream | pre-gate | gate |
| --- | --- | --- | --- |
| small / float16 / 150s | 506.5ms | 503.3ms | **404.0ms** |
| small / int8_float16 / 150s | 605.1ms | 611.4ms | **497.5ms** |
| large-v3 / float16 / 150s | 1239.0ms | 1242.4ms | **1151.2ms** |
| large-v3 / int8_float16 / 150s | 1214.4ms | 1224.3ms | **1135.2ms** |

Put another way, what enabling timestamps costs:

| config | upstream | gate |
| --- | --- | --- |
| small / float16 / 150s / beam5 | +174.5ms | +73.3ms |
| small / int8_float16 / 150s / beam5 | +241.3ms | +130.5ms |
| large-v3 / float16 / 150s / beam5 | +255.3ms | +164.0ms |
| large-v3 / int8_float16 / 150s / beam5 | +226.3ms | +153.9ms |

Roughly 40% of the cost of the timestamp rule is gone. What remains is the rest of the
rule, which still walks sequences on the host.

Raw measurements are in `bench_gate.json`.
