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

Worth trying, in rough order of expected payoff:

- Keep the decision on the device. The result only drives which tokens get disabled,
  which is already applied through a device-side mask, so the bool never has to reach
  the host.
- Failing that, do every batch row in one kernel and one transfer, instead of two
  round trips per row.
- The `LogSoftMax` removal is still correct and still worth keeping on CPU, where
  there is no round trip and the arithmetic is the cost. It just should not be
  described as a GPU optimization.
