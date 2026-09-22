# Where Whisper decoding spends GPU time

Nsight Systems trace of 10 beam-5 decodes of 150s audio with `whisper-small`,
float16, timestamps on, on an A10G. Profiled after the timestamp gate landed.

## The headline

| | |
| --- | --- |
| GPU kernel busy | 2.911 s |
| host CUDA runtime API | 4.387 s |
| kernel launches | 572,666 |
| mean kernel duration | 5.08 us |
| mean `cudaLaunchKernel` | 5.06 us |

**The host spends more time in the CUDA API than the GPU spends computing, and the
average kernel takes exactly as long to launch as it does to run.** This decode is
launch-bound, not compute-bound. Making individual kernels faster cannot help much
while there are half a million of them.

## Three levers, largest first

### 1. Too many tiny kernels

Six `cub` elementwise `for_each` kernels account for about 540 ms of GPU time across
roughly 265,000 instances — 1.7 to 3.2 us each. At ~5 us of launch cost apiece, they
cost more on the host than they do on the device.

These are elementwise operations: bias adds, activations, scaling, masking. Fusing
them into their neighbouring GEMM epilogues, or into each other, removes both the
kernel and its launch.

The other route is CUDA Graphs. A decode step replays an almost identical sequence of
launches every time, which is the case graph capture exists for: capture one step,
replay it, and the per-launch host cost disappears. It also removes most of lever 2
and lever 3 as a side effect. This is the single highest-value change visible in the
trace.

### 2. Allocation churn

281,121 `cudaMallocAsync` and 281,121 `cudaFreeAsync` calls, together about 849 ms of
host time, roughly 19% of all CUDA API time. That is one allocate-free pair for
essentially every other kernel launch. Workspace and intermediate tensors are being
created and destroyed inside the decode loop rather than held across steps.

### 3. Small transfers that remain

| direction | count | total |
| --- | --- | --- |
| host to device | 7,988 | 634 MB, but median ~0 and max 79.7 MB (model load) |
| device to host | 2,548 | 0.076 MB total, about 30 bytes each |

The device-to-host traffic is 255 transfers per decode of about 30 bytes. Byte volume
is irrelevant; each one is a round trip. This is the same shape of problem the
timestamp gate fixed, in whatever still reads scalars back per step. Worth finding the
callers before optimizing anything else small.

Most host-to-device copies are also tiny and are likely the `DisableTokens` flat index
arrays, which are built on the host and uploaded every step.

## What is already handled

`timestamp_gate_kernel` shows 1,196 instances at 37.8 us, 1.6% of GPU time. It replaced
about 576 us per call of blocking round trips, so it is a large net win even at that
cost. Widening it from 256 to 1024 threads is in; it is still only 1 block per checked
row, so at beam 5 it uses 5 of 80 SMs. Splitting a row across blocks with a two-stage
reduction would shrink it further, but at 1.6% that is not where the time is.

## Not worth chasing

Individual GEMM kernels look healthy. The top entry,
`cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn_align8`, is 17.6% of GPU
time over 61,308 instances at 8.4 us, which is reasonable for these shapes. There is no
single slow kernel to fix; the cost is spread across a very large number of small ones.
