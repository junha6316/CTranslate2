"""Diff two nsys stats CSV sets (cuda_api_sum + cuda_gpu_kern_sum).
Usage: nsys_diff.py LABEL_A LABEL_B   (files /opt/real/r6/nsys_<label>_cuda_*.csv)
Counts are totals over the profiled reps; per-decode = / reps (passed via R6_REPS, default 11).
"""
import csv, os, sys
A, B = sys.argv[1], sys.argv[2]
REPS = int(os.environ.get("R6_REPS", "11"))
D = "/opt/real/r6"

def load(label, rep):
    path = f"{D}/nsys_{label}_{rep}.csv"
    rows = list(csv.DictReader(open(path)))
    return {r["Name"]: r for r in rows}

def num(r, k):
    return float(r.get(k, 0) or 0) if r else 0.0

api_a, api_b = load(A, "cuda_api_sum"), load(B, "cuda_api_sum")
tot_a = sum(num(r, "Total Time (ns)") for r in api_a.values())
tot_b = sum(num(r, "Total Time (ns)") for r in api_b.values())
print(f"== CUDA API ({A} -> {B}), per decode = total/{REPS}")
print(f"{'api':34s} {'calls A':>9s} {'calls B':>9s} {'ms A':>9s} {'ms B':>9s} {'dms':>9s}")
names = sorted(set(api_a) | set(api_b),
               key=lambda n: -abs(num(api_b.get(n), "Total Time (ns)") - num(api_a.get(n), "Total Time (ns)")))
for n in names[:18]:
    ra, rb = api_a.get(n), api_b.get(n)
    ca, cb = num(ra, "Num Calls"), num(rb, "Num Calls")
    ta, tb = num(ra, "Total Time (ns)") / 1e6, num(rb, "Total Time (ns)") / 1e6
    print(f"{n[:34]:34s} {ca/REPS:9.1f} {cb/REPS:9.1f} {ta/REPS:9.3f} {tb/REPS:9.3f} {(tb-ta)/REPS:+9.3f}")
for n in ["cudaMallocAsync", "cudaFreeAsync", "cudaMemcpyAsync", "cudaLaunchKernel",
          "cudaStreamSynchronize", "cudaMalloc", "cudaFree"]:
    ra, rb = api_a.get(n), api_b.get(n)
    print(f"KEY {n:24s} calls/decode {num(ra,'Num Calls')/REPS:8.1f} -> {num(rb,'Num Calls')/REPS:8.1f}"
          f"  ms/decode {num(ra,'Total Time (ns)')/1e6/REPS:8.3f} -> {num(rb,'Total Time (ns)')/1e6/REPS:8.3f}"
          f"  avg_us {num(ra,'Avg (ns)')/1e3:8.1f} -> {num(rb,'Avg (ns)')/1e3:8.1f}")
print(f"TOTAL CUDA API ms/decode {tot_a/1e6/REPS:.3f} -> {tot_b/1e6/REPS:.3f} ({(tot_b-tot_a)/1e6/REPS:+.3f})")

ka, kb = load(A, "cuda_gpu_kern_sum"), load(B, "cuda_gpu_kern_sum")
ktot_a = sum(num(r, "Total Time (ns)") for r in ka.values())
ktot_b = sum(num(r, "Total Time (ns)") for r in kb.values())
print(f"\n== GPU kernels, total kernel ms/decode {ktot_a/1e6/REPS:.3f} -> {ktot_b/1e6/REPS:.3f} "
      f"({(ktot_b-ktot_a)/1e6/REPS:+.3f})")
def top(k):
    return sorted(k, key=lambda n: -num(k[n], "Total Time (ns)"))[:10]
for tag, k, other in [("top10 " + A, ka, kb), ("top10 " + B, kb, ka)]:
    print(f"-- {tag}")
    for n in top(k):
        r, o = k[n], other.get(n)
        print(f"  {num(r,'Total Time (ns)')/1e6/REPS:8.3f} ms/dec  calls/dec {num(r,'Instances')/REPS:7.1f}"
              f"  avg_us {num(r,'Avg (ns)')/1e3:8.2f}  (other: {num(o,'Total Time (ns)')/1e6/REPS:8.3f} ms,"
              f" {num(o,'Instances')/REPS:7.1f}, {num(o,'Avg (ns)')/1e3:8.2f}us)  {n[:70]}")
new = [n for n in kb if n not in ka]
gone = [n for n in ka if n not in kb]
print(f"-- kernels only in {B}: {[(n[:60], round(num(kb[n],'Total Time (ns)')/1e6/REPS,3)) for n in new][:8]}")
print(f"-- kernels only in {A}: {[(n[:60], round(num(ka[n],'Total Time (ns)')/1e6/REPS,3)) for n in gone][:8]}")
