import json
def load(f):
    return {(r["ts"],r["beam"]):tuple(r["tok_ids"]) for r in json.load(open(f))}
base=load("r3base_eager.json"); r4e=load("r4_eager.json"); r4g=load("r4_graphs_default.json")
tiers={t:load("r4_graphs_r%d.json"%t) for t in [96,128,160,256]}
bflash=load("r3base_eager_flash.json"); r4flash=load("r4_eager_flash.json")
def cmp(a,b,name):
    same=all(a[k]==b[k] for k in a)
    tag="IDENTICAL" if same else "DIFF"
    print(name+": "+tag)
    if not same:
        for k in sorted(a):
            if a[k]!=b[k]:
                print("   MISMATCH ts=%s b=%s len %d vs %d"%(k[0],k[1],len(a[k]),len(b[k])))
cmp(base,r4e,"r3base_eager vs r4_eager (must be identical)")
cmp(r4e,r4g,"r4_eager vs r4_graphs_default")
for t in [96,128,160,256]:
    cmp(r4e,tiers[t],"r4_eager vs r4_graphs_r%d"%t)
cmp(bflash,r4flash,"r3base_eager_flash vs r4_eager_flash")
