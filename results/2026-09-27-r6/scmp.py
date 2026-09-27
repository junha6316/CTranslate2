"""Compare scores.py outputs: exact tokens+scores. Usage: scmp.py A.json B.json"""
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
diff = [(x["model"], x["ctype"], x["timestamps"], x["beam"], x["batch"]) for x, y in zip(a, b)
        if x["tok_ids"] != y["tok_ids"] or x["scores"] != y["scores"]]
print(f"{sys.argv[1]} vs {sys.argv[2]}: {len(a)} cfgs, diffs {len(diff)}", diff[:10])
