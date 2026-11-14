import json, statistics as st
from collections import defaultdict
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
base = {}
for p in ("results/stage0/G-A/summary.jsonl", "results/colab/fp32/summary.jsonl", "results/stage0/G-D/summary.jsonl"):
    for r in load(p):
        if r["alpha"] == 1.0 and r["seed"] in (0, 1, 2) and r["status"] == "ok":
            base[(r["width"], r["gamma"], r["seed"])] = r
pil = [r for r in load("results/stage0/G-S1P/summary.jsonl")]
print("non-ok:", [r["run_id"] for r in pil if r["status"] != "ok"])
by = defaultdict(list)
for r in pil: by[(r["width"], r["penalty"], r["gamma"], r["lam"])].append(r)
gstar = {15: -0.55, 25: -1.0, 63: -2.0}
for w in (15, 25, 63):
    print(f"\n=== width {w}  (seeds 0-2 means; baseline at gamma* = {gstar[w]})")
    for g in (0.0, gstar[w]):
        b = [base[(w, g, s)] for s in (0, 1, 2) if (w, g, s) in base]
        print(f"baseline gamma {g:+.2f}: loss {st.mean(r['test_loss'] for r in b):.3e}  overlap {st.mean(r['mso_all'] for r in b):.4f}  kept {st.mean(r['phi_half'] for r in b):.3f}  (n={len(b)})")
    L = st.mean(base[(w, gstar[w], s)]["test_loss"] for s in (0, 1, 2))
    for pen, g in (("plain", 0.0), ("plain", gstar[w]), ("weighted", gstar[w])):
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            rs = by[(w, pen, g, lam)]
            l = st.mean(r["test_loss"] for r in rs)
            sd = st.pstdev([r["test_loss"] for r in rs]) / l
            print(f"{pen:8s} g={g:+.2f} lam={lam:g}: loss {l:.3e} ({100*(l/L-1):+6.1f}% vs baseline g*)  sd {100*sd:3.0f}%  overlap {st.mean(r['mso_all'] for r in rs):.4f}  kept {st.mean(r['phi_half'] for r in rs):.3f}")
