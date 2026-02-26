"""
Diagnostica del val set e baseline random R@10.
Esecuzione: cd python-ml && .venv/bin/python ../diagnose.py
"""
import math, random
from collections import defaultdict, Counter
from hgt.graph import build_graph_data
from hgt.config import TRAINING_DATA_DIR

print("Carico il grafo…")
bundle = build_graph_data(data_dir=TRAINING_DATA_DIR)
val    = bundle["val_data"]
node_ids = bundle["node_ids"]

pairs  = val["val_pairs"]        # [(atype, aid, itype, iid), ...]
seen   = val["seen_train_by_anchor"]

# ── 1. Quante val interactions ha ogni utente? ─────────────────────────────
by_anchor: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
for atype, aid, itype, iid in pairs:
    by_anchor[(atype, aid)][itype].append(iid)

n_per_anchor = [(k, sum(len(v) for v in d.values())) for k, d in by_anchor.items()]
counts = [n for _, n in n_per_anchor]
print(f"\n── Val anchors totali: {len(counts)}")
print(f"   Val items per anchor — min={min(counts)}  median={sorted(counts)[len(counts)//2]}  max={max(counts)}  media={sum(counts)/len(counts):.2f}")
c = Counter(counts)
print(f"   Distribuzione: { {k: c[k] for k in sorted(c)[:10]} }")

# ── 2. Breakdown per task ──────────────────────────────────────────────────
task_counts: dict[tuple, int] = defaultdict(int)
for atype, aid, itype, iid in pairs:
    task_counts[(atype, itype)] += 1
print(f"\n── Pairs per task:")
for (a, b), n in sorted(task_counts.items()):
    n_anchors = len({aid for (a2, aid, b2, _) in pairs if a2==a and b2==b})
    print(f"   {a}->{b}: {n} pairs, {n_anchors} anchors, media {n/n_anchors:.2f} items/anchor")

# ── 3. Random baseline R@10 ───────────────────────────────────────────────
print(f"\n── Random baseline R@10 (K=10):")
n_events = len(node_ids["event"])
n_spaces  = len(node_ids["space"])
n_users   = len(node_ids["user"])

for (a, b), _ in sorted(task_counts.items()):
    catalog_size = {"event": n_events, "space": n_spaces, "user": n_users}[b]
    # expected recall@10 = 10 / catalog_size  (1 relevant item)
    baseline = 10 / catalog_size
    print(f"   {a}->{b}: catalog={catalog_size}  random R@10 ≈ {baseline:.4f} ({baseline*100:.2f}%)")

# ── 4. Quanti anchor hanno ≥2 val items (metriche affidabili)? ────────────
reliable = sum(1 for n in counts if n >= 2)
print(f"\n── Anchor con ≥2 val items (stima affidabile): {reliable}/{len(counts)} ({100*reliable/len(counts):.1f}%)")

# ── 5. Overlap train/val ──────────────────────────────────────────────────
overlaps = 0
for (atype, aid), by_type in by_anchor.items():
    train_seen = seen.get((atype, aid), set())
    for itype, iids in by_type.items():
        for iid in iids:
            if iid in train_seen:
                overlaps += 1
print(f"\n── Val items che compaiono anche in train: {overlaps} (dovrebbe essere 0)")

print("\nDone.")
