import os
import json
from collections import defaultdict
import random

def test_split():
    base_dir = "/Users/blank/Desktop/dating-profile/matcher/python-ml/training-data"
    
    with open(os.path.join(base_dir, "events.json")) as f:
        events = json.load(f)
    print(f"Total events: {len(events)}")
    
    hosted_recs = []
    for e in events:
        if e.get("spaceId"):
            hosted_recs.append((e["id"], e["spaceId"], 1.0, ""))
            
    print(f"Total hosted_by edges (event->space): {len(hosted_recs)}")
    
    # Simulate grouping by src_id (event)
    by_src = defaultdict(list)
    for r in hosted_recs:
        by_src[r[0]].append(r)
        
    print(f"Unique source events: {len(by_src)}")
    
    # Show distribution of edges per event
    lengths = [len(recs) for recs in by_src.values()]
    print(f"Edges per event: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.2f}")
    
    # Simulate split logic
    val_ratio = 0.2
    n_val_total = 0
    
    for sid, recs in list(by_src.items())[:5]:
        if len(recs) == 1:
            n_v = 0
            print(f"Event {sid} has 1 edge -> N_V = {n_v} (NEVER HELD OUT)")
        else:
            n_v = max(1, int(len(recs) * val_ratio))
            if len(recs) - n_v < 1:
                n_v = len(recs) - 1
            print(f"Event {sid} has {len(recs)} edges -> N_V = {n_v}")
        n_val_total += n_v
        
    print(f"Total validation edges that would be generated: {n_val_total}")

if __name__ == '__main__':
    test_split()
