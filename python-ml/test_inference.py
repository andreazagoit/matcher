import torch
import torch.nn.functional as F
from hgt.train import load_model_and_graph
from hgt.model import device

def diagnose_zeros():
    model, bundle = load_model_and_graph("/Users/blank/Desktop/dating-profile/matcher/python-ml/hgt_weights.pt")
    val_data = bundle["val_data"]
    train_data = bundle["train_data"].to(device)
    node_ids = bundle["node_ids"]
    
    # ── Get embeddings using the full graph ───────────────────────────────────
    with torch.no_grad():
        emb = model.forward_graph(
            {t: train_data[t].x for t in train_data.node_types},
            {et: train_data[et].edge_index for et in train_data.edge_types},
        )
    
    print("Testing zero-recall paths: event->space, space->event")
    
    target_tasks = ["event->space", "space->event"]
    results = {k: [] for k in target_tasks}
    
    id_to_idx = {
        t: {iid: i for i, iid in enumerate(ids)}
        for t, ids in node_ids.items()
    }
    
    for (atype, aid, itype, iid) in val_data["val_pairs"]:
        task_name = f"{atype}->{itype}"
        if task_name in target_tasks and len(results[task_name]) < 5:
            aidx = id_to_idx.get(atype, {}).get(aid)
            iidx = id_to_idx.get(itype, {}).get(iid)
            if aidx is None or iidx is None:
                continue
            
            a_emb = emb[atype][aidx].unsqueeze(0).cpu() # [1, D]
            mat = emb[itype].cpu()                      # [N, D]
            
            sims = F.cosine_similarity(a_emb, mat).clone()
            
            seen = val_data.get("seen_train_by_anchor", {}).get((atype, aid), set())
            
            # Mask seen items
            for exc in seen | {aid}:
                idx = id_to_idx[itype].get(exc)
                if idx is not None:
                    sims[idx] = -float("inf")
            
            target_score = sims[iidx].item()
            rank = (sims > target_score).sum().item() + 1
            total = mat.size(0)
            
            results[task_name].append({
                "target_score": target_score,
                "rank": rank,
                "total": total,
                "max_score": sims.max().item(),
                "min_score": sims.min().item(),
                "mean_score": sims[sims > -100].mean().item() # Ignore infinities
            })
    
    for task_name, items in results.items():
        print(f"\n=== {task_name} ===")
        if not items:
            print("No test cases found.")
        for i, res in enumerate(items):
            print(f"Sample {i+1}: Rank {res['rank']:4d}/{res['total']:4d} | Target Score: {res['target_score']:.4f} (Max: {res['max_score']:.4f}, Mean: {res['mean_score']:.4f})")

if __name__ == '__main__':
    diagnose_zeros()
