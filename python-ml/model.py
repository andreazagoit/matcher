"""
Single-tower entity encoder.

All entity types (user, event, space) pass through the same encoder network,
producing embeddings in a shared 64-dim space.
Cosine similarity between any two embeddings is meaningful regardless of entity type.
"""

from __future__ import annotations
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from config import FEATURE_DIM, EMBED_DIM, LEARNING_RATE, EPOCHS, BATCH_SIZE, DROPOUT, MODEL_WEIGHTS_PATH

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─── Architecture ──────────────────────────────────────────────────────────────

class EntityEncoder(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))

    def encode(self, features: list[float]) -> torch.Tensor:
        """Encode a single entity from its feature vector."""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return self(x).squeeze(0)


# ─── Contrastive loss ──────────────────────────────────────────────────────────

def contrastive_loss(
    anchor_emb: torch.Tensor,
    item_emb: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    For positive pairs (label=1): maximize cosine similarity.
    For negative pairs (label=0): push similarity below margin.
    """
    sim = F.cosine_similarity(anchor_emb, item_emb)
    pos_loss = (1 - sim) * labels
    neg_loss = torch.clamp(sim - margin, min=0.0) * (1 - labels)
    return (pos_loss + neg_loss).mean()


# ─── Training ──────────────────────────────────────────────────────────────────

def train(
    triplets: list[tuple[list[float], list[float], int]],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> EntityEncoder:
    if not triplets:
        print("No training data. Returning untrained model.")
        model = EntityEncoder().to(device)
        return model

    anchors = torch.tensor([t[0] for t in triplets], dtype=torch.float32)
    items = torch.tensor([t[1] for t in triplets], dtype=torch.float32)
    labels = torch.tensor([t[2] for t in triplets], dtype=torch.float32)

    dataset = TensorDataset(anchors, items, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EntityEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_anchors, batch_items, batch_labels in loader:
            batch_anchors = batch_anchors.to(device)
            batch_items = batch_items.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            anchor_emb = model(batch_anchors)
            item_emb = model(batch_items)
            loss = contrastive_loss(anchor_emb, item_emb, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    return model


# ─── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: EntityEncoder, path: str = MODEL_WEIGHTS_PATH) -> None:
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_WEIGHTS_PATH) -> Optional[EntityEncoder]:
    if not os.path.exists(path):
        return None
    model = EntityEncoder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ─── Similarity search ─────────────────────────────────────────────────────────

def find_similar(
    model: EntityEncoder,
    query_features: "list[float]",
    candidates: "dict[str, tuple[str, list[float]]]",
    target_type: Optional[str] = None,
    limit: int = 10,
) -> "list[dict]":
    """
    Returns top-N most similar entities to the query.

    Args:
        model:          trained EntityEncoder
        query_features: feature vector of the source entity
        candidates:     {entity_id: (entity_type, features)} — all entities to compare against
        target_type:    if set, filter results to this type ('user', 'event', 'space')
        limit:          max results to return

    Returns:
        list of {"id": ..., "type": ..., "score": ...} sorted by score desc
    """
    model.eval()
    query_emb = model.encode(query_features)

    filtered = {
        eid: (etype, fvec)
        for eid, (etype, fvec) in candidates.items()
        if target_type is None or etype == target_type
    }

    if not filtered:
        return []

    ids = list(filtered.keys())
    types = [filtered[eid][0] for eid in ids]
    vecs = torch.tensor(
        [filtered[eid][1] for eid in ids], dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        embs = model(vecs)
        scores = F.cosine_similarity(query_emb.unsqueeze(0), embs)

    scores_list = scores.cpu().tolist()
    ranked = sorted(zip(ids, types, scores_list), key=lambda x: x[2], reverse=True)

    return [
        {"id": eid, "type": etype, "score": round(float(score), 4)}
        for eid, etype, score in ranked[:limit]
    ]
