"""
FastAPI ML Service — embedding inference for new entities.

POST /embed  → generates a 256-dim embedding from raw entity data
              uses forward_single (no graph required for new entities)

Batch re-embedding of existing entities is handled by ml:sync,
which uses forward_graph over the full heterogeneous graph.

Startup:
  uvicorn ml.serve:app --reload
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch

from ml.config import EMBED_DIM
from ml.features import build_user_features, build_event_features, build_space_features
from ml.model import HetEncoder, load_model, device


# ── State ─────────────────────────────────────────────────────────────────────

_model: Optional[HetEncoder] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    if _model:
        print("Model weights loaded.")
    else:
        print("No model weights found — run 'npm run ml:train' first.")
    yield


app = FastAPI(
    title="Matcher ML Service",
    description="Generates 256-dim embeddings for users, events and spaces.",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class UserData(BaseModel):
    tags: list[str] = []
    birthdate: Optional[str] = None                  # "YYYY-MM-DD"
    gender: Optional[str] = None                     # "man" | "woman" | "non_binary"
    relationship_intent: list[str] = []
    smoking: Optional[str] = None
    drinking: Optional[str] = None
    activity_level: Optional[str] = None
    interaction_count: int = 0


class EventData(BaseModel):
    tags: list[str] = []
    starts_at: Optional[str] = None
    avg_attendee_age: Optional[float] = None
    attendee_count: int = 0
    days_until_event: Optional[int] = None
    max_attendees: Optional[int] = None
    is_paid: bool = False
    price_cents: Optional[int] = None


class SpaceData(BaseModel):
    tags: list[str] = []
    avg_member_age: Optional[float] = None
    member_count: int = 0
    event_count: int = 0


class EmbedRequest(BaseModel):
    entity_type: str                          # "user" | "event" | "space"
    user:  Optional[UserData]  = None
    event: Optional[EventData] = None
    space: Optional[SpaceData] = None


class EmbedResponse(BaseModel):
    entity_type: str
    embedding: list[float]                    # EMBED_DIM floats
    model_used: str                           # "hgt" | "untrained"


# ── Feature extraction ────────────────────────────────────────────────────────

def _build_features(req: EmbedRequest) -> tuple[str, list[float]]:
    if req.entity_type == "user":
        if not req.user:
            raise HTTPException(400, "user data required for entity_type=user")
        return "user", build_user_features(
            birthdate=req.user.birthdate,
            tags=req.user.tags,
            gender=req.user.gender,
            relationship_intent=req.user.relationship_intent,
            smoking=req.user.smoking,
            drinking=req.user.drinking,
            activity_level=req.user.activity_level,
            interaction_count=req.user.interaction_count,
        )

    if req.entity_type == "event":
        if not req.event:
            raise HTTPException(400, "event data required for entity_type=event")
        return "event", build_event_features(
            tags=req.event.tags,
            starts_at=req.event.starts_at,
            avg_attendee_age=req.event.avg_attendee_age,
            attendee_count=req.event.attendee_count,
            days_until_event=req.event.days_until_event,
            max_attendees=req.event.max_attendees,
            is_paid=req.event.is_paid,
            price_cents=req.event.price_cents,
        )

    if req.entity_type == "space":
        if not req.space:
            raise HTTPException(400, "space data required for entity_type=space")
        return "space", build_space_features(
            tags=req.space.tags,
            avg_member_age=req.space.avg_member_age,
            member_count=req.space.member_count,
            event_count=req.space.event_count,
        )

    raise HTTPException(400, "entity_type must be 'user', 'event', or 'space'")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """
    Generate an L2-normalised embedding from raw entity data (no DB access).

    Uses forward_single — the encoder-only path of HetEncoder.
    For proper graph-context embeddings on existing entities, run ml:sync.

    Next.js flow for new entities:
      1. POST /embed with raw entity data
      2. Save entity + embedding to DB in a single transaction
    """
    entity_type, features = _build_features(req)

    if _model is not None:
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = _model.forward_single(x, entity_type).squeeze(0).cpu().tolist()
        model_used = "hgt"
    else:
        embedding  = (features + [0.0] * EMBED_DIM)[:EMBED_DIM]
        model_used = "untrained"

    return EmbedResponse(entity_type=entity_type, embedding=embedding, model_used=model_used)
