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

from hgt.config import EMBED_DIM
from hgt.features import build_user_features, build_event_features, build_space_features
from hgt.model import HetEncoder, load_model, device


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
    relationshipIntent: list[str] = []
    smoking: Optional[str] = None
    drinking: Optional[str] = None
    activityLevel: Optional[str] = None


class EventData(BaseModel):
    tags: list[str] = []
    startsAt: Optional[str] = None
    avgAttendeeAge: Optional[float] = None
    attendeeCount: int = 0
    daysUntilEvent: Optional[int] = None
    maxAttendees: Optional[int] = None
    isPaid: bool = False
    priceCents: Optional[int] = None


class SpaceData(BaseModel):
    tags: list[str] = []
    avgMemberAge: Optional[float] = None
    memberCount: int = 0
    eventCount: int = 0


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
            relationship_intent=req.user.relationshipIntent,
            smoking=req.user.smoking,
            drinking=req.user.drinking,
            activity_level=req.user.activityLevel,
        )

    if req.entity_type == "event":
        if not req.event:
            raise HTTPException(400, "event data required for entity_type=event")
        return "event", build_event_features(
            tags=req.event.tags,
            starts_at=req.event.startsAt,
            avg_attendee_age=req.event.avgAttendeeAge,
            attendee_count=req.event.attendeeCount,
            days_until_event=req.event.daysUntilEvent,
            max_attendees=req.event.maxAttendees,
            is_paid=req.event.isPaid,
            price_cents=req.event.priceCents,
        )

    if req.entity_type == "space":
        if not req.space:
            raise HTTPException(400, "space data required for entity_type=space")
        return "space", build_space_features(
            tags=req.space.tags,
            avg_member_age=req.space.avgMemberAge,
            member_count=req.space.memberCount,
            event_count=req.space.eventCount,
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
