"""
FastAPI ML Service — embedding inference only.

Endpoints:
  POST /embed  — generate a 64-dim embedding from raw entity data (stateless, no DB)

Training is handled offline:
  1. npm run ml:export   → writes python-ml/training-data/*.json
  2. npm run ml:train    → python3 train.py  (reads JSON, trains, saves model_weights.pt)
  3. Restart this server to pick up the new weights.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import EMBED_DIM
from features import build_user_features, build_event_features, build_space_features
from model import HetEncoder, load_model


# ─── State ─────────────────────────────────────────────────────────────────────

_model: Optional[HetEncoder] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    if _model:
        print("Loaded existing model weights.")
    else:
        print("No model weights found. Run 'npm run ml:train' to train.")
    yield


app = FastAPI(
    title="Matcher ML Service — HGT-lite",
    description="Generates 64-dim embeddings for users, events and spaces.",
    lifespan=lifespan,
)


# ─── Request / Response schemas ────────────────────────────────────────────────

class UserData(BaseModel):
    tag_weights: dict[str, float] = {}
    birthdate: Optional[str] = None                  # "YYYY-MM-DD"
    gender: Optional[str] = None                     # "man" | "woman" | "non_binary"
    relationship_intent: list[str] = []              # ["serious_relationship", ...]
    smoking: Optional[str] = None                    # "never" | "sometimes" | "regularly"
    drinking: Optional[str] = None
    activity_level: Optional[str] = None             # "sedentary" | ... | "very_active"
    interaction_count: int = 0                       # events attended + spaces joined


class EventData(BaseModel):
    tags: list[str] = []
    starts_at: Optional[str] = None                  # ISO timestamp
    avg_attendee_age: Optional[float] = None         # real for past, registered for upcoming
    attendee_count: int = 0                          # status='attended' (past) or 'going' (upcoming)
    days_until_event: Optional[int] = None           # days from today; negative = past
    max_attendees: Optional[int] = None              # events.max_attendees (None = no cap)
    is_paid: bool = False                            # True if price > 0
    price_cents: Optional[int] = None                # price in cents


class SpaceData(BaseModel):
    tags: list[str] = []
    avg_member_age: Optional[float] = None
    member_count: int = 0
    event_count: int = 0                             # events in this space


class EmbedRequest(BaseModel):
    entity_type: str                                 # "user" | "event" | "space"
    user: Optional[UserData] = None
    event: Optional[EventData] = None
    space: Optional[SpaceData] = None


class EmbedResponse(BaseModel):
    entity_type: str
    embedding: list[float]                           # always 64-dim
    model_used: str                                  # "hgt" | "untrained"


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _build_features(req: EmbedRequest) -> tuple[str, list[float]]:
    if req.entity_type == "user":
        if not req.user:
            raise HTTPException(400, "user data required for entity_type=user")
        return "user", build_user_features(
            birthdate=req.user.birthdate,
            tag_weights=req.user.tag_weights,
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


# ─── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """
    Generate a 64-dim L2-normalised embedding from raw entity data.
    No DB access — pure computation.

    Next.js flow:
      1. Collect entity data
      2. POST /embed
      3. Save entity + embedding to DB in a single transaction
    """
    entity_type, features = _build_features(req)

    if _model is not None:
        embedding  = _model.encode(entity_type, features).cpu().tolist()
        model_used = "hgt"
    else:
        embedding  = (features + [0.0] * EMBED_DIM)[:EMBED_DIM]
        model_used = "untrained"

    return EmbedResponse(entity_type=entity_type, embedding=embedding, model_used=model_used)
