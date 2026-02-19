"""
FastAPI ML Service — embedding generation and model training.

Endpoints:
  POST /embed    — generate a 64-dim embedding from raw entity data (stateless, no DB)
  POST /train    — train the model from DB interactions
  GET  /health   — health check + model status
"""

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config import NEGATIVE_SAMPLES
from data import build_training_triplets
from features import build_user_features, build_event_features, build_space_features
from model import EntityEncoder, train, save_model, load_model


# ─── State ─────────────────────────────────────────────────────────────────────

_model: Optional[EntityEncoder] = None
_is_training: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    if _model:
        print("Loaded existing model weights.")
    else:
        print("No model weights found. Use POST /train to train the model.")
    yield


app = FastAPI(
    title="Matcher ML Service",
    description="Generates 64-dim embeddings for users, events and spaces.",
    lifespan=lifespan,
)


# ─── Schemas ───────────────────────────────────────────────────────────────────

class UserData(BaseModel):
    tag_weights: "dict[str, float]" = {}
    birthdate: Optional[str] = None           # "YYYY-MM-DD"
    interaction_count: int = 0


class EventData(BaseModel):
    tags: "list[str]" = []
    avg_attendee_age: Optional[float] = None
    attendee_count: int = 0


class SpaceData(BaseModel):
    tags: "list[str]" = []
    avg_member_age: Optional[float] = None
    member_count: int = 0


class EmbedRequest(BaseModel):
    entity_type: str                          # "user" | "event" | "space"
    user: Optional[UserData] = None
    event: Optional[EventData] = None
    space: Optional[SpaceData] = None


class EmbedResponse(BaseModel):
    entity_type: str
    embedding: "list[float]"                  # always 64-dim
    model_used: str                           # "ml" or "untrained"


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "is_training": _is_training,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """
    Generate a 64-dim embedding from raw entity data.
    No DB access — pure computation.

    Next.js flow:
      1. Collect entity data (before saving to DB)
      2. Call POST /embed
      3. Save entity + embedding to DB in a single transaction
    """
    if req.entity_type == "user":
        if not req.user:
            raise HTTPException(status_code=400, detail="user data required for entity_type=user")
        features = build_user_features(
            birthdate=req.user.birthdate,
            tag_weights=req.user.tag_weights,
            interaction_count=req.user.interaction_count,
        )
    elif req.entity_type == "event":
        if not req.event:
            raise HTTPException(status_code=400, detail="event data required for entity_type=event")
        features = build_event_features(
            tags=req.event.tags,
            avg_attendee_age=req.event.avg_attendee_age,
            attendee_count=req.event.attendee_count,
        )
    elif req.entity_type == "space":
        if not req.space:
            raise HTTPException(status_code=400, detail="space data required for entity_type=space")
        features = build_space_features(
            tags=req.space.tags,
            avg_member_age=req.space.avg_member_age,
            member_count=req.space.member_count,
        )
    else:
        raise HTTPException(status_code=400, detail="entity_type must be user, event, or space")

    if _model is not None:
        embedding = _model.encode(features).cpu().tolist()
        model_used = "ml"
    else:
        # Untrained model: return the raw feature vector padded/truncated to 64-dim
        # This is a deterministic fallback so Next.js can still store something
        from config import EMBED_DIM
        embedding = (features + [0.0] * EMBED_DIM)[:EMBED_DIM]
        model_used = "untrained"

    return EmbedResponse(
        entity_type=req.entity_type,
        embedding=embedding,
        model_used=model_used,
    )


@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """Train the model from DB interactions. Runs in background."""
    global _is_training
    if _is_training:
        raise HTTPException(status_code=409, detail="Training already in progress.")
    background_tasks.add_task(_run_training)
    return {"status": "training_started"}


async def _run_training():
    global _model, _is_training
    _is_training = True
    try:
        print("Building training triplets from DB...")
        loop = asyncio.get_event_loop()
        triplets = await loop.run_in_executor(
            None,
            lambda: build_training_triplets(negative_samples=NEGATIVE_SAMPLES),
        )
        print(f"Training on {len(triplets)} triplets...")
        model = await loop.run_in_executor(None, lambda: train(triplets))
        save_model(model)
        _model = model
        print("Training complete.")
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        _is_training = False
