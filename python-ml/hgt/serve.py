"""
FastAPI ML Service — embedding inference for new entities.

POST /embed/user     → 256-dim embedding from raw user profile data
POST /embed/event    → 256-dim embedding from raw event data
POST /embed/space    → 256-dim embedding from raw space data
POST /embed/category → 64-dim OAI embedding + 256-dim HGT embedding for a category

Startup:
  uvicorn hgt.serve:app --reload
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import psycopg2
import json

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)
oai_client = AsyncOpenAI()

from hgt.config import EMBED_DIM, CATEGORY_EMBED_DIM, DATABASE_URL
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
    description="Generates 256-dim embeddings for users, events, spaces and categories.",
    lifespan=lifespan,
)


# ─── Category Embedding Cache (Postgres) ──────────────────────────────────────
import time

CATEGORY_CACHE: dict[str, dict] = {}  # {category_id: {"emb": [...], "ts": float}}
CACHE_TTL = 300  # 5 minutes


def get_category_embeddings_for_list(categories: list[str]) -> list[list[float]]:
    """Fetch 64d embeddings for a list of categories from Postgres (with TTL cache)."""
    if not categories:
        return []

    embeddings = []
    missing = []
    now = time.time()

    for c in categories:
        sanitized = c.lower().replace(" ", "_")
        entry = CATEGORY_CACHE.get(sanitized)
        if entry and (now - entry["ts"] < CACHE_TTL):
            embeddings.append(entry["emb"])
        else:
            missing.append(sanitized)

    if missing:
        try:
            with psycopg2.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, embedding FROM categories WHERE id = ANY(%s)",
                        (missing,),
                    )
                    for row in cur.fetchall():
                        cat_id, emb_data = row
                        if emb_data is not None:
                            emb = json.loads(emb_data) if isinstance(emb_data, str) else emb_data
                            CATEGORY_CACHE[cat_id] = {"emb": emb, "ts": now}

            for c in missing:
                if c in CATEGORY_CACHE:
                    embeddings.append(CATEGORY_CACHE[c]["emb"])
        except Exception as e:
            print(f"Error fetching category embeddings from DB: {e}")

    return embeddings


# ─── Request / Response schemas ───────────────────────────────────────────────

class UserData(BaseModel):
    categories: list[str] = []          # category IDs from impressions
    birthdate: Optional[str] = None     # "YYYY-MM-DD"
    gender: Optional[str] = None        # "man" | "woman" | "non_binary"
    relationshipIntent: list[str] = []
    smoking: Optional[str] = None
    drinking: Optional[str] = None
    activityLevel: Optional[str] = None


class EventData(BaseModel):
    categories: list[str] = []
    startsAt: Optional[str] = None
    avgAttendeeAge: Optional[float] = None
    attendeeCount: int = 0
    daysUntilEvent: Optional[int] = None
    maxAttendees: Optional[int] = None
    isPaid: bool = False
    priceCents: Optional[int] = None


class SpaceData(BaseModel):
    categories: list[str] = []
    avgMemberAge: Optional[float] = None
    memberCount: int = 0
    eventCount: int = 0


class CategoryData(BaseModel):
    name: str  # Generates 64-dim OAI embedding and 256-dim HGT embedding


class CategoryResponse(BaseModel):
    embedding: list[float]           # 256-dim HGT embedding for graph DB
    category_embedding: list[float]  # 64-dim OAI embedding for categories table
    model_used: str                  # "hgt" | "untrained"


class EntityResponse(BaseModel):
    embedding: list[float]  # EMBED_DIM floats
    model_used: str         # "hgt" | "untrained"


# ── Endpoint Routes ────────────────────────────────────────────────────────────

@app.post("/embed/user", response_model=EntityResponse)
def embed_user(req: UserData):
    cat_embs = get_category_embeddings_for_list(req.categories)
    features = build_user_features(
        birthdate=req.birthdate,
        category_embeddings=cat_embs,
        gender=req.gender,
        relationship_intent=req.relationshipIntent,
        smoking=req.smoking,
        drinking=req.drinking,
        activity_level=req.activityLevel,
    )
    return _infer_entity("user", features)


@app.post("/embed/event", response_model=EntityResponse)
def embed_event(req: EventData):
    cat_embs = get_category_embeddings_for_list(req.categories)
    features = build_event_features(
        category_embeddings=cat_embs,
        starts_at=req.startsAt,
        avg_attendee_age=req.avgAttendeeAge,
        attendee_count=req.attendeeCount,
        days_until_event=req.daysUntilEvent,
        max_attendees=req.maxAttendees,
        is_paid=req.isPaid,
        price_cents=req.priceCents,
    )
    return _infer_entity("event", features)


@app.post("/embed/space", response_model=EntityResponse)
def embed_space(req: SpaceData):
    cat_embs = get_category_embeddings_for_list(req.categories)
    features = build_space_features(
        category_embeddings=cat_embs,
        avg_member_age=req.avgMemberAge,
        member_count=req.memberCount,
        event_count=req.eventCount,
    )
    return _infer_entity("space", features)


@app.post("/embed/category", response_model=CategoryResponse)
async def embed_category(req: CategoryData):
    # 1. Fetch 64-dim semantic text embedding from OpenAI
    response = await oai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.name,
        dimensions=CATEGORY_EMBED_DIM,
    )
    category_embedding = response.data[0].embedding

    # 2. Project into 256-dim HGT latent space
    if _model is not None:
        x = torch.tensor(category_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = _model.forward_single(x, "category").squeeze(0).cpu().tolist()
        model_used = "hgt"
    else:
        embedding  = (category_embedding + [0.0] * EMBED_DIM)[:EMBED_DIM]
        model_used = "untrained"

    return CategoryResponse(
        embedding=embedding,
        category_embedding=category_embedding,
        model_used=model_used,
    )


def _infer_entity(entity_type: str, features: list[float]) -> EntityResponse:
    if _model is not None:
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = _model.forward_single(x, entity_type).squeeze(0).cpu().tolist()
        model_used = "hgt"
    else:
        embedding  = (features + [0.0] * EMBED_DIM)[:EMBED_DIM]
        model_used = "untrained"

    return EntityResponse(embedding=embedding, model_used=model_used)
