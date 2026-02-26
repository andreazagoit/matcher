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
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import psycopg2
import json

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)
oai_client = AsyncOpenAI()

from hgt.config import EMBED_DIM, TAG_EMBED_DIM, DATABASE_URL
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


# ─── Request / Response schemas ────────────────────────────────────────────────

# ─── Tag Embedding Cache (Postgres) ───────────────────────────────────────────
import time

TAG_CACHE: dict[str, dict] = {} # {tag_id: {"emb": [...], "ts": float}}
CACHE_TTL = 300 # 5 minutes

def get_tag_embeddings_for_list(tags: list[str]) -> list[list[float]]:
    """Fetch 64d embeddings for a list of tags from Postgres (with caching & TTL)."""
    if not tags:
        return []
        
    embeddings = []
    missing_tags = []
    now = time.time()
    
    for t in tags:
        sanitized = t.lower().replace(" ", "_")
        cache_entry = TAG_CACHE.get(sanitized)
        if cache_entry and (now - cache_entry["ts"] < CACHE_TTL):
            embeddings.append(cache_entry["emb"])
        else:
            missing_tags.append(sanitized)
            
    if missing_tags:
        try:
            with psycopg2.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    query = "SELECT id, embedding FROM tags WHERE id = ANY(%s)"
                    cur.execute(query, (missing_tags,))
                    rows = cur.fetchall()
                    for row in rows:
                        tag_id = row[0]
                        emb_data = row[1]
                        if emb_data is not None:
                            # pgvector returns a string like '[0.1, 0.2]'
                            emb = json.loads(emb_data) if isinstance(emb_data, str) else emb_data
                            TAG_CACHE[tag_id] = {"emb": emb, "ts": now}
                            
            # Add fetched ones (or fallback for completely missing ones)
            for t in missing_tags:
                if t in TAG_CACHE:
                    embeddings.append(TAG_CACHE[t]["emb"])
        except Exception as e:
            print(f"Error fetching tag embeddings from DB: {e}")
            
    return embeddings

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


class TagData(BaseModel):
    name: str  # Generates 64-dim OAI embedding and 256-dim HGT embedding on the fly


class TagResponse(BaseModel):
    embedding: list[float]                    # 256-dim embeddings for graph DB
    tag_embedding: list[float]                # 64-dim OAI embeddings
    model_used: str                           # "hgt" | "untrained"

class EntityResponse(BaseModel):
    embedding: list[float]                    # EMBED_DIM floats
    model_used: str                           # "hgt" | "untrained"


# ── Feature extraction ────────────────────────────────────────────────────────

# ── Endpoint Routes ─────────────────────────────────────────────────────────────

@app.post("/embed/user", response_model=EntityResponse)
def embed_user(req: UserData):
    tag_embs = get_tag_embeddings_for_list(req.tags)
    features = build_user_features(
        birthdate=req.birthdate,
        tag_embeddings=tag_embs,
        num_tags=len(req.tags),
        gender=req.gender,
        relationship_intent=req.relationshipIntent,
        smoking=req.smoking,
        drinking=req.drinking,
        activity_level=req.activityLevel,
    )
    return _infer_entity("user", features)

@app.post("/embed/event", response_model=EntityResponse)
def embed_event(req: EventData):
    tag_embs = get_tag_embeddings_for_list(req.tags)
    features = build_event_features(
        tag_embeddings=tag_embs,
        num_tags=len(req.tags),
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
    tag_embs = get_tag_embeddings_for_list(req.tags)
    features = build_space_features(
        tag_embeddings=tag_embs,
        num_tags=len(req.tags),
        avg_member_age=req.avgMemberAge,
        member_count=req.memberCount,
        event_count=req.eventCount,
    )
    return _infer_entity("space", features)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/embed/tag", response_model=TagResponse)
async def embed_tag(req: TagData):
    # 1. Fetch 64-dim semantic text embedding from OpenAI
    response = await oai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.name,
        dimensions=TAG_EMBED_DIM,
    )
    tag_embedding = response.data[0].embedding
        
    # 2. Project into 256-dim HGT latent space
    if _model is not None:
         x = torch.tensor(tag_embedding, dtype=torch.float32).unsqueeze(0).to(device)
         with torch.no_grad():
             embedding = _model.forward_single(x, "tag").squeeze(0).cpu().tolist()
         model_used = "hgt"
    else:
         embedding  = (tag_embedding + [0.0] * EMBED_DIM)[:EMBED_DIM]
         model_used = "untrained"
             
    return TagResponse(
         embedding=embedding, 
         tag_embedding=tag_embedding, 
         model_used=model_used
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
