# Matcher ML Service

Recommendation service for users, events, and spaces.
Generates embeddings in a shared 64-dim space so any entity can be compared with any other.

## Setup

```bash
cp .env.example .env
# edit .env with your DATABASE_URL

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
DATABASE_URL=postgresql://... uvicorn server:app --reload --port 8000
```

## Endpoints

### GET /health
Returns model status.
```json
{ "status": "ok", "model_loaded": true, "is_training": false }
```

### POST /train
Triggers training from DB data (runs in background).
```bash
curl -X POST http://localhost:8000/train
```

### POST /recommend
Get recommendations for any entity.

```bash
# Events recommended for a user
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "user-uuid", "entity_type": "user", "target_type": "event", "limit": 10}'

# Users similar to another user (people matching)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "user-uuid", "entity_type": "user", "target_type": "user", "limit": 8}'

# Spaces recommended for a user
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "user-uuid", "entity_type": "user", "target_type": "space", "limit": 10}'

# Users likely to join a space (for invitations)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "space-uuid", "entity_type": "space", "target_type": "user", "limit": 20}'

# All types at once (omit target_type)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "user-uuid", "entity_type": "user", "limit": 15}'
```

### GET /similar/{entity_type}/{entity_id}
Shorthand for same-type similarity.
```bash
# Similar events to an event
curl http://localhost:8000/similar/event/event-uuid

# Similar spaces to a space
curl http://localhost:8000/similar/space/space-uuid
```

## Response format

```json
{
  "source_id": "user-uuid",
  "source_type": "user",
  "model_used": "ml",
  "recommendations": [
    { "id": "event-uuid", "type": "event", "score": 0.92 },
    { "id": "space-uuid", "type": "space", "score": 0.85 },
    { "id": "user-uuid",  "type": "user",  "score": 0.78 }
  ]
}
```

`model_used` is `"ml"` when the trained model is available, or `"jaccard_fallback"` for cold start.

## How it works

### Feature vector (45-dim per entity)
- `[0:40]`  Tag weights: user interests (0.0–1.0), event/space tags (1.0 if present)
- `[40]`    Age normalized 0–1 (user age / avg attendee age / avg member age)
- `[41:44]` Entity type one-hot: [user, event, space]
- `[44]`    Popularity normalized 0–1

### Training data
Positive interactions from DB:
- `event_attendees` (going/attended) → user ↔ event
- `members` (active) → user ↔ space
- `conversations` (active) → user ↔ user

Negatives: random sampling (5 per positive).

### Cold start
When model is not trained (no `model_weights.pt`), falls back to Jaccard tag similarity.
Train the model as soon as you have 100+ interactions for meaningful results.

## Deploy (Railway)

1. Create a Railway project
2. Set `DATABASE_URL` environment variable
3. Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Trigger training via cron from Next.js: `POST https://your-service.railway.app/train`
