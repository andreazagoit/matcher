# Matcher ML Service

HGT-lite embedding service for the Matcher platform.  
Generates L2-normalised embeddings in a shared 256-dim space so any entity (user / event / space) can be compared against any other.

---

## Package layout

```
python-ml/
├── ml/                         ← installable Python package
│   ├── config.py               — hyperparameters, vocabularies, feature dimensions
│   ├── utils.py                — numeric / date helpers
│   ├── features.py             — feature-vector builders (user / event / space)
│   ├── data.py                 — training-data loader (reads JSON exports)
│   ├── modeling/
│   │   └── hgt.py              — HGT-lite architecture, loss, TripletDataset, batch encode
│   ├── training/
│   │   └── trainer.py          — training loop, hard-negative mining, save/load
│   ├── evaluation/
│   │   └── metrics.py          — Recall@K / NDCG@K
│   └── serving/
│       └── api.py              — FastAPI application (POST /embed)
├── train.py                    ← entry point: train the model
├── server.py                   ← entry point: start the FastAPI server
├── embed_all.py                ← entry point: batch-embed all DB entities
├── generate_synthetic.py       ← entry point: generate synthetic training data
├── training-data/              ← JSON exports (git-ignored)
├── hgt_weights.pt            ← trained checkpoint (git-ignored)
└── pyproject.toml
```

---

## Setup

```bash
# From the repo root
npm run ml:setup        # creates .venv and pip install -e python-ml/

# Or manually
cd python-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy the environment file and set your database URL:

```bash
cp .env.example .env
# edit .env → set DATABASE_URL
```

---

## Workflow

### 1. Get training data

**Real data** (requires a running database):
```bash
npm run ml:export
```

**Synthetic data** (no DB needed — great for development):
```bash
npm run ml:generate
```

Both commands write JSON files to `training-data/`.

### 2. Train

```bash
npm run ml:train

# Options
python train.py --epochs 100 --patience 20
python train.py --hard-neg-every 10 --n-hard-neg 3
python train.py --val-ratio 0.15 --eval-every 5
```

Saves the best checkpoint to `hgt_weights.pt`.

### 3. Embed all entities

```bash
npm run ml:embed-all
```

Reads all users / events / spaces from PostgreSQL, generates embeddings with the trained model, and upserts them into the `embeddings` table.

### 4. Start the inference server

```bash
npm run ml:serve
# → http://localhost:8000
```

---

## API

### `POST /embed`

Generate an embedding from raw entity data (no DB access).

**Request:**
```json
{
  "entity_type": "user",
  "user": {
    "tag_weights": { "hiking": 0.9, "coffee": 0.6 },
    "birthdate": "1995-04-12",
    "gender": "woman",
    "relationship_intent": ["serious_relationship"],
    "smoking": "never",
    "drinking": "sometimes",
    "activity_level": "active",
    "interaction_count": 14
  }
}
```

**Response:**
```json
{
  "entity_type": "user",
  "embedding": [0.021, -0.043, ...],   // 256-dim, L2-normalised
  "model_used": "hgt"                  // "hgt" | "untrained"
}
```

The same schema works for `entity_type: "event"` and `entity_type: "space"`.

**Next.js integration flow:**
1. Collect entity data server-side
2. `POST /embed` → receive embedding
3. Save entity + embedding to the DB in a single transaction

---

## Architecture

```
UserEncoder  (60-dim)  ─┐
EventEncoder (51-dim)  ─┤─→ HGTLayer1 → HGTLayer2 → OutputProj → L2-norm → 256-dim
SpaceEncoder (43-dim)  ─┘
```

**Feature dimensions:**
| Entity | Dim | Features Built From JSON |
|---|---|---|
| User   | 219  | tag weights + age + gender + rel_intent + smoking + drinking + activity |
| Event  | 51  | tags (40) + avg_age (1) + count (1) + days_until (1) + fill_rate (1) + is_paid (1) + price (1) + time_cyclical (5) |
| Space  | 43  | tags (40) + avg_age (1) + member_count (1) + event_count (1) |

**Training:**
- Loss: InfoNCE (in-batch negatives) + weighted margin loss
- Hard negative mining every N epochs
- DropGraph (50/50): alternates between graph-augmented and encoder-only forward passes so the inference path (no graph) receives equal gradient
- Validation: Recall@10 / NDCG@10 per task (user→event, user→space, user→user, …)
- Early stopping on micro-average Recall@10

**Cold start:** Falls back to weighted Jaccard tag similarity when `hgt_weights.pt` is missing.

---

## Deploy

Start command (Railway / Render / Fly):

```bash
uvicorn server:app --host 0.0.0.0 --port $PORT
```

Set `DATABASE_URL` as an environment variable.
