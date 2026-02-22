#!/usr/bin/env python3
"""
Batch re-embed all entities and write training-data/embeddings.json.

Run after training to regenerate embeddings:
  npm run ml:reembed

Then push embeddings to the DB:
  npm run ml:update-embeddings

Output format (embeddings.json):
  [
    {"id": "<uuid>", "type": "user"|"event"|"space", "embedding": [64 floats]},
    ...
  ]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

from config import TRAINING_DATA_DIR, MODEL_WEIGHTS_PATH
from data import _read, users_to_features, events_to_features, spaces_to_features
from model import load_model, encode_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch re-embed all entities using the trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=TRAINING_DATA_DIR,
        help="Directory with JSON export files",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_WEIGHTS_PATH,
        help="Path to model_weights.pt",
    )
    parser.add_argument(
        "--out-file",
        default=os.path.join(TRAINING_DATA_DIR, "embeddings.json"),
        help="Output file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Encoding batch size",
    )
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    model = load_model(args.model_path)
    if model is None:
        print(
            f"Error: no trained model at {args.model_path}.\n"
            "Run 'npm run ml:train' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load entity data ───────────────────────────────────────────────────────
    print(f"Loading entity data from {args.data_dir}...")
    try:
        users_raw  = _read(args.data_dir, "users.json")
        events_raw = _read(args.data_dir, "events.json")
        spaces_raw = _read(args.data_dir, "spaces.json")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    user_features  = users_to_features(users_raw)
    event_features = events_to_features(events_raw)
    space_features = spaces_to_features(spaces_raw)

    all_features: dict[str, tuple[str, list[float]]] = {}
    for uid, fvec in user_features.items():
        all_features[uid] = ("user", fvec)
    for eid, fvec in event_features.items():
        all_features[eid] = ("event", fvec)
    for sid, fvec in space_features.items():
        all_features[sid] = ("space", fvec)

    print(
        f"  {len(all_features):,} entities  "
        f"({len(user_features):,} users / "
        f"{len(event_features):,} events / "
        f"{len(space_features):,} spaces)"
    )

    # ── Encode ─────────────────────────────────────────────────────────────────
    print(f"Encoding (batch_size={args.batch_size})...", flush=True)
    t0 = time.time()
    embeddings = encode_all(model, all_features, batch_size=args.batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(embeddings):,} embeddings)")

    # ── Write output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    out = [
        {
            "id":        eid,
            "type":      all_features[eid][0],
            "embedding": emb.cpu().tolist(),
        }
        for eid, emb in embeddings.items()
    ]
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(out, f)

    size_mb = os.path.getsize(args.out_file) / 1_048_576
    print(f"  ✓ {args.out_file}  ({size_mb:.1f} MB, {len(out):,} records)")
    print("\nRun 'npm run ml:update-embeddings' to push to the database.")


if __name__ == "__main__":
    main()
