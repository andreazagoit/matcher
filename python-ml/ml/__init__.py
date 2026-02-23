"""
Matcher ML — HGT-lite embedding service.

Package layout
──────────────
  ml.config       — hyperparameters, vocabularies, feature dimensions
  ml.utils        — numeric / date helpers
  ml.features     — feature-vector builders (user / event / space)
  ml.data         — training-data loader
  ml.modeling     — HetEncoder architecture, loss, dataset utilities
  ml.training     — training loop, hard-negative mining, checkpoint I/O
  ml.evaluation   — Recall@K / NDCG@K metrics
  ml.serving      — FastAPI application
  ml.inference    — batch embedding pipeline (DB → embeddings → DB)
  ml.synthetic    — synthetic training data generator
"""
