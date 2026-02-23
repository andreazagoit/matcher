"""
Matcher ML — HGT embedding service (PyG HGTConv).

Package layout
──────────────
  ml.config    — hyperparameters, vocabularies, graph metadata (METADATA, HGT_HEADS …)
  ml.utils     — numeric / date helpers
  ml.features  — feature-vector builders (user / event / space)
  ml.model     — HetEncoder: PyG HGTConv model + save/load helpers
  ml.graph     — HeteroData builder from training-data/*.json
  ml.train     — training loop, Recall@K evaluation, early stopping  → ml:train
  ml.sync      — batch DB → graph → forward_graph → upsert             → ml:sync
  ml.serve     — FastAPI /embed for new entities (forward_single)      → ml:serve
  ml.synthetic — synthetic training data generator                     → ml:generate
"""
