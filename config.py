CONFIG = {

    # ========================
    # MODEL SIZE (balanced)
    # ========================

    "api_in": 20,         # 20 molecular descriptors
    "api_hidden": 256,
    "api_out": 128,

    "per_unit_vocab": 16,  # Padded for safety
    "per_unit_emb": 16,
    "strength_out": 64,

    "route_vocab": 64,    # Padded for safety
    "route_emb": 32,

    "form_vocab": 128,    # Padded for safety
    "form_emb": 32,

    "context_out": 192,

    "excipient_emb": 128,

    "scorer_hidden": 128,

    # ========================
    # GNN (HetGNN)
    # ========================

    "gnn_hidden": 256,
    "gnn_layers": 4,
    "gnn_dropout": 0.2,
    "jaccard_threshold": 0.1,
    "similarity_threshold": 0.6,

    # ========================
    # REGULARIZATION
    # ========================

    "dropout_api": 0.1,
    "dropout_context": 0.2,
    "dropout_scorer": 0.2,

    # ========================
    # TRAINING
    # ========================

    "lr": 5e-4,
    "batch_size": 128,
    "epochs": 100,

    # ========================
    # INFERENCE
    # ========================

    "top_k": 10,
    "threshold": 0.5,

    # ========================
    # SYSTEM
    # ========================

    "device": "cuda",
    "seed": 42
}