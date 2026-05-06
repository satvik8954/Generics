CONFIG = {

    # ========================
    # MODEL SIZE (balanced)
    # ========================

    "api_in": 20,         # 20 molecular descriptors
    "api_hidden": 256,
    "api_out": 128,

    "per_unit_vocab": 4,  # {1, mL, g, L}
    "per_unit_emb": 16,
    "strength_out": 64,

    "route_vocab": 16,    # 14 routes in data, pad to 16
    "route_emb": 32,

    "form_vocab": 110,    # 103 dosage forms in data, pad to 110
    "form_emb": 32,

    "context_out": 192,

    "excipient_emb": 128,

    "scorer_hidden": 128,

    # ========================
    # GNN (HetGNN)
    # ========================

    "gnn_hidden": 256,
    "gnn_heads": 2,
    "gnn_layers": 2,
    "gnn_dropout": 0.2,
    "jaccard_threshold": 0.1,
    "similarity_threshold": 0.4,

    # ========================
    # REGULARIZATION
    # ========================

    "dropout_api": 0.1,
    "dropout_context": 0.2,
    "dropout_scorer": 0.2,

    # ========================
    # TRAINING
    # ========================

    "lr": 1e-4,
    "batch_size": 64,
    "epochs": 30,

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