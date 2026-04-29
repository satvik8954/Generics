CONFIG = {

    # ========================
    # MODEL SIZE (balanced)
    # ========================

    "api_in": 20,
    "api_hidden": 256,
    "api_out": 128,

    "per_unit_vocab": 4,
    "per_unit_emb": 16,
    "strength_out": 64,

    "route_vocab": 16,
    "route_emb": 32,

    "form_vocab": 16,
    "form_emb": 32,

    "context_out": 192,

    "excipient_emb": 128,

    "scorer_hidden": 128,

    # ========================
    # REGULARIZATION (IMPORTANT)
    # ========================

    "dropout_api": 0.1,
    "dropout_context": 0.2,
    "dropout_scorer": 0.2,

    # ========================
    # TRAINING (TUNED FOR YOUR DATA)
    # ========================

    "lr": 1e-4,          # stable for your size
    "batch_size": 64,    # good for 30k dataset
    "epochs": 30,        # enough (don’t overtrain)

    "weight_decay": 1e-5,  # prevents overfitting
    "grad_clip": 1.0,

    # ========================
    # LOSS (CRITICAL FOR YOU)
    # ========================

    "use_pos_weight": True,
    "pos_weight_clip": 30.0,   # slightly lower than default

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