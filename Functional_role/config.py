"""
config.py — shared paths, role taxonomy, and constants for the whole
Task B (formulation-level excipient role assignment) pipeline.
"""

# ─────────────────────────────────────────────
# PATHS (adjust these if you move the data)
# ─────────────────────────────────────────────
\

FUNCTIONAL_CSV = "data/functional_categories_excipients_FILLED_1.csv"
UNII_CSV       = "data/excipients_by_unii.csv"
ORAL_CSV       = "data/oral_only.csv"

OUTPUT_DIR          = "outputs"
PICKLE_PATH         = "weak_role_labels.pkl"
HIGH_CONF_CSV_PATH  = f"{OUTPUT_DIR}/weak_labels_high_confidence.csv"
PASS2_CSV_PATH      = f"{OUTPUT_DIR}/weak_labels_pass2.csv"
MODEL_PATH          = f"{OUTPUT_DIR}/task_b_model.pkl"

# Precomputed RDKit descriptors per API, keyed by api_unii
API_FEATURES_CSV = "data/api_features.csv"
API_FEATURE_COLS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings", "RingCount",
    "FractionCSP3", "HeavyAtomCount", "NumValenceElectrons", "MolMR",
    "LabuteASA", "BalabanJ", "BertzCT", "HallKierAlpha", "NumSaturatedRings",
    "NumHeteroatoms", "NHOHCount",
]
NUM_API_FEATURES = len(API_FEATURE_COLS)

# ─────────────────────────────────────────────
# ROLE TAXONOMY — raw HPE "Category of Use" columns collapsed into
# canonical oral-relevant roles. Keep this identical everywhere the
# taxonomy is used (weak labeling, Task A, Task B) or labels won't line up.
# ─────────────────────────────────────────────
ROLE_TAXONOMY = {
    "binder":                      ["tablet binder", "binding agent"],
    "filler":                      ["tablet and capsule diluent", "diluent", "tablet diluent",
                                     "tablet filler", "tablet and capsule filler", "bulking agent",
                                     "directly compressible tablet excipient"],
    "disintegrant":                ["tablet and capsule disintegrant", "tablet disintegrant",
                                     "disintegrant"],
    "lubricant":                   ["tablet and capsule lubricant", "lubricant", "tablet lubricant",
                                     "antiadherent"],
    "glidant":                     ["glidant", "anticaking agent", "adsorbent"],
    "coating_agent":                ["coating agent", "film-forming agent", "enteric coating agent",
                                     "membrane-forming agent", "polishing agent", "encapsulating agent"],
    "controlled_release":          ["controlled-release agent", "sustained-release agent",
                                     "extended-release agent", "modified-release agent",
                                     "release-modifying agent", "matrix-forming agent", "osmotic agent"],
    "solvent":                     ["solvent", "cosolvent", "water-miscible cosolvent"],
    "surfactant":                  ["surfactant", "nonionic surfactant", "anionic surfactant",
                                     "cationic surfactant", "wetting agent", "emulsifying agent",
                                     "fecal softener"],
    "plasticizer":                 ["plasticizer"],
    "alkalizing_agent":            ["alkalizing agent", "antacid"],
    "acidifying_agent":            ["acidifying agent", "acidulant"],
    "humectant":                   ["humectant"],
    "sweetening_agent":            ["sweetening agent"],
    "flavoring_agent":             ["flavoring agent", "flavor enhancer", "cooling agent",
                                     "taste-masking agent"],
    "preservative":                ["antimicrobial preservative", "preservative"],
    "antioxidant":                 ["antioxidant"],
    "buffering_agent":             ["buffering agent"],
    "suspending_thickening_agent": ["suspending agent", "viscosity-increasing agent",
                                     "thickening agent", "gelling agent", "rheology modifier",
                                     "gel base", "viscosity-controlling agent"],
    "stabilizing_agent":           ["stabilizing agent", "emulsion stabilizer", "foam stabilizer"],
    "solubilizing_agent":          ["solubilizing agent", "dissolution enhancer",
                                     "solubility enhancing agent", "complexing agent",
                                     "chelating agent", "sequestering agent"],
    "granulation_aid":             ["granulation aid", "compression aid"],
    "dispersing_agent":            ["dispersing agent", "foaming agent", "antifoaming agent"],
    "bioadhesive_agent":           ["bioadhesive material", "mucoadhesive"],
}
ROLE_NAMES = list(ROLE_TAXONOMY.keys())
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLE_NAMES)}
NUM_ROLES = len(ROLE_NAMES)

# roles that should have AT MOST ONE dominant excipient per formulation
EXCLUSIVE_ROLES = {
    "binder", "filler", "disintegrant", "lubricant", "glidant",
    "coating_agent", "controlled_release", "solvent",
}

# minimum Pass 1 occurrences required before trusting an excipient-specific
# prior over the global fallback prior (see weak_labels.py estimate_priors)
MIN_UNII_COUNT = 5

# roles that are commonly tagged as a SECONDARY/loose HPE category for many
# excipients (e.g. lots of binders also get "granulation aid" listed) and
# tend to become a false-default fallback once an excipient's real role is
# taken by a stronger competitor. Penalize these unless there's specific
# evidence (n_samples >= MIN_UNII_COUNT) for THIS excipient in THIS role.
GENERIC_ROLES = {"granulation_aid"}
GENERIC_ROLE_PENALTY = 0.15