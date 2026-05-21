"""
incompatibility_engine.py — Chemical incompatibility reranker for ExciPick.

Sits on top of GNN recommendations and penalizes excipients that would
chemically conflict with the API or with each other.

Three layers:
  1. API vs Excipient clash
  2. Excipient vs Excipient pairwise clash
  3. Flag inference for excipients not in the handbook DB

Usage (standalone):
    from incompatibility_engine import IncompatibilityEngine
    engine = IncompatibilityEngine()
    reranked = engine.rerank(api_smiles, gnn_predictions)

Usage (in predict.py):
    from incompatibility_engine import IncompatibilityEngine
    engine = IncompatibilityEngine()
    predictions = engine.rerank(api_smiles, predictions)
"""

import json
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# ─────────────────────────────────────────────
# CLASH MATRIX
# Each entry: (flag_on_A, flag_on_B) → (severity, label)
# Severity: 1.0 = definite reaction, 0.5 = likely, 0.3 = minor risk
# ─────────────────────────────────────────────
CLASH_MATRIX = {
    ("acid_risk",             "alkaline_risk"):        (1.0, "acid_base_reaction"),
    ("alkaline_risk",         "acid_risk"):            (1.0, "acid_base_reaction"),
    ("oxidizing_agent_risk",  "reducing_agent_risk"):  (1.0, "redox_reaction"),
    ("reducing_agent_risk",   "oxidizing_agent_risk"): (1.0, "redox_reaction"),
    ("reducing_sugar_risk",   "amine_reactive"):       (0.8, "maillard_reaction"),
    ("amine_reactive",        "reducing_sugar_risk"):  (0.8, "maillard_reaction"),
    ("amine_reactive",        "carbonyl_reactive"):    (0.8, "schiff_base_formation"),
    ("carbonyl_reactive",     "amine_reactive"):       (0.8, "schiff_base_formation"),
    ("metal_ion_risk",        "thiol_reactive"):       (0.7, "metal_thiol_chelation"),
    ("thiol_reactive",        "metal_ion_risk"):       (0.7, "metal_thiol_chelation"),
    ("oxidizing_agent_risk",  "thiol_reactive"):       (0.7, "thiol_oxidation"),
    ("thiol_reactive",        "oxidizing_agent_risk"): (0.7, "thiol_oxidation"),
    ("salicylate_risk",       "alkaline_risk"):        (0.6, "salicylate_hydrolysis"),
    ("alkaloid_risk",         "acid_risk"):            (0.6, "alkaloid_salt_formation"),
    ("moisture_risk",         "moisture_risk"):        (0.3, "hygroscopic_competition"),
}

# ─────────────────────────────────────────────
# SMARTS PATTERNS for API flag detection
# ─────────────────────────────────────────────
API_SMARTS = {
    "acid_risk":            "[CX3](=O)[OX2H1]",           # carboxylic acid
    "alkaline_risk":        "[NX3;H2,H1;!$(NC=O)]",       # primary/secondary amine
    "amine_reactive":       "[NX3;H2,H1;!$(NC=O)]",       # same — amine groups
    "thiol_reactive":       "[SX2H]",                      # thiol
    "carbonyl_reactive":    "[CX3H1](=O)",                 # aldehyde
    "reducing_agent_risk":  "[OX2H][CX4][OX2H]",          # diol (reducing)
    "oxidizing_agent_risk": "[OX1]~[OX1]",                 # peroxide
    "salicylate_risk":      "c1ccccc1C(=O)O",              # salicylate core
    "alkaloid_risk":        "[nX3]",                       # aromatic nitrogen (alkaloid-like)
    "metal_ion_risk":       "[OX2H][cX3]:[cX3][OX2H]",    # catechol (chelating)
}

# ─────────────────────────────────────────────
# FUNCTIONAL GROUP → FLAG MAPPING
# Maps functional_group keys from excipientsFeaturesDB to incompatibility flags
# ─────────────────────────────────────────────
FG_TO_FLAG = {
    "Carboxylic_Acid": "acid_risk",
    "Phenol":          "acid_risk",
    "Primary_Amine":   "amine_reactive",
    "Secondary_Amine": "amine_reactive",
    "Tertiary_Amine":  "alkaline_risk",
    "Thiol":           "thiol_reactive",
    "Aldehyde":        "carbonyl_reactive",
    "Ketone":          "carbonyl_reactive",
    "Nitro":           "oxidizing_agent_risk",
    "Azo":             "oxidizing_agent_risk",
    "Urea":            "amine_reactive",
    "Sulfonamide":     "acid_risk",
}

HANDBOOK_FLAGS = [
    "acid_risk", "alkaline_risk", "oxidizing_agent_risk", "reducing_agent_risk",
    "reducing_sugar_risk", "metal_ion_risk", "moisture_risk", "salicylate_risk",
    "alkaloid_risk", "antibiotic_risk", "amine_reactive", "thiol_reactive",
    "carbonyl_reactive", "plasticizer_risk", "surfactant_risk",
]


class IncompatibilityEngine:
    """
    Reranks GNN excipient recommendations by penalizing chemical incompatibilities.

    Args:
        handbook_flags_path: path to incompatibilities_flags.json
        features_db_path:    path to excipientsFeaturesDB.csv
        clash_lambda:        weight of clash penalty vs GNN score (default 0.5)
        pairwise_lambda:     weight of pairwise clash penalty (default 0.3)
    """

    def __init__(
        self,
        handbook_flags_path: str = "incompatibilities_flags.json",
        features_db_path: str = "excipientsFeaturesDB.csv",
        clash_lambda: float = 0.15,
        pairwise_lambda: float = 0.08,
    ):
        self.clash_lambda = clash_lambda
        self.pairwise_lambda = pairwise_lambda

        self.handbook_db = self._load_handbook_flags(handbook_flags_path)
        self.features_db = self._load_features_db(features_db_path)
        self._flag_cache = {}  # cache computed flags per excipient name

        print(f"  [IncompatibilityEngine] Handbook flags: {len(self.handbook_db)} excipients")
        print(f"  [IncompatibilityEngine] Features DB:    {len(self.features_db)} excipients")

    # ─────────────────────────────────────────────
    # LOADING
    # ─────────────────────────────────────────────

    def _load_handbook_flags(self, path: str) -> dict:
        """Load handbook flags. Returns {excipient_name_upper: {flag: bool}}"""
        if not os.path.exists(path):
            print(f"  [IncompatibilityEngine] WARNING: {path} not found, handbook flags disabled")
            return {}
        with open(path) as f:
            entries = json.load(f)
        db = {}
        for entry in entries:
            name = entry.get("excipient_name", "").upper().strip()
            if name:
                db[name] = {flag: bool(entry.get(flag, False)) for flag in HANDBOOK_FLAGS}
        return db

    def _load_features_db(self, path: str) -> dict:
        """Load features DB. Returns {drug_name_upper: row_dict}"""
        if not os.path.exists(path):
            print(f"  [IncompatibilityEngine] WARNING: {path} not found, feature inference disabled")
            return {}
        df = pd.read_csv(path)
        db = {}
        for _, row in df.iterrows():
            name = str(row.get("drug_name", "")).upper().strip()
            if name:
                db[name] = row.to_dict()
        return db

    # ─────────────────────────────────────────────
    # FLAG RESOLUTION
    # ─────────────────────────────────────────────

    def get_excipient_flags(self, excipient_name: str) -> dict:
        """
        Get incompatibility flags for an excipient.
        Priority: handbook (ground truth) > inferred from features > all False.
        """
        key = excipient_name.upper().strip()

        if key in self._flag_cache:
            return self._flag_cache[key]

        # Priority 1 — handbook flags (ground truth)
        if key in self.handbook_db:
            flags = self.handbook_db[key]
            self._flag_cache[key] = flags
            return flags

        # Priority 2 — infer from molecular features
        if key in self.features_db:
            flags = self._infer_flags_from_features(self.features_db[key])
            self._flag_cache[key] = flags
            return flags

        # Priority 3 — unknown, treat as safe
        flags = {flag: False for flag in HANDBOOK_FLAGS}
        self._flag_cache[key] = flags
        return flags

    def _infer_flags_from_features(self, row: dict) -> dict:
        """Infer incompatibility flags from molecular descriptors and functional groups."""
        flags = {flag: False for flag in HANDBOOK_FLAGS}

        try:
            hbd = float(row.get("hbonddonors", 0) or 0)
            hba = float(row.get("hbondacceptors", 0) or 0)
            logp = float(row.get("logp", 0) or 0)

            # moisture_risk — highly hydrophilic = hygroscopic
            flags["moisture_risk"] = (hbd + hba) > 8

            # Parse functional groups
            fg_raw = row.get("functional_group", "{}")
            try:
                fg = json.loads(fg_raw) if isinstance(fg_raw, str) else {}
            except Exception:
                fg = {}

            # Map functional groups to flags
            for fg_key, flag_name in FG_TO_FLAG.items():
                if fg.get(fg_key, 0):
                    flags[flag_name] = True

            # acid_risk — low logP + carboxylic acid or phenol
            if logp < 1 and (fg.get("Carboxylic_Acid", 0) or fg.get("Phenol", 0)):
                flags["acid_risk"] = True

            # reducing_sugar_risk — alcohol + aldehyde = reducing sugar pattern
            if fg.get("Alcohol", 0) and fg.get("Aldehyde", 0):
                flags["reducing_sugar_risk"] = True

        except Exception:
            pass

        return flags

    def get_api_flags(self, api_smiles: str) -> dict:
        """
        Detect incompatibility flags for an API from its SMILES using SMARTS matching.
        """
        flags = {flag: False for flag in HANDBOOK_FLAGS}

        if not api_smiles or not isinstance(api_smiles, str):
            return flags

        try:
            mol = Chem.MolFromSmiles(api_smiles)
            if mol is None:
                return flags

            for flag, smarts in API_SMARTS.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    flags[flag] = True

        except Exception:
            pass

        return flags

    # ─────────────────────────────────────────────
    # CLASH SCORING
    # ─────────────────────────────────────────────

    def compute_clash_score(self, flags_a: dict, flags_b: dict) -> tuple[float, list[str]]:
        """
        Compute clash score between two flag dicts.
        Returns (score, [clash_labels])
        """
        score = 0.0
        clashes = []

        for (flag_x, flag_y), (severity, label) in CLASH_MATRIX.items():
            if flags_a.get(flag_x) and flags_b.get(flag_y):
                score += severity
                if label not in clashes:
                    clashes.append(label)

        return score, clashes

    # ─────────────────────────────────────────────
    # MAIN RERANK
    # ─────────────────────────────────────────────

    def rerank(
        self,
        api_smiles: str,
        predictions: list[tuple[str, float]],
        top_n_candidates: int = 30,
    ) -> list[dict]:
        """
        Rerank GNN predictions using incompatibility penalties.

        Args:
            api_smiles:       SMILES string of the API
            predictions:      list of (excipient_name, gnn_prob) from predict_excipients()
            top_n_candidates: how many top GNN candidates to consider before reranking

        Returns:
            list of dicts with keys:
                name, gnn_score, clash_score, final_score,
                api_clashes, pairwise_clashes, source
        """
        # Work on top-N candidates only
        candidates = predictions[:top_n_candidates]

        # Get API flags
        api_flags = self.get_api_flags(api_smiles)
        active_api_flags = [f for f, v in api_flags.items() if v]

        # Get excipient flags for all candidates
        exc_flags = {
            name: self.get_excipient_flags(name)
            for name, _ in candidates
        }

        # Determine flag source for display
        def get_source(name):
            key = name.upper().strip()
            if key in self.handbook_db:
                return "handbook"
            if key in self.features_db:
                return "inferred"
            return "unknown"

        # Score each candidate
        scored = []
        for name, gnn_prob in candidates:
            flags = exc_flags[name]

            # Layer 1 — API vs excipient clash
            api_clash_score, api_clash_labels = self.compute_clash_score(api_flags, flags)

            scored.append({
                "name":           name,
                "gnn_score":      gnn_prob,
                "api_clash_score": api_clash_score,
                "api_clashes":    api_clash_labels,
                "pairwise_clash_score": 0.0,
                "pairwise_clashes": [],
                "flags":          flags,
                "source":         get_source(name),
            })

        # Layer 2 — Pairwise excipient-excipient clashes
        # For each candidate, accumulate penalty from clashing with other candidates
        for i, entry_a in enumerate(scored):
            for j, entry_b in enumerate(scored):
                if i >= j:
                    continue
                pair_score, pair_labels = self.compute_clash_score(
                    entry_a["flags"], entry_b["flags"]
                )
                if pair_score > 0:
                    entry_a["pairwise_clash_score"] += pair_score
                    entry_b["pairwise_clash_score"] += pair_score
                    for label in pair_labels:
                        clash_note = f"{label} (with {entry_b['name']})"
                        if clash_note not in entry_a["pairwise_clashes"]:
                            entry_a["pairwise_clashes"].append(clash_note)
                        clash_note_b = f"{label} (with {entry_a['name']})"
                        if clash_note_b not in entry_b["pairwise_clashes"]:
                            entry_b["pairwise_clashes"].append(clash_note_b)

        # Compute final score
        for entry in scored:
            entry["pairwise_clash_score"] = min(entry["pairwise_clash_score"], 1.0)
        for entry in scored:
            entry["final_score"] = (
                entry["gnn_score"]
                - self.clash_lambda    * entry["api_clash_score"]
                - self.pairwise_lambda * entry["pairwise_clash_score"]
            )
            # Remove internal flags dict from output
            del entry["flags"]

        # Sort by final score
        scored.sort(key=lambda x: x["final_score"], reverse=True)

        return scored

    def format_output(self, reranked: list[dict], api_flags: dict = None) -> str:
        """
        Format reranked results for display in predict.py.
        """
        lines = []
        lines.append(f"\n{'─'*60}")
        lines.append("RERANKED EXCIPIENTS (with incompatibility penalties)")
        lines.append(f"{'─'*60}")
        lines.append(
            f"{'#':<3} {'Excipient':<35} {'GNN':>6} {'Final':>6} {'Flags'}"
        )
        lines.append("─" * 60)

        for i, entry in enumerate(reranked, 1):
            has_clash = entry["api_clashes"] or entry["pairwise_clashes"]
            flag_str = ""
            if entry["api_clashes"]:
                flag_str += f"⚠ API: {', '.join(entry['api_clashes'])}"
            if entry["pairwise_clashes"]:
                flag_str += f"  ⚠ Pair: {'; '.join(entry['pairwise_clashes'][:2])}"
            if not has_clash:
                flag_str = "✓ safe"
            if entry["source"] == "unknown":
                flag_str += " [flags unknown]"

            lines.append(
                f"{i:<3} {entry['name']:<35} "
                f"{entry['gnn_score']:>5.1%} "
                f"{entry['final_score']:>5.1%}  "
                f"{flag_str}"
            )

        return "\n".join(lines)
