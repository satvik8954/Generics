"""
gold_set_evaluate.py — batch-compares model predictions against real,
textbook-verified excipient role assignments (e.g. Remington formulation
tables), instead of checking the model against its own weak-label heuristic.

This is the actual ground-truth validation loop we've been missing —
every previous accuracy number was the model agreeing with a rule that
generated its own training labels. This one checks against real pharmacist-
verified formulations.

WORKFLOW:
    1. Open gold_set_formulations.csv (a starter template with 2 example
       formulations is generated below if it doesn't exist yet).
    2. Add more rows as you find more textbook formulation tables —
       one row per excipient, grouped by formulation_id. Every row in a
       formulation_id group must carry the SAME api_unii value.
    3. Run: python gold_set_evaluate.py
    4. Get per-row verdicts + running accuracy stats, saved to
       gold_set_results.csv

CSV FORMAT (gold_set_formulations.csv):
    formulation_id, formulation_name, dosage_form, api_unii, excipient_name, textbook_role

    api_unii: the UNII of the formulation's active ingredient, used to pull
        its RDKit descriptors from config.API_FEATURES_CSV — this is what
        the model was actually trained on (see features.py), so leaving it
        blank means the model falls back to a zeroed/"average API" feature
        vector instead of the real one, which is NOT what evaluate.py does
        for the held-out split. Leave blank only if you don't know the UNII
        yet; the script will warn you and evaluate with zeroed API features
        for that formulation rather than silently guessing.

    textbook_role can list multiple roles comma-separated (e.g.
    "Sweetener, cosolvent") — the row counts as correct if the model's
    prediction matches ANY of the listed roles.
"""

import os
import re

import pandas as pd

from config import UNII_CSV, ROLE_NAMES
from roles import build_unii_to_roles
from features import load_api_features
from predict import predict_formulation

GOLD_INPUT_PATH = "Data/gold_set_formulations_new.csv"
GOLD_RESULTS_PATH = "Data/gold_set_results_new.csv"

# Maps free-text textbook wording -> canonical role names. Separate from
# ROLE_TAXONOMY (which maps raw HPE monograph category strings) because
# textbook prose uses more casual phrasing than HPE's formal categories.
TEXTBOOK_ROLE_ALIASES = {
    "binder": "binder", "binding agent": "binder",
    "filler": "filler", "diluent": "filler", "bulking agent": "filler",
    "disintegrant": "disintegrant",
    "lubricant": "lubricant",
    "glidant": "glidant", "flow aid": "glidant",
    "coating agent": "coating_agent", "coating": "coating_agent",
    "film former": "coating_agent", "film-former": "coating_agent",
    "controlled release": "controlled_release", "release modifier": "controlled_release",
    "sustained release": "controlled_release", "extended release": "controlled_release",
    "solvent": "solvent", "cosolvent": "solvent", "vehicle": "solvent",
    "surfactant": "surfactant", "wetting agent": "surfactant", "emulsifier": "surfactant",
    "emulsifying agent": "surfactant",
    "plasticizer": "plasticizer",
    "alkalizing agent": "alkalizing_agent",
    "acidifying agent": "acidifying_agent",
    "humectant": "humectant",
    "sweetener": "sweetening_agent", "sweetening agent": "sweetening_agent",
    "flavor": "flavoring_agent", "flavoring agent": "flavoring_agent", "flavour": "flavoring_agent",
    "preservative": "preservative", "antimicrobial preservative": "preservative",
    "antioxidant": "antioxidant",
    "buffering agent": "buffering_agent", "buffer": "buffering_agent",
    "suspending agent": "suspending_thickening_agent", "thickening agent": "suspending_thickening_agent",
    "viscosity agent": "suspending_thickening_agent", "gelling agent": "suspending_thickening_agent",
    "stabilizing agent": "stabilizing_agent", "stabilizer": "stabilizing_agent",
    "solubilizing agent": "solubilizing_agent", "solubilizer": "solubilizing_agent",
    "granulation aid": "granulation_aid",
    "dispersing agent": "dispersing_agent",
    "bioadhesive": "bioadhesive_agent", "mucoadhesive": "bioadhesive_agent",
    # intentionally unmapped (not in your 24-role taxonomy):
    # "dye", "colorant", "color" — dropped earlier for <10 supporting excipients
}


def generate_starter_template():
    """Creates the 2 example formulations you already reviewed, as a
    starting template, if the input file doesn't exist yet.

    api_unii is left BLANK here deliberately — I don't have verified UNII
    codes for donepezil / valproic acid memorized reliably enough to hard-code
    them into a data file that feeds your model's actual features. Fill them
    in from data/api_features.csv (or DailyMed) before running an accuracy
    check you plan to trust.
    """
    rows = [
        # Donepezil ODT (Zydus)
        ("F1", "Donepezil ODT (Zydus)", "ORALLY DISINTEGRATING TABLET", "", "Crospovidone", "Disintegrant"),
        ("F1", "Donepezil ODT (Zydus)", "ORALLY DISINTEGRATING TABLET", "", "Magnesium stearate", "Lubricant"),
        ("F1", "Donepezil ODT (Zydus)", "ORALLY DISINTEGRATING TABLET", "", "Mannitol", "Sweetener"),
        ("F1", "Donepezil ODT (Zydus)", "ORALLY DISINTEGRATING TABLET", "", "Silicon dioxide, colloidal", "Glidant"),
        ("F1", "Donepezil ODT (Zydus)", "ORALLY DISINTEGRATING TABLET", "", "Sucralose", "Sweetener"),
        # Depakene (valproic acid) oral solution
        ("F2", "Depakene oral solution", "SOLUTION", "", "Glycerin", "Sweetener, cosolvent"),
        ("F2", "Depakene oral solution", "SOLUTION", "", "Methylparaben", "Antimicrobial preservative"),
        ("F2", "Depakene oral solution", "SOLUTION", "", "Propylparaben", "Antimicrobial preservative"),
        ("F2", "Depakene oral solution", "SOLUTION", "", "Sorbitol", "Sweetener, cosolvent"),
        ("F2", "Depakene oral solution", "SOLUTION", "", "Purified water", "Solvent"),
        ("F2", "Depakene oral solution", "SOLUTION", "", "Sucrose", "Sweetener"),
        ("F2", "Depakene oral solution", "SOLUTION", "", "FD&C Red No. 40", "Dye"),  # untestable, no role in taxonomy
        ("F2", "Depakene oral solution", "SOLUTION", "", "Cherry flavor", "Flavor"),  # untestable, no UNII
    ]
    df = pd.DataFrame(rows, columns=["formulation_id", "formulation_name", "dosage_form",
                                      "api_unii", "excipient_name", "textbook_role"])
    os.makedirs(os.path.dirname(GOLD_INPUT_PATH), exist_ok=True)
    df.to_csv(GOLD_INPUT_PATH, index=False)
    print(f"[OK] Created starter template at {GOLD_INPUT_PATH}")
    print("     Add more rows (one per excipient, grouped by formulation_id) as you find more textbook examples.")
    print("     NOTE: api_unii is blank in the template — fill it in per formulation_id before trusting results.")


def build_name_to_unii():
    df = pd.read_csv(UNII_CSV)
    name_to_unii = {}
    for _, row in df.iterrows():
        for alias in str(row["excipient_names"]).split(","):
            name_to_unii[alias.strip().upper()] = row["unii"]
    return name_to_unii


def lookup_unii(excipient_name: str, name_to_unii: dict):
    key = excipient_name.strip().upper()
    if key in name_to_unii:
        return name_to_unii[key], "exact"
    # substring fallback (either direction)
    for known_name, unii in name_to_unii.items():
        if key in known_name or known_name in key:
            return unii, "substring"
    return None, "not_found"


def map_textbook_roles(text: str):
    """Splits 'Sweetener, cosolvent' -> ['sweetening_agent', 'solvent'],
    silently dropping phrases with no canonical mapping (e.g. 'Dye')."""
    canonical = []
    unmapped = []
    for phrase in re.split(r"[,/]", text):
        phrase = phrase.strip().lower()
        if not phrase:
            continue
        mapped = TEXTBOOK_ROLE_ALIASES.get(phrase)
        if mapped:
            canonical.append(mapped)
        else:
            unmapped.append(phrase)
    return canonical, unmapped


def evaluate_gold_set():
   

    gold = pd.read_csv(GOLD_INPUT_PATH)

    if "api_unii" not in gold.columns:
        print("[WARN] gold_set_formulations.csv has no api_unii column — every")
        print("       formulation will be evaluated with ZEROED API features,")
        print("       which does not match how the model was trained. Add an")
        print("       api_unii column (one value per formulation_id) to fix this.")
        gold["api_unii"] = None

    name_to_unii = build_name_to_unii()
    unii_to_roles = build_unii_to_roles()

    print("Loading API descriptors...")
    api_to_feats = load_api_features()  # real descriptors, not {}

    all_rows = []
    n_missing_api = 0
    for formulation_id, group in gold.groupby("formulation_id"):
        dosage_form = group["dosage_form"].iloc[0]
        formulation_name = group["formulation_name"].iloc[0]

        api_unii_raw = group["api_unii"].iloc[0]
        api_unii = str(api_unii_raw).strip() if pd.notna(api_unii_raw) and str(api_unii_raw).strip() else None
        if api_unii is None:
            n_missing_api += 1
        elif api_unii not in api_to_feats:
            print(f"[WARN] api_unii '{api_unii}' for {formulation_name!r} not found in "
                  f"{'API_FEATURES_CSV'} — will fall back to zeroed API features for this formulation.")

        # resolve UNIIs for every excipient in this formulation first
        resolved = []
        for _, row in group.iterrows():
            unii, match_type = lookup_unii(row["excipient_name"], name_to_unii)
            resolved.append({"excipient_name": row["excipient_name"], "unii": unii,
                              "match_type": match_type, "textbook_role_raw": row["textbook_role"]})

        # build the candidate excipient list (only those with a resolved unii)
        excipient_uniis = [r["unii"] for r in resolved if r["unii"] is not None]
        preds_by_unii = {}
        if excipient_uniis:
            preds = predict_formulation(dosage_form, excipient_uniis,
                                         api_unii=api_unii, unii_to_roles=unii_to_roles,
                                         api_to_feats=api_to_feats)
            preds_by_unii = {p["unii"]: p for p in preds}

        for r in resolved:
            canonical_truth, unmapped = map_textbook_roles(r["textbook_role_raw"])
            pred_entry = preds_by_unii.get(r["unii"]) if r["unii"] else None
            predicted_role = pred_entry["role"] if pred_entry else None
            confidence = pred_entry["confidence"] if pred_entry else None

            if r["unii"] is None:
                verdict = "UNTESTABLE (no UNII match)"
            elif r["unii"] not in unii_to_roles:
                verdict = "UNTESTABLE (no HPE capability data)"
            elif not canonical_truth:
                verdict = f"UNTESTABLE (textbook role(s) {unmapped} not in your taxonomy)"
            elif predicted_role is None:
                verdict = "UNTESTABLE (model gave no prediction)"
            elif predicted_role in canonical_truth:
                verdict = "CORRECT"
            else:
                verdict = "MISMATCH"

            all_rows.append({
                "formulation_id": formulation_id, "formulation_name": formulation_name,
                "dosage_form": dosage_form, "api_unii": api_unii,
                "excipient_name": r["excipient_name"],
                "unii": r["unii"], "unii_match_type": r["match_type"],
                "textbook_role_raw": r["textbook_role_raw"],
                "textbook_role_canonical": "|".join(canonical_truth),
                "predicted_role": predicted_role, "confidence": confidence,
                "verdict": verdict,
            })

    results = pd.DataFrame(all_rows)
    results.to_csv(GOLD_RESULTS_PATH, index=False)

    testable = results[results["verdict"].isin(["CORRECT", "MISMATCH"])]
    n_correct = (testable["verdict"] == "CORRECT").sum()
    n_total = len(testable)

    print(f"\n{'=' * 60}")
    print(f"GOLD-SET EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total rows: {len(results)}  |  Testable: {n_total}  |  Untestable: {len(results) - n_total}")
    if n_missing_api:
        print(f"[WARN] {n_missing_api} formulation(s) evaluated with ZEROED API features "
              f"(no api_unii given) — accuracy above may not reflect the trained model's "
              f"real behavior for those rows.")
    if n_total:
        print(f"Accuracy on testable rows: {n_correct}/{n_total} = {n_correct/n_total*100:.1f}%")

    print(f"\nUntestable breakdown:")
    print(results[~results["verdict"].isin(["CORRECT", "MISMATCH"])]["verdict"].value_counts().to_string())

    if n_total:
        print(f"\nMismatches (worth reviewing):")
        mismatches = testable[testable["verdict"] == "MISMATCH"]
        if len(mismatches):
            print(mismatches[["formulation_name", "excipient_name", "predicted_role",
                               "textbook_role_canonical", "confidence"]].to_string(index=False))
        else:
            print("  none!")

    print(f"\n[OK] Full results saved to {GOLD_RESULTS_PATH}")
    return results


if __name__ == "__main__":
    evaluate_gold_set()