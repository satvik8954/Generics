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
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              matthews_corrcoef, accuracy_score, balanced_accuracy_score)

from config import FUNCTIONAL_CSV, ROLE_NAMES
from roles import build_unii_to_roles
from features import load_api_features
from predict import predict_formulation

# Manual overrides for excipient names that the fuzzy substring lookup in
# lookup_unii() gets wrong (e.g. "Hydroxypropyl Methyl Cellulose" and
# "Microcrystalline Cellulose" both contain the word "Cellulose", so a loose
# substring-in-either-direction match can silently resolve to the wrong one)
# or can't find at all under that exact wording.
MANUAL_UNII_OVERRIDES = {
    "HYDROXYPROPYL METHYL CELLULOSE": "3NXW29V3WO",  # = Hypromellose, NOT MCC
    "HYDROXYETHYLCELLULOSE": "T4V6TWG28D",  # generic "unspecified" grade, NOT MCC
    "DISODIUM PHOSPHATE": "22ADO53M6F",  # = Sodium Phosphate, Dibasic, Anhydrous
    "DISODIUM HYDROGEN PHOSPHATE": "22ADO53M6F",  # same compound, different name -- NOT bare "Sodium"
    "MONOBASIC SODIUM PHOSPHATE": "KH7I04HPUU",  # = Sodium Phosphate, Monobasic, Anhydrous
    "SODIUM STARCH GLYCOLATE": "H8AV0SQX4D",  # generic Type A, NOT plain corn starch
    "SILICIFIED MICROCRYSTALLINE CELLULOSE": "88X4A2YW6T",  # generic 125um grade, NOT plain MCC
    "POLYVINYLPYRROLIDONE": "FZ989GH94E",  # = generic unspecified Povidone/PVP
    "ETHANOL": "3K9958V90M",  # = "Alcohol" (USP name for ethanol), NOT Phenoxyethanol
    "STRAWBERRY FLAVOR": "4J2TY8Y81V",  # dedicated Strawberry entry, NOT bare "Berry"
    # (BERRY is a substring literally embedded inside the word STRAWBERRY --
    # they're two different real flavor entries in the source file)
    "METHACRYLIC ACID COPOLYMER": "5KY68S2577",  # = Eudragit S100 (1:2 ratio), NOT the bare monomer
    "TITANIUM DIOXIDE": None,  # NOT in this dataset at all -- bare "Titanium" (elemental) is a
    # different substance; explicitly block rather than silently accept a wrong match
}

GOLD_INPUT_PATH = "data/gold_set_formulations_final.csv"
GOLD_RESULTS_PATH = "data/gold_set_results.csv"

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
    # new phrasing from patent-derived examples:
    "disintegrating agent": "disintegrant",
    "alkaline agent": "alkalizing_agent",
    "viscosity enhancer": "suspending_thickening_agent",
    "aromatic ingredient": "flavoring_agent",
    "processing solvent": "solvent", "granulation solvent": "solvent",
    "volatilizable organic solvent": "solvent",
    "enteric coating material": "coating_agent", "enteric coating polymer": "coating_agent",
    "water-soluble polymer used for seal coating": "coating_agent",
    "buffer system component": "buffering_agent",
    "matrix former": "controlled_release",  # HPMC-type extended-release matrix polymers
    "thickener": "suspending_thickening_agent",
    "enteric polymer": "coating_agent",
    # intentionally unmapped (not in your 24-role taxonomy, or too vague/explicitly
    # withheld by the patent to trust as ground truth):
    # "dye", "colorant", "color", "inert processing aid" (patent explicitly avoids
    # calling this lubricant/glidant), "sugar alcohol" (not a functional role),
    # "inert core" (a physical component, not a functional excipient role)
    # "not explicitly stated", "miscellaneous ingredient", "sub-coating material",
    # "component of seal coating", "component of enteric coating" — patent doesn't
    # commit to a specific functional claim, so these stay untestable rather than
    # being force-mapped to a guess
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


def build_name_to_unii(verbose: bool = True) -> dict:
    """
    Builds {alias_name: unii} from FUNCTIONAL_CSV (unii is embedded directly
    per row now, no separate join file). Splits excipient_names on commas to
    recover individual aliases, BUT filters out any fragment that appears in
    2+ DIFFERENT rows before ever inserting it as a key.

    Why: many rows store ONE true USP/NF name that itself contains internal
    commas (e.g. "Sodium Phosphate, Dibasic, Anhydrous" is a single name, not
    three aliases). Blind splitting shreds these into bare fragments
    ("SODIUM PHOSPHATE", "DIBASIC", "ANHYDROUS") — and a bare fragment like
    "DIBASIC" or "ANHYDROUS" is a generic modifier word that shows up across
    several genuinely different excipients' names, so inserting it as a
    standalone key means whichever row happens to be processed first wins
    that key for every unrelated excipient sharing the word. This dataset
    has no num_names column (an earlier fix used that from a different
    source file) to tell us up front which fields are safe to split, so
    instead: build a frequency count of every fragment across ALL rows
    first, then only ever insert a fragment as a key if it's UNIQUE to one
    row. Fragments appearing in multiple rows are dropped entirely rather
    than guessed at — the full, unsplit excipient_names field is always
    inserted too, so nothing is silently unresolvable, just less eagerly
    split.
    """
    df = pd.read_csv(FUNCTIONAL_CSV)

    # pass 1: count how many DIFFERENT rows each fragment appears in
    fragment_row_count = Counter()
    row_fragments = []
    for _, row in df.iterrows():
        names_field = str(row["excipient_names"])
        frags = set(f.strip().upper() for f in names_field.split(",") if f.strip())
        row_fragments.append((row["unii"], names_field.strip().upper(), frags))
        for f in frags:
            fragment_row_count[f] += 1

    # pass 2: only insert fragments unique to one row; always insert the
    # full unsplit field too, as a safety net
    name_to_unii = {}
    collisions = []
    for unii, full_field, frags in row_fragments:
        if pd.isna(unii):
            continue
        if full_field not in name_to_unii:
            name_to_unii[full_field] = unii
        for frag in frags:
            if fragment_row_count[frag] >= 2:
                continue  # generic/ambiguous fragment, seen in 2+ rows -- skip
            # A bare single-word fragment (no space) is a dangerous key even
            # if it's row-unique: the separate substring-fallback matcher in
            # lookup_unii() will match it against ANY longer name containing
            # that word (e.g. "CELLULOSE" -> would match "Hydroxypropyl
            # Methyl Cellulose", "Hydroxyethylcellulose", etc., all silently
            # resolving to whichever row this bare fragment came from). Only
            # allow a single-word fragment through if that word genuinely IS
            # the whole field (e.g. "TALC"), not a decomposition byproduct of
            # a longer comma-joined name (e.g. "CELLULOSE" pulled out of
            # "Cellulose, Microcrystalline").
            if " " not in frag and frag != full_field:
                continue
            if frag in name_to_unii and name_to_unii[frag] != unii:
                collisions.append((frag, name_to_unii[frag], unii))
                continue
            name_to_unii[frag] = unii

    if verbose and collisions:
        seen = set()
        print(f"[WARN] {len(collisions)} unexpected collisions among "
              f"supposedly-unique fragments in {FUNCTIONAL_CSV}:")
        for key, kept, rejected in collisions:
            if key in seen:
                continue
            seen.add(key)
            print(f"    {key!r}: kept {kept}, rejected {rejected}")

    return name_to_unii


def lookup_unii(excipient_name: str, name_to_unii: dict):
    key = excipient_name.strip().upper()
    if key in MANUAL_UNII_OVERRIDES:
        return MANUAL_UNII_OVERRIDES[key], "manual_override"
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


GOLD_CONFUSION_MATRIX_PATH = "data/gold_set_confusion_matrix.csv"
GOLD_PER_CLASS_METRICS_PATH = "data/gold_set_per_class_metrics.csv"


def build_metrics_labels(testable: pd.DataFrame):
    """
    Turns the (possibly multi-label) testable rows into a single y_true /
    y_pred pair per row so sklearn's single-label metrics functions apply.

    textbook_role_canonical can list 2+ acceptable roles (e.g.
    "sweetening_agent|solvent"), and the existing verdict logic already
    counts a row CORRECT if predicted_role matches ANY of them. To keep
    the confusion matrix / precision / recall consistent with that same
    notion of "correct":
      - if predicted_role is one of the acceptable truths -> y_true is set
        to predicted_role too, so the row lands on the diagonal (agrees
        with the verdict already computed).
      - otherwise -> y_true is the FIRST listed truth role. This is a
        simplification (multi-label ground truth forced into single-label
        metrics) and is flagged via `is_multilabel_row` so you can see how
        many mismatches involved an ambiguous truth set.
    """
    y_true, y_pred, is_multilabel_row = [], [], []
    for _, row in testable.iterrows():
        truths = row["textbook_role_canonical"].split("|")
        pred = row["predicted_role"]
        y_true.append(pred if pred in truths else truths[0])
        y_pred.append(pred)
        is_multilabel_row.append(len(truths) > 1)
    return np.array(y_true), np.array(y_pred), np.array(is_multilabel_row)


def print_and_save_metrics(testable: pd.DataFrame):
    """
    Prints + saves:
      - overall accuracy and balanced accuracy
      - Matthews correlation coefficient (MCC) -- a single number that's
        far more honest than accuracy on a class-imbalanced role set like
        this one (a model that just always predicts "filler" can still put
        up decent raw accuracy; MCC punishes that much harder)
      - per-class precision / recall / F1 / support (classification_report
        equivalent, but built manually so it's easy to also dump to CSV)
      - full confusion matrix (rows = textbook truth, cols = predicted),
        saved to CSV since it's too wide to read comfortably in a terminal
    """
    if len(testable) == 0:
        print("\n[METRICS] No testable rows -- nothing to compute.")
        return

    y_true, y_pred, is_multilabel_row = build_metrics_labels(testable)
    n_multilabel = int(is_multilabel_row.sum())

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print("METRICS (confusion matrix / precision / recall / F1 / MCC)")
    print(f"{'=' * 60}")
    if n_multilabel:
        print(f"[NOTE] {n_multilabel}/{len(testable)} testable rows had multiple acceptable "
              f"textbook roles. A correct-either-way match is scored on-diagonal; a mismatch "
              f"is scored against only the FIRST listed truth role, so these metrics are a "
              f"slightly stricter lower bound than the plain accuracy printed above.")
    print(f"\nAccuracy:          {acc:.4f}")
    print(f"Balanced accuracy:  {bal_acc:.4f}  (per-class recall averaged -- robust to class imbalance)")
    print(f"Matthews corrcoef:  {mcc:.4f}  (-1..1, 0 = random; best single summary stat for imbalanced multi-class)")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({
        "role": labels, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    }).sort_values("support", ascending=False)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    print(f"\nPer-class (sorted by support):")
    print(per_class.to_string(index=False, formatters={
        "precision": "{:.3f}".format, "recall": "{:.3f}".format, "f1": "{:.3f}".format}))
    print(f"\n  macro avg      precision={macro_p:.3f}  recall={macro_r:.3f}  f1={macro_f1:.3f}")
    print(f"  weighted avg   precision={weighted_p:.3f}  recall={weighted_r:.3f}  f1={weighted_f1:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels],
                          columns=[f"pred:{l}" for l in labels])

    os.makedirs(os.path.dirname(GOLD_CONFUSION_MATRIX_PATH), exist_ok=True)
    cm_df.to_csv(GOLD_CONFUSION_MATRIX_PATH)
    per_class.to_csv(GOLD_PER_CLASS_METRICS_PATH, index=False)
    print(f"\n[OK] Confusion matrix saved to {GOLD_CONFUSION_MATRIX_PATH}")
    print(f"[OK] Per-class precision/recall/F1 saved to {GOLD_PER_CLASS_METRICS_PATH}")


def evaluate_gold_set():
    if not os.path.exists(GOLD_INPUT_PATH):
        generate_starter_template()

    gold = pd.read_csv(GOLD_INPUT_PATH)

    if "api_unii" not in gold.columns:
        print("[WARN] gold_set_formulations.csv has no api_unii column — every")
        print("       formulation will be evaluated with ZEROED API features,")
        print("       which does not match how the model was trained. Add an")
        print("       api_unii column (one value per formulation_id) to fix this.")
        gold["api_unii"] = None

    has_frozen_unii = "excipient_unii" in gold.columns
    name_to_unii = None  # only built lazily, if any row actually needs the fallback
    unii_to_roles = build_unii_to_roles()

    print("Loading API descriptors...")
    api_to_feats = load_api_features()  # real descriptors, not {}

    all_rows = []
    n_missing_api = 0
    n_fallback_lookups = 0
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

        # resolve UNIIs for every excipient in this formulation first --
        # prefer the frozen excipient_unii column (verified once, ahead of
        # time) over re-deriving it from name-matching every run. Every
        # collision bug we've hit (HPMC/MCC, Ethanol/Phenoxyethanol,
        # Strawberry/Berry, the phosphates...) came from doing this
        # resolution live, repeatedly, against a source file that can
        # itself change shape between runs. Fuzzy lookup is now only a
        # fallback for a brand-new row that hasn't been verified yet.
        resolved = []
        for _, row in group.iterrows():
            frozen = row.get("excipient_unii") if has_frozen_unii else None
            if pd.notna(frozen) and str(frozen).strip():
                unii, match_type = str(frozen).strip(), "frozen"
            else:
                if name_to_unii is None:
                    name_to_unii = build_name_to_unii()
                unii, match_type = lookup_unii(row["excipient_name"], name_to_unii)
                n_fallback_lookups += 1
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
    if n_fallback_lookups:
        print(f"[NOTE] {n_fallback_lookups} row(s) had no frozen excipient_unii and used the "
              f"live name-matching fallback — verify these and add them to the excipient_unii "
              f"column once confirmed correct, so future runs don't re-derive them.")
    if n_missing_api:
        print(f"[WARN] {n_missing_api} formulation(s) evaluated with ZEROED API features "
              f"(no api_unii given) — accuracy above may not reflect the trained model's "
              f"real behavior for those rows.")
    if n_total:
        print(f"Accuracy on testable rows: {n_correct}/{n_total} = {n_correct/n_total*100:.1f}%")

    print_and_save_metrics(testable)

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