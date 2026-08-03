"""
predict.py — given a new formulation (dosage form + list of excipient UNIIs),
predicts the functional role each excipient is playing, using the trained
model + constrained decoding (never predicts a role the excipient can't
structurally have) + exclusive-role slot resolution (at most one excipient
gets an exclusive role like binder/filler/lubricant per formulation).

Usage as a script (edit the example at the bottom), or import
`predict_formulation` directly:

    from predict import predict_formulation
    predict_formulation("TABLET", ["70097M6I30", "OP1R32D61U", "M28OL1HH48"])
"""

import pickle

from collections import Counter

import numpy as np

from config import MODEL_PATH, ROLE_NAMES, NUM_ROLES, EXCLUSIVE_ROLES, ROLE_CAPACITY
from roles import build_unii_to_roles
from features import build_feature_vector, load_api_features, apply_coocc_scaler


def get_probs(clf, label_encoder, X):
    probs_partial = clf.predict_proba(X)
    probs = np.zeros((len(X), NUM_ROLES), dtype=np.float32)
    for col_idx, encoded_label in enumerate(clf.classes_):
        original_role_idx = int(label_encoder.classes_[int(encoded_label)])
        probs[:, original_role_idx] = probs_partial[:, col_idx]
    return probs


def predict_formulation(dosage_form: str, excipient_uniis: list, api_unii: str = None,
                         unii_to_roles: dict = None, api_to_feats: dict = None):
    """
    Returns list of dicts: [{"unii": ..., "role": ..., "confidence": ...}, ...]
    one entry per excipient, in the same order as excipient_uniis.
    """
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    clf = saved["model"]
    label_encoder = saved["label_encoder"]
    coocc_scaler = saved["coocc_scaler"]

    if unii_to_roles is None:
        unii_to_roles = build_unii_to_roles()
    if api_to_feats is None:
        api_to_feats = load_api_features()

    # score every (excipient, candidate role) pair
    scored = []  # (score, unii, role_idx)
    per_unii_probs = {}
    for unii in excipient_uniis:
        others = [u for u in excipient_uniis if u != unii]
        feats = build_feature_vector(unii, dosage_form, others, unii_to_roles,
                                      api_unii=api_unii, api_to_feats=api_to_feats).reshape(1, -1)
        feats = apply_coocc_scaler(feats, coocc_scaler)
        cap_mask = feats[0, :NUM_ROLES]
        if cap_mask.sum() == 0:
            per_unii_probs[unii] = None  # no known capability at all
            continue
        p = get_probs(clf, label_encoder, feats)[0] * cap_mask
        per_unii_probs[unii] = p
        for role_idx in np.nonzero(cap_mask)[0]:
            scored.append((p[role_idx], unii, role_idx))

    scored.sort(key=lambda x: -x[0])

    assigned = {}
    # NOTE: this used to be `filled_exclusive = set()`, which hard-coded
    # capacity 1 for every exclusive role regardless of config.ROLE_CAPACITY
    # -- so raising ROLE_CAPACITY["solvent"] to 2 (to allow real multi-
    # cosolvent systems like water + ethanol + glycerin + propylene glycol)
    # had NO effect here, because this function never even imported
    # ROLE_CAPACITY. Training-label generation (weak_labels.py Pass 2)
    # respected the raised capacity; inference (this function) silently
    # didn't, so the fix looked like it wasn't working even though it was
    # -- it just never reached this code path. Now a Counter tracks how
    # many winners each exclusive role has taken so far, and a role only
    # gets blocked once it hits its own configured capacity.
    filled_exclusive_counts = Counter()
    for score, unii, role_idx in scored:
        if unii in assigned:
            continue
        role = ROLE_NAMES[role_idx]
        if role in EXCLUSIVE_ROLES and filled_exclusive_counts[role] >= ROLE_CAPACITY.get(role, 1):
            continue
        assigned[unii] = {"role": role, "confidence": float(score)}
        if role in EXCLUSIVE_ROLES:
            filled_exclusive_counts[role] += 1

    results = []
    for unii in excipient_uniis:
        if unii in assigned:
            results.append({"unii": unii, **assigned[unii]})
        else:
            reason = "no known HPE capability" if per_unii_probs.get(unii) is None else "slot already taken"
            results.append({"unii": unii, "role": None, "confidence": 0.0, "note": reason})
    return results


if __name__ == "__main__":
    # example: alpha-tocopherol tablet excipients from oral_only.csv row 3
    example_form = "TABLET"
    example_excipients = [
        "3SY5LH9PMK",  # anhydrous lactose
        "M28OL1HH48",  # croscarmellose sodium
        "H3R47K3TBD",  # FD&C Blue No. 1
        "EWQ57Q8I5X",  # lactose monohydrate
        "70097M6I30",  # magnesium stearate
        "K0KQV10C35",  # povidone K25
        "368GB5141J",  # sodium lauryl sulfate
        "C151H8M554",  # sucrose
    ]
    predictions = predict_formulation(example_form, example_excipients)
    print(f"\nPredicted roles for {example_form} formulation:\n")
    for p in predictions:
        if p["role"]:
            print(f"  {p['unii']:<15} -> {p['role']:<25} (confidence {p['confidence']:.3f})")
        else:
            print(f"  {p['unii']:<15} -> UNASSIGNED  ({p['note']})")