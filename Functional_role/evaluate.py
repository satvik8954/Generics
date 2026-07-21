"""
evaluate.py — evaluates a trained Task B model.

Reports:
    1. Raw (unconstrained) accuracy/F1 on the held-out val split
    2. Constrained-decoding accuracy/F1 (predictions masked to the
       excipient's own raw HPE capability set — never predicts a
       structurally impossible role)
    3. Same, but split into "easy" (excipient only ever has 1 possible role)
       vs "hard" (2+ possible roles) subsets — the hard subset is the real
       signal, since the easy subset is trivially solved by masking alone
    4. Agreement with the Pass 2 heuristic on a sample of ambiguous cases
       the model never trained on (out-of-sample cross-check between the
       two independent weak-labeling signals)

Usage:
    python evaluate.py
    python evaluate.py --pass2-sample 1000
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import MODEL_PATH, PASS2_CSV_PATH, ROLE_NAMES, NUM_ROLES
from roles import build_unii_to_roles
from features import build_feature_vector, load_formulation_excipients, load_api_features


def get_probs(clf, label_encoder, X):
    """Returns an (n, NUM_ROLES) probability matrix in ORIGINAL role-index
    space, regardless of which encoded classes the model saw in training."""
    probs_partial = clf.predict_proba(X)
    probs = np.zeros((len(X), NUM_ROLES), dtype=np.float32)
    for col_idx, encoded_label in enumerate(clf.classes_):
        original_role_idx = int(label_encoder.classes_[int(encoded_label)])
        probs[:, original_role_idx] = probs_partial[:, col_idx]
    return probs


def constrained_predict(probs, capability_masks):
    preds = []
    for i in range(len(probs)):
        mask = capability_masks[i]
        p = probs[i] * mask
        if p.sum() == 0:
            p = probs[i]  # fallback if capability mask wipes everything out
        preds.append(int(np.argmax(p)))
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass2-sample", type=int, default=500)
    args = parser.parse_args()

    print("Loading model + stashed val split...")
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    clf, model_type = saved["model"], saved["model_type"]
    label_encoder = saved["label_encoder"]
    X_val, y_val_encoded = saved["X_val"], saved["y_val"]
    y_val = label_encoder.inverse_transform(y_val_encoded)  # back to original ROLE_TO_IDX space

    capability_masks = X_val[:, :NUM_ROLES]

    print("\n[1] Unconstrained eval...")
    y_pred_raw_encoded = clf.predict(X_val)
    y_pred_raw = label_encoder.inverse_transform(y_pred_raw_encoded)
    print(f"  Accuracy: {accuracy_score(y_val, y_pred_raw):.4f}")
    print(f"  Macro F1: {f1_score(y_val, y_pred_raw, average='macro'):.4f}")

    print("\n[2] Constrained-decoding eval (masked to HPE capability)...")
    probs = get_probs(clf, label_encoder, X_val)
    y_pred = constrained_predict(probs, capability_masks)
    print(f"  Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"  Macro F1: {f1_score(y_val, y_pred, average='macro'):.4f}")

    print("\n[3] Easy vs hard subset...")
    cap_sums = capability_masks.sum(axis=1)
    hard_idx = np.where(cap_sums >= 2)[0]
    easy_idx = np.where(cap_sums < 2)[0]
    print(f"  Easy (1 possible role): {len(easy_idx)} examples")
    if len(easy_idx) > 0:
        print(f"    Accuracy: {accuracy_score(y_val[easy_idx], y_pred[easy_idx]):.4f}")
    print(f"  Hard (2+ possible roles, real disambiguation): {len(hard_idx)} examples")
    if len(hard_idx) > 0:
        print(f"    Accuracy: {accuracy_score(y_val[hard_idx], y_pred[hard_idx]):.4f}")
        print(f"    Macro F1: {f1_score(y_val[hard_idx], y_pred[hard_idx], average='macro'):.4f}")

    present = sorted(set(y_val[hard_idx].tolist()) | set(y_pred[hard_idx].tolist())) if len(hard_idx) else []
    if present:
        print("\n  Per-class report (hard subset only):")
        print(classification_report(y_val[hard_idx], y_pred[hard_idx], labels=present,
                                     target_names=[ROLE_NAMES[i] for i in present], zero_division=0))

    print(f"\n[4] Agreement with Pass 2 heuristic (out-of-sample, n={args.pass2_sample})...")
    unii_to_roles = build_unii_to_roles()
    formulation_excipients = load_formulation_excipients()
    api_to_feats = load_api_features()
    pass2 = pd.read_csv(PASS2_CSV_PATH)
    sample = pass2.sample(min(args.pass2_sample, len(pass2)), random_state=1)

    agree, total = 0, 0
    for _, row in sample.iterrows():
        other_uniis = formulation_excipients.get(row["row_idx"], [])
        feats = build_feature_vector(row["excipient_unii"], row["dosage_form"],
                                      other_uniis, unii_to_roles,
                                      api_unii=row["api_unii"], api_to_feats=api_to_feats).reshape(1, -1)
        if feats[0, :NUM_ROLES].sum() == 0:
            continue
        p = get_probs(clf, label_encoder, feats)[0]
        p = p * feats[0, :NUM_ROLES]
        pred_role = ROLE_NAMES[int(np.argmax(p))]
        total += 1
        if pred_role == row["role"]:
            agree += 1
    if total:
        print(f"  Agreement: {agree}/{total} = {agree/total*100:.1f}%")
        print("  (Low/moderate agreement is expected and fine — it tells you where")
        print("   the two independent weak-label signals disagree, which is exactly")
        print("   where manual eyeballing is most valuable.)")


if __name__ == "__main__":
    main()