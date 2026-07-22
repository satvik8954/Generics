"""
weak_labels.py — Pass 1 / Pass 2 weak label generation for Task B.

Pass 1 (unambiguous): after filtering an excipient's HPE capability set by
    dosage-form relevance, if exactly one candidate role remains, label it
    directly. Confidence = 1.0. Pure logical elimination, no guessing.

Pass 2 (greedy, slot-constrained): for excipients still left with 2+
    candidates, score each (excipient, role) pair using an empirical prior
    P(role | excipient) estimated from Pass 1 occurrences (Laplace-smoothed,
    with a minimum-sample-count fallback to the global role prior), penalize
    by number of candidates, then greedily assign roles per formulation —
    "exclusive" roles (binder, filler, lubricant, etc.) get at most one
    excipient per formulation.

Run directly to regenerate weak_role_labels.pkl + the two CSVs:
    python weak_labels.py
"""

import json
import pickle
import os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from config import (ORAL_CSV, PICKLE_PATH, HIGH_CONF_CSV_PATH, PASS2_CSV_PATH,
                     ROLE_NAMES, EXCLUSIVE_ROLES, MIN_UNII_COUNT,
                     GENERIC_ROLES, GENERIC_ROLE_PENALTY)
from roles import build_unii_to_roles, allowed_roles_for_form, form_bucket


def load_formulations():
    df = pd.read_csv(ORAL_CSV)
    formulations = []
    for idx, row in df.iterrows():
        try:
            items = json.loads(row["inactive_ingredients"])
        except Exception:
            continue
        uniis = [str(it.get("unii", "")).strip() for it in items if it.get("unii")]
        if not uniis:
            continue
        formulations.append({
            "row_idx": idx,
            "api_unii": row["api_unii"],
            "dosage_form": row["primary_dosage_form"],
            "excipient_uniis": uniis,
        })
    print(f"  Loaded {len(formulations)} formulations with parseable excipients")
    return formulations


def pass1_unambiguous(formulations, unii_to_roles):
    results = []
    for f in formulations:
        allowed = allowed_roles_for_form(f["dosage_form"])
        exc_candidates = {}
        for unii in f["excipient_uniis"]:
            caps = unii_to_roles.get(unii, set())
            candidates = caps & allowed
            if candidates:
                exc_candidates[unii] = candidates

        assignments = []
        remaining = {}
        for unii, cands in exc_candidates.items():
            if len(cands) == 1:
                role = next(iter(cands))
                assignments.append({"unii": unii, "role": role,
                                     "confidence": 1.0, "source": "pass1"})
            else:
                remaining[unii] = cands

        results.append({
            "row_idx": f["row_idx"], "api_unii": f["api_unii"],
            "dosage_form": f["dosage_form"], "bucket": form_bucket(f["dosage_form"]),
            "assignments": assignments, "remaining": remaining,
        })

    n_pass1 = sum(len(r["assignments"]) for r in results)
    n_remaining = sum(len(r["remaining"]) for r in results)
    print(f"  Pass 1: {n_pass1} unambiguous labels, {n_remaining} still ambiguous")
    return results


def estimate_priors(results):
    """
    Returns (prior_fn, sample_count_fn).

    prior_fn(unii, role, bucket) -> smoothed P(role | unii, bucket), with a
        3-level fallback chain:
          1. (unii, bucket)-specific prior, if that combo has
             >= MIN_UNII_COUNT Pass-1 occurrences (real, dosage-form-aware
             evidence for THIS excipient in THIS kind of formulation)
          2. unii-level prior across ALL buckets combined, if that has
             >= MIN_UNII_COUNT occurrences (some evidence for this
             excipient, just not bucket-specific)
          3. global role prior (no real evidence for this excipient at all)

    This exists because a single unii-level prior silently mixes e.g.
    "mannitol as filler in tablets" (hundreds of occurrences) with
    "mannitol as sweetener in liquids/ODTs" (a couple dozen) into one
    number that's dominated by whichever dosage form is more common
    overall — wrongly overriding the sweetener signal even when scoring
    a liquid/ODT formulation specifically.

    sample_count_fn(unii, bucket=None) -> how many Pass-1 occurrences back
        this excipient's prior at the given bucket (or across all buckets
        if bucket=None) — diagnostic, also used for the GENERIC_ROLES
        penalty check.
    """
    unii_bucket_role_counts = defaultdict(lambda: defaultdict(Counter))
    unii_role_counts = defaultdict(Counter)
    global_role_counts = Counter()
    for r in results:
        bucket = r["bucket"]
        for a in r["assignments"]:
            unii_bucket_role_counts[a["unii"]][bucket][a["role"]] += 1
            unii_role_counts[a["unii"]][a["role"]] += 1
            global_role_counts[a["role"]] += 1
    total = sum(global_role_counts.values()) or 1
    global_prior = {role: count / total for role, count in global_role_counts.items()}
    num_roles = len(ROLE_NAMES)

    def prior(unii, role, bucket):
        # Level 1: bucket-specific
        bucket_counts = unii_bucket_role_counts.get(unii, {}).get(bucket)
        total_ub = sum(bucket_counts.values()) if bucket_counts else 0
        if total_ub >= MIN_UNII_COUNT:
            role_count = bucket_counts.get(role, 0)
            return (role_count + 1) / (total_ub + num_roles)

        # Level 2: unii-level, all buckets combined
        counts = unii_role_counts.get(unii)
        total_u = sum(counts.values()) if counts else 0
        if total_u >= MIN_UNII_COUNT:
            role_count = counts.get(role, 0)
            return (role_count + 1) / (total_u + num_roles)

        # Level 3: global fallback
        return global_prior.get(role, 1e-6)

    def sample_count(unii, bucket=None):
        if bucket is not None:
            bucket_counts = unii_bucket_role_counts.get(unii, {}).get(bucket)
            return sum(bucket_counts.values()) if bucket_counts else 0
        counts = unii_role_counts.get(unii)
        return sum(counts.values()) if counts else 0

    return prior, sample_count


def pass2_optimal(results, prior_fn, sample_count_fn):
    """
    Step 1: for EXCLUSIVE roles, solve the whole formulation's assignment
    jointly via the Hungarian algorithm (maximize total confidence), instead
    of greedily walking a single sorted list — this avoids the "loser gets
    shoved into whatever's left" failure mode.

    Before entering Hungarian, each excipient's best EXCLUSIVE-role prior is
    compared against its own best NON-exclusive-role prior. If the
    non-exclusive alternative scores higher, the excipient opts out of the
    exclusive-role competition entirely instead of being forced to bid on a
    role it's actually unlikely to hold. Without this, an excipient with
    only one (low-probability) exclusive candidate — e.g. mannitol, which is
    usually a sweetener (52%) but only rarely a filler (2%) in liquid/ODT
    formulations — gets dragged into Hungarian anyway just because it HAS an
    eligible exclusive slot, often "winning" it by default (no one else
    wants it), producing a confidently wrong label. Opting out lets it be
    scored honestly in Step 2 instead, where its real best role can win.

    Step 2: whatever's left (excipients with no exclusive role, who opted
    out, or who lost the Hungarian competition) gets the best remaining
    candidate role, with a penalty applied to GENERIC_ROLES unless there's
    excipient-specific evidence for it.
    """
    n_pass2 = 0
    n_opted_out = 0
    for r in results:
        if not r["remaining"]:
            continue
        bucket = r["bucket"]
        remaining = r["remaining"]
        excipients = list(remaining.keys())
        filled_exclusive_roles = {a["role"] for a in r["assignments"] if a["role"] in EXCLUSIVE_ROLES}

        # ---- Step 1: Hungarian assignment for exclusive-role slots ----
        exclusive_cands = {u: (remaining[u] & EXCLUSIVE_ROLES) - filled_exclusive_roles
                            for u in excipients}

        # opt-out check: only let an excipient into Hungarian if its best
        # exclusive-role score actually beats its best non-exclusive score
        E = []
        for u in excipients:
            excl = exclusive_cands[u]
            if not excl:
                continue  # no exclusive candidate at all -> Step 2 by default
            non_excl = remaining[u] - EXCLUSIVE_ROLES
            best_excl_score = max(prior_fn(u, role, bucket) for role in excl)
            best_non_excl_score = (max(prior_fn(u, role, bucket) for role in non_excl)
                                    if non_excl else -1.0)
            if best_non_excl_score > best_excl_score:
                n_opted_out += 1
                continue  # opted out -> handled entirely in Step 2
            E.append(u)

        R = sorted({role for u in E for role in exclusive_cands[u]})

        matched = {}  # unii -> (role, raw_score)
        if E and R:
            INVALID = 1e6
            cost = np.full((len(E), len(R)), INVALID, dtype=np.float64)
            raw_scores = np.zeros((len(E), len(R)), dtype=np.float64)
            for i, u in enumerate(E):
                for j, role in enumerate(R):
                    if role in exclusive_cands[u]:
                        score = prior_fn(u, role, bucket)
                        raw_scores[i, j] = score
                        cost[i, j] = -score
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < INVALID / 2:  # a real pairing, not the sentinel
                    matched[E[i]] = (R[j], raw_scores[i, j])

        for u, (role, raw_score) in matched.items():
            n_candidates = len(remaining[u])
            n_samples = sample_count_fn(u, bucket)
            r["assignments"].append({
                "unii": u, "role": role,
                "confidence": float(raw_score / n_candidates),
                "raw_prior": float(raw_score),
                "n_candidates": n_candidates, "n_samples": n_samples,
                "source": "pass2",
            })
            filled_exclusive_roles.add(role)
            n_pass2 += 1

        # ---- Step 2: leftover excipients (no exclusive candidate, opted
        #      out, or lost Hungarian) -> best remaining candidate role,
        #      with a penalty on generic/fallback-prone roles ----
        new_remaining = {}
        for u in excipients:
            if u in matched:
                continue
            cands = remaining[u] - filled_exclusive_roles
            if not cands:
                new_remaining[u] = remaining[u]
                continue

            n_samples = sample_count_fn(u, bucket)
            best_role, best_score, best_raw = None, -1.0, 0.0
            for role in cands:
                raw_score = prior_fn(u, role, bucket)
                score = raw_score / len(cands)
                if role in GENERIC_ROLES and n_samples < MIN_UNII_COUNT:
                    score *= GENERIC_ROLE_PENALTY
                if score > best_score:
                    best_role, best_score, best_raw = role, score, raw_score

            r["assignments"].append({
                "unii": u, "role": best_role,
                "confidence": float(best_score),
                "raw_prior": float(best_raw),
                "n_candidates": len(cands), "n_samples": n_samples,
                "source": "pass2",
            })
            n_pass2 += 1

        r["remaining"] = new_remaining

    print(f"  Pass 2: {n_pass2} additional labels via optimal exclusive-slot assignment + fallback")
    print(f"  ({n_opted_out} excipient-formulation pairs opted out of exclusive-role "
          f"competition because their non-exclusive alternative scored higher)")


def build_weak_labels():
    """Full pipeline. Returns `results` (list of per-formulation dicts)."""
    print("[1] Building unii -> role capability map...")
    unii_to_roles = build_unii_to_roles()

    print("[2] Loading formulations...")
    formulations = load_formulations()

    print("[3] Pass 1 (unambiguous)...")
    results = pass1_unambiguous(formulations, unii_to_roles)

    print("[4] Estimating empirical priors from Pass 1...")
    prior_fn, sample_count_fn = estimate_priors(results)

    print("[5] Pass 2 (optimal exclusive-slot assignment)...")
    pass2_optimal(results, prior_fn, sample_count_fn)

    return results


def save_outputs(results):
    total_assignments = sum(len(r["assignments"]) for r in results)
    total_dropped = sum(len(r["remaining"]) for r in results)
    role_dist = Counter()
    conf_by_source = defaultdict(list)
    for r in results:
        for a in r["assignments"]:
            role_dist[a["role"]] += 1
            conf_by_source[a["source"]].append(a["confidence"])

    print(f"\nTotal labeled pairs: {total_assignments}  |  Dropped: {total_dropped}")
    for source, confs in conf_by_source.items():
        print(f"  {source}: n={len(confs)}, mean_conf={np.mean(confs):.3f}")

    output = {
        "records": [{k: v for k, v in r.items() if k != "remaining"} for r in results],
        "role_names": ROLE_NAMES,
        "exclusive_roles": sorted(EXCLUSIVE_ROLES),
        "stats": {"total_assignments": total_assignments, "total_dropped": total_dropped,
                  "role_distribution": dict(role_dist)},
    }
    with open(PICKLE_PATH, "wb") as fh:
        pickle.dump(output, fh)
    print(f"[OK] Pickle saved to {PICKLE_PATH}")

    rows_pass1, rows_pass2 = [], []
    for r in results:
        for a in r["assignments"]:
            row = {"row_idx": r["row_idx"], "api_unii": r["api_unii"],
                   "dosage_form": r["dosage_form"], "excipient_unii": a["unii"],
                   "role": a["role"], "confidence": a["confidence"]}
            if a["source"] == "pass1":
                rows_pass1.append(row)
            else:
                row["raw_prior"] = a.get("raw_prior")
                row["n_candidates"] = a.get("n_candidates")
                row["n_samples"] = a.get("n_samples")
                rows_pass2.append(row)

    os.makedirs(os.path.dirname(HIGH_CONF_CSV_PATH), exist_ok=True)
    pd.DataFrame(rows_pass1).to_csv(HIGH_CONF_CSV_PATH, index=False)
    pd.DataFrame(rows_pass2).to_csv(PASS2_CSV_PATH, index=False)
    print(f"[OK] High-confidence CSV: {HIGH_CONF_CSV_PATH} ({len(rows_pass1)} rows)")
    print(f"[OK] Pass2 CSV: {PASS2_CSV_PATH} ({len(rows_pass2)} rows)")


if __name__ == "__main__":
    results = build_weak_labels()
    save_outputs(results)