"""
eyeball_sample.py — produces a human-reviewable CSV: for a sample of pass2
assignments, show the FULL excipient list of that formulation (not just the
one row), the dosage form, and the assigned role + confidence + diagnostics,
so you can actually judge pharmaceutically whether the assignment looks right.

Prioritizes the most informative rows to review:
    - lowest-confidence assignments (most likely wrong)
    - highest-confidence assignments (spot-check that "confident" isn't fooling you)
    - a random middle sample

Usage:
    python eyeball_sample.py --n 40
"""

import argparse
import json

import pandas as pd

from config import ORAL_CSV, PASS2_CSV_PATH, UNII_CSV


def load_unii_to_name():
    df = pd.read_csv(UNII_CSV)
    # excipient_names column can contain multiple comma-separated aliases;
    # take the first one as the canonical display name
    name_map = {}
    for _, row in df.iterrows():
        first_name = str(row["excipient_names"]).split(",")[0].strip()
        name_map[row["unii"]] = first_name
    return name_map


def load_formulation_lookup():
    oral = pd.read_csv(ORAL_CSV)
    lookup = {}
    for idx, row in oral.iterrows():
        try:
            items = json.loads(row["inactive_ingredients"])
            names = [it.get("name", "?") for it in items]
            lookup[idx] = {"api_unii": row["api_unii"], "dosage_form": row["primary_dosage_form"],
                           "all_excipient_names": names}
        except Exception:
            continue
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="rows per bucket (low/high/random)")
    args = parser.parse_args()

    pass2 = pd.read_csv(PASS2_CSV_PATH)
    lookup = load_formulation_lookup()
    unii_to_name = load_unii_to_name()

    low = pass2.nsmallest(args.n, "confidence")
    high = pass2.nlargest(args.n, "confidence")
    mid = pass2.sample(args.n, random_state=1)

    rows = []
    for bucket_name, bucket_df in [("LOW_CONF", low), ("HIGH_CONF", high), ("RANDOM", mid)]:
        for _, r in bucket_df.iterrows():
            ctx = lookup.get(r["row_idx"], {})
            rows.append({
                "bucket": bucket_name,
                "row_idx": r["row_idx"],
                "dosage_form": r["dosage_form"],
                "all_excipients_in_formulation": "; ".join(ctx.get("all_excipient_names", [])),
                "target_excipient_unii": r["excipient_unii"],
                "target_excipient_name": unii_to_name.get(r["excipient_unii"], "?"),
                "assigned_role": r["role"],
                "confidence": round(r["confidence"], 4),
                "n_candidates": r.get("n_candidates"),
                "n_samples_backing_prior": r.get("n_samples"),
                "your_verdict_correct_y_n": "",  # fill this in by hand
            })

    out_path = "outputs/pass2_eyeball_sample.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] Saved {len(rows)} rows to {out_path}")
    print("Open it, look at 'all_excipients_in_formulation' + 'assigned_role',")
    print("and fill in 'your_verdict_correct_y_n' by hand for each row.")


if __name__ == "__main__":
    main()
