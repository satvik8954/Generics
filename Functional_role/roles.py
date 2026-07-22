"""
roles.py — excipient capability lookup (unii -> possible roles) and
dosage-form-based role filtering.
"""

import pandas as pd

from config import FUNCTIONAL_CSV, UNII_CSV, ROLE_TAXONOMY, ROLE_NAMES


def build_unii_to_roles() -> dict:
    """
    Returns {unii: set(canonical_role_names)} built from the HPE-derived
    one-hot functional-category CSV, joined to UNII via excipient name.
    """
    df_roles = pd.read_csv(FUNCTIONAL_CSV)
    df_unii = pd.read_csv(UNII_CSV)
    name_to_unii = dict(zip(df_unii["excipient_names"], df_unii["unii"]))

    raw_cols = [c for c in df_roles.columns if c not in
                ("excipient_names", "hpe_monograph", "hpe_page", "match_method")]

    alias_to_canonical = {}
    for canonical, aliases in ROLE_TAXONOMY.items():
        for alias in aliases:
            alias_to_canonical[alias.lower().strip()] = canonical

    unii_to_roles = {}
    unmatched_names = 0
    for _, row in df_roles.iterrows():
        name = row["excipient_names"]
        unii = name_to_unii.get(name)
        if unii is None:
            unmatched_names += 1
            continue
        roles = set()
        for col in raw_cols:
            val = row[col]
            if pd.notna(val) and float(val) == 1.0:
                canonical = alias_to_canonical.get(col.lower().strip())
                if canonical:
                    roles.add(canonical)
        if roles:
            unii_to_roles[unii] = roles

    print(f"  unii_to_roles: {len(unii_to_roles)} excipients "
          f"(unmatched names: {unmatched_names})")
    return unii_to_roles


def allowed_roles_for_form(form: str) -> set:
    """
    Filters the full role taxonomy down to roles that make sense for a
    given dosage-form string (e.g. drop 'coating_agent' if not coated).
    """
    f = form.upper()
    allowed = set(ROLE_NAMES)

    is_solid = (("TABLET" in f) or ("CAPSULE" in f) or ("PELLET" in f)
                or ("GRANULE" in f and "SUSPENSION" not in f)
                or ("POWDER" in f and "SUSPENSION" not in f and "SOLUTION" not in f))
    is_liquid = any(k in f for k in ["SOLUTION", "SUSPENSION", "SYRUP", "ELIXIR",
                                     "LIQUID", "CONCENTRATE", "RINSE"])
    is_coated = "COAT" in f
    is_modified_release = any(k in f for k in ["EXTENDED RELEASE", "DELAYED RELEASE", "SUSTAINED"])
    is_chewable_or_odt = ("CHEWABLE" in f) or ("ORALLY DISINTEGRATING" in f)

    if not is_solid:
        allowed -= {"binder", "disintegrant", "lubricant", "glidant", "granulation_aid"}
        if is_liquid:
            allowed -= {"filler"}
    if not is_coated:
        allowed -= {"coating_agent", "plasticizer"}
    if not is_modified_release:
        allowed -= {"controlled_release"}
    if not is_liquid:
        allowed -= {"solvent", "sweetening_agent", "flavoring_agent", "buffering_agent",
                    "suspending_thickening_agent", "humectant"}
    if is_chewable_or_odt:
        allowed |= {"sweetening_agent", "flavoring_agent"}

    return allowed


def form_bucket(form: str) -> str:
    """
    Collapses a raw dosage-form string into one of a small number of
    coarse buckets, used to keep empirical priors from mixing e.g.
    'mannitol as filler in tablets' with 'mannitol as sweetener in ODTs'
    into one misleading number.
    """
    f = form.upper()
    is_liquid = any(k in f for k in ["SOLUTION", "SUSPENSION", "SYRUP", "ELIXIR",
                                     "LIQUID", "CONCENTRATE", "RINSE"])
    is_chew_odt = ("CHEWABLE" in f) or ("ORALLY DISINTEGRATING" in f)
    is_solid = (("TABLET" in f) or ("CAPSULE" in f) or ("PELLET" in f)
                or ("GRANULE" in f and "SUSPENSION" not in f)
                or ("POWDER" in f and "SUSPENSION" not in f and "SOLUTION" not in f))
    # liquid and chewable/ODT forms are merged into one bucket: both share
    # the same real behavior for versatile excipients (mannitol/sorbitol/
    # sucrose act as sweeteners/mouthfeel agents in both), and keeping them
    # separate fragments the already-sparse pass1 evidence for each so badly
    # that chewable/ODT ends up with ~0 samples and falls through to the
    # same tablet-dominated global number anyway — defeating the point.
    if is_liquid or is_chew_odt:
        return "liquid_or_chewable"
    if is_solid:
        return "solid"
    return "other"


def dosage_form_bucket_features(form: str):
    """
    Small fixed-size numeric summary of a dosage-form string, used as a
    model feature (instead of a huge one-hot over every raw form string).
    """
    import numpy as np
    f = form.upper()
    is_tablet = "TABLET" in f
    is_capsule = "CAPSULE" in f
    is_liquid = any(k in f for k in ["SOLUTION", "SUSPENSION", "SYRUP", "ELIXIR",
                                     "LIQUID", "CONCENTRATE", "RINSE"])
    is_er = any(k in f for k in ["EXTENDED RELEASE", "DELAYED RELEASE", "SUSTAINED"])
    is_coated = "COAT" in f
    is_chew_odt = ("CHEWABLE" in f) or ("ORALLY DISINTEGRATING" in f)
    return np.array([is_tablet, is_capsule, is_liquid, is_er, is_coated, is_chew_odt],
                     dtype=np.float32)