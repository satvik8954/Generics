import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
VALID_DOSE_UNITS = {"mg":1.0, "g":1000.0, "ug":1e-3, "ng":1e-6, "kg":1e6}
VALID_PER_UNIT = {"1", "mL", "g"}

EXCLUDE_FORMS = {"KIT", "GAS"}

MIN_EXCIPIENTS = 2
MAX_EXCIPIENTS = 30

DOSE_MIN = 0.0001   # mg
DOSE_MAX = 10000    # mg


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data.csv")

print("Initial:", df.shape)


# -----------------------------
# F1: Active ingredients present
# -----------------------------
df = df[df["active_ingredients"].notna()]


# -----------------------------
# F2: Excipients present
# -----------------------------
df = df[df["excipients"].notna()]


# -----------------------------
# F3: Single API only
# -----------------------------
df["api_count"] = df["active_ingredients"].apply(lambda x: len(eval(x)))
df = df[df["api_count"] == 1]


# -----------------------------
# F4: Common routes (>=50)
# -----------------------------
route_counts = df["route"].value_counts()
valid_routes = route_counts[route_counts >= 50].index
df = df[df["route"].isin(valid_routes)]


# -----------------------------
# F5: Mass units only
# -----------------------------
df = df[df["dose_unit"].isin(VALID_DOSE_UNITS.keys())]
df = df[df["per_unit"].isin(VALID_PER_UNIT)]


# -----------------------------
# F6: Parseable dose
# -----------------------------
df["dose_value"] = pd.to_numeric(df["dose_value"], errors="coerce")
df = df[df["dose_value"].notna()]


# -----------------------------
# F7: Convert to mg + range filter
# -----------------------------
df["dose_mg"] = df.apply(
    lambda row: row["dose_value"] * VALID_DOSE_UNITS[row["dose_unit"]],
    axis=1
)

df = df[(df["dose_mg"] >= DOSE_MIN) & (df["dose_mg"] <= DOSE_MAX)]


# -----------------------------
# F8: Dosage form present
# -----------------------------
df = df[df["dosage_form"].notna()]


# -----------------------------
# F9: Remove KIT, GAS
# -----------------------------
df["base_form_raw"] = df["dosage_form"].str.split(",").str[0].str.upper()
df = df[~df["base_form_raw"].isin(EXCLUDE_FORMS)]


# -----------------------------
# F10: Min excipients
# -----------------------------
df["excipient_list"] = df["excipients"].apply(lambda x: list(set(eval(x))))
df = df[df["excipient_list"].apply(len) >= MIN_EXCIPIENTS]


# -----------------------------
# F11: Max excipients
# -----------------------------
df = df[df["excipient_list"].apply(len) <= MAX_EXCIPIENTS]

print("After filtering:", df.shape)


# -----------------------------
# STRENGTH NORMALIZATION
# -----------------------------
df["log_dose_mg"] = np.log10(df["dose_mg"] + 1e-9)

# Z-score
mean = df["log_dose_mg"].mean()
std = df["log_dose_mg"].std()

df["log_dose_mg"] = (df["log_dose_mg"] - mean) / std


# -----------------------------
# PER UNIT ENCODING
# -----------------------------
per_unit_map = {"1":0, "mL":1, "g":2}
df["per_unit_id"] = df["per_unit"].map(per_unit_map)


# -----------------------------
# ROUTE ENCODING
# -----------------------------
route_vocab = {r:i for i,r in enumerate(df["route"].unique())}
df["route_id"] = df["route"].map(route_vocab)


# -----------------------------
# DOSAGE FORM PARSING
# -----------------------------
def parse_dosage(row):
    tokens = [t.strip().upper() for t in row.split(",")]
    base = tokens[0]
    modifiers = tokens[1:]
    return base, modifiers

df["base_form"], df["modifiers"] = zip(*df["dosage_form"].apply(parse_dosage))


# Base form encoding
base_vocab = {b:i for i,b in enumerate(df["base_form"].unique())}
df["base_form_id"] = df["base_form"].map(base_vocab)


# Modifier multi-hot
all_mods = set()
for mods in df["modifiers"]:
    all_mods.update(mods)

all_mods = list(all_mods)

for m in all_mods:
    df[f"mod_{m}"] = df["modifiers"].apply(lambda x: 1 if m in x else 0)


# -----------------------------
# EXCIPIENT VOCAB BUILD
# -----------------------------
from collections import Counter

exc_counter = Counter()

for exc_list in df["excipient_list"]:
    exc_counter.update(exc_list)

# keep freq >=3
exc_vocab = {exc:i for i,(exc,count) in enumerate(exc_counter.items()) if count >=3}


def map_exc_list(lst):
    return [exc_vocab.get(e, -1) for e in lst if e in exc_vocab]

df["excipient_ids"] = df["excipient_list"].apply(map_exc_list)


# -----------------------------
# FINAL SAVE
# -----------------------------
df.to_pickle("processed_v1.pkl")

print("✅ Preprocessing complete")
print("Final dataset size:", df.shape)