import pandas as pd
import json

df = pd.read_json('Generics/Data/drug_api_excipient_mapped.jsonl', lines=True)

def get_exc_set(exc_list):
    if not isinstance(exc_list, list): return tuple()
    try:
        if len(exc_list) > 0 and isinstance(exc_list[0], dict):
            return tuple(sorted(set(str(i.get('excipient_unii', '')).strip() for i in exc_list if str(i.get('excipient_unii', '')).strip())))
        elif len(exc_list) > 0 and isinstance(exc_list[0], str):
            return tuple(sorted(set(str(x).strip() for x in exc_list if str(x).strip())))
    except: pass
    return tuple()

def get_api_set(api_list):
    if not isinstance(api_list, list): return tuple()
    try:
        if len(api_list) > 0 and isinstance(api_list[0], dict):
            apis = []
            for i in api_list:
                unii = str(i.get('api_unii', '')).strip()
                dose = str(i.get('dose_value', '')).strip()
                unit = str(i.get('dose_unit', '')).strip()
                if unii:
                    apis.append(f"{unii}_{dose}_{unit}")
            return tuple(sorted(set(apis)))
        elif len(api_list) > 0 and isinstance(api_list[0], str):
            return tuple(sorted(set(str(x).strip() for x in api_list if str(x).strip())))
    except: pass
    return tuple()

df['exc_set'] = df['excipients'].apply(get_exc_set)
df['api_set'] = df['active_ingredients'].apply(get_api_set)

df['route_clean'] = df['route'].astype(str).str.strip().str.upper()
df['dosage_form_clean'] = df['dosage_form'].astype(str).str.strip().str.upper()

# Dedup
before = len(df)
df_dedup = df.drop_duplicates(subset=['api_set', 'exc_set', 'route_clean', 'dosage_form_clean'])
after = len(df_dedup)

print(f'Total rows initially: {before}')
print(f'After deduplication: {after} (Duplicates removed: {before - after})')

# Filter unapproved drugs using results.csv
df_results = pd.read_csv('Generics/Data/results.csv', low_memory=False)
df_results['product_name_lower'] = df_results['product_name'].astype(str).str.lower()
df_results['generic_name_lower'] = df_results['generic_name'].astype(str).str.lower()

unapproved_keywords = ['unapproved']
def is_unapproved(app_type):
    if pd.isna(app_type):
        return False
    return any(kw in str(app_type).lower() for kw in unapproved_keywords)

df_results['is_unapproved'] = df_results['application_type'].apply(is_unapproved)
unapproved_map = {}
for _, row in df_results.iterrows():
    key = (row['product_name_lower'], row['generic_name_lower'])
    if row['is_unapproved']:
        unapproved_map[key] = True

df_dedup['product_name_lower'] = df_dedup['product_name'].astype(str).str.lower()
df_dedup['generic_name_lower'] = df_dedup['generic_name'].astype(str).str.lower()

def check_approved(row):
    key = (row['product_name_lower'], row['generic_name_lower'])
    return key not in unapproved_map

approved_mask = df_dedup.apply(check_approved, axis=1)
df_dedup = df_dedup[approved_mask]
print(f'After removing unapproved drugs: {len(df_dedup)}')

# Format into CSV matching f3 structure where possible
# For compatibility with preprocess.py:
# We need: api_unii, dose_mg, route, primary_dosage_form, inactive_ingredients (json str of excipients)
def extract_single_api_unii(api_list):
    if isinstance(api_list, list) and len(api_list) > 0 and isinstance(api_list[0], dict):
        return str(api_list[0].get('api_unii', '')).strip()
    return ''

def extract_dose_mg(api_list):
    if isinstance(api_list, list) and len(api_list) > 0 and isinstance(api_list[0], dict):
        return pd.to_numeric(api_list[0].get('dose_value', 0), errors='coerce')
    return 0

def extract_denominator_unit(api_list):
    if isinstance(api_list, list) and len(api_list) > 0 and isinstance(api_list[0], dict):
        return str(api_list[0].get('per_unit', '1')).strip()
    return '1'

def extract_api_smiles(api_list):
    if isinstance(api_list, list) and len(api_list) > 0 and isinstance(api_list[0], dict):
        return str(api_list[0].get('api_smiles', '')).strip()
    return ''

df_dedup['api_unii'] = df_dedup['active_ingredients'].apply(extract_single_api_unii)
df_dedup['api_smiles'] = df_dedup['active_ingredients'].apply(extract_api_smiles)
df_dedup['dose_mg'] = df_dedup['active_ingredients'].apply(extract_dose_mg)
df_dedup['denominator_unit'] = df_dedup['active_ingredients'].apply(extract_denominator_unit)
df_dedup['route'] = df_dedup['route_clean']
df_dedup['primary_dosage_form'] = df_dedup['dosage_form_clean']

# Re-encode excipients as JSON string like in the old format
def format_inactive_ingredients(exc_list):
    if not isinstance(exc_list, list): return "[]"
    formatted = []
    for e in exc_list:
        if isinstance(e, dict):
            # Format to mimic the old inactive_ingredients
            name = str(e.get('excipient_name', '')).strip()
            unii = str(e.get('excipient_unii', '')).strip()
            formatted.append({"name": name, "unii": unii})
        else:
            formatted.append({"name": str(e).strip()})
    return json.dumps(formatted)

df_dedup['inactive_ingredients'] = df_dedup['excipients'].apply(format_inactive_ingredients)

out_cols = ['api_unii', 'api_smiles', 'dose_mg', 'denominator_unit', 'route', 'primary_dosage_form', 'inactive_ingredients']
out_path = 'Generics/Data/mapped_formulations.csv'
df_dedup[out_cols].to_csv(out_path, index=False)
print(f'Saved {len(df_dedup)} formulations to {out_path}')
