import pandas as pd

# 1. Cargar el CSV original (metadatos completos)
input_csv = "CSV/AGP_joined_metadata_counts.csv"   # <- cambia por el nombre de tu archivo
df = pd.read_csv(input_csv, low_memory=False)

# 2. Columnas que queremos conservar
columns_to_keep = [
    "DIABETES",
    "IBD",
    "SEASONAL_ALLERGIES",
    "FOODALLERGIES_PEANUTS",
    "FOODALLERGIES_OTHER",
    "NONFOODALLERGIES_SUN",
    "NONFOODALLERGIES_DANDER",
    "MIGRAINE",
    "PKU",
    "NONFOODALLERGIES_BEESTINGS",
    "NONFOODALLERGIES_DRUG",
    "ASTHMA"
]

# 3. Comprobar qué columnas existen realmente en el CSV
existing_columns = [c for c in columns_to_keep if c in df.columns]

missing_columns = set(columns_to_keep) - set(existing_columns)
if missing_columns:
    print("⚠️ Columnas no encontradas en el CSV:", missing_columns)

# 4. Crear nuevo DataFrame solo con esas columnas
df_selected = df[existing_columns]

# 5. Guardar nuevo CSV
output_csv = "AGP_selected_conditions.csv"
df_selected.to_csv(output_csv, index=False)

print("✅ CSV creado correctamente:", output_csv)
print("Columnas incluidas:", existing_columns)
