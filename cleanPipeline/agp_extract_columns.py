# agp_extract_columns_fixed2.py
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(r"C:\Users\pablo\Documents\MetadataAGP")
metadata_csv = DATA_DIR / "AGP_joined_metadata_counts.csv"
ag_txt_path = DATA_DIR / "AG.txt"
out_csv = DATA_DIR / "AGP_selected_columns.csv"

if not metadata_csv.exists():
    raise FileNotFoundError(f"No encontrado: {metadata_csv}")
if not ag_txt_path.exists():
    print("Aviso: AG.txt no encontrado; se generarán columnas de filo vacías si falta.")
    
# Leer metadata (intenta utf-8, si falla usa latin1)
try:
    df = pd.read_csv(metadata_csv, index_col=0, dtype=str, low_memory=False, encoding='utf-8')
except Exception:
    df = pd.read_csv(metadata_csv, index_col=0, dtype=str, low_memory=False, encoding='latin1')

# Mapeo fijo usando los nombres que aparecen en la lista que pegaste
mapping = {
    "ID_Muestra": ["SAMPLE_ID", "SAMPLE_NAME", "SAMPLE", "HOST_SUBJECT_ID"],
    "Edad": ["AGE"],
    "Sexo": ["SEX"],
    "IMC": ["BMI", "BMI", "BMI"],
    "Tipo_de_dieta": ["DIET_TYPE", "DIET"],
    "Estado_de_salud": ["DIABETES", "IBD", "ASTHMA", "CONDITIONS_MEDICATION"],
    "Uso_antibióticos_últimos_6_meses": ["ANTIBIOTIC_MEDS", "ANTIBIOTIC_CONDITION", "ANTIBIOTIC_SELECT"],
    "Actividad_física": ["EXERCISE_FREQUENCY"],
    "Horas_de_sueño": ["SLEEP_DURATION"],
    "Consumo_de_fibra": ["FIBER_GRAMS"],
    "consumo_frutas_verduras": ["PRIMARY_VEGETABLE"],
    "ingesta_azucares_añadidos": ["SUGAR", "ADDED_SUGARS"],
    "sintomas_ansiedad_estres": ["ANXIETY", "STRESS"],
    "presencia_enfermedades_cronicas": ["DIABETES", "IBD", "ASTHMA", "CHRONIC"],
    "consumo_alcohol": ["ALCOHOL_FREQUENCY"],
    "consumo_alcohol_categoria": ["ALCOHOL_FREQUENCY"],
    "Bienestar_subjetivo": ["WELLBEING", "SUBJECTIVE_WELLBEING"],
    "Diversidad_Shannon": ["shannon", "SHANNON"]
}

def find_col(cands, cols):
    for c in cands:
        if c in cols:
            return c
    # try lowercase approximate
    lower = {col.lower(): col for col in cols}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

# extraer las columnas mapeadas
out = pd.DataFrame(index=df.index)
for new, cands in mapping.items():
    col = find_col(cands, df.columns)
    if col is not None:
        out[new] = df[col].values
    else:
        # si es 'Estado_de_salud' y no hay una sola columna, dejamos NA (lo construiremos luego)
        out[new] = pd.NA

# si ID_Muestra no fue encontrada, usar index como ID
if out["ID_Muestra"].isna().all():
    out["ID_Muestra"] = df.index.astype(str)

# ---- calcular phylum % desde AG.txt (si está) ----
phyla = ["Firmicutes", "Bacteroidetes", "Actinobacteria", "Proteobacteria"]
for p in phyla:
    out[p + "_%"] = pd.NA
out["F_B_ratio"] = pd.NA

if ag_txt_path.exists():
    print("Cargando AG.txt para calcular porcentajes por filo (esto puede tardar un poco)...")
    ag = pd.read_csv(ag_txt_path, sep="\t", index_col=0, low_memory=False)
    # detectar columna de taxonomía
    tax_col = None
    for c in ag.columns[::-1]:
        if "tax" in c.lower() or "taxonomy" in c.lower():
            tax_col = c
            break
    if tax_col is None:
        print("No se encontró columna de taxonomía en AG.txt. Se dejarán los phyla vacíos.")
    else:
        taxonomy = ag[tax_col].astype(str)
        counts = ag.drop(columns=[tax_col]).T  # muestras x features
        # alinear muestras con out index
        common = counts.index.intersection(out.index)
        counts = counts.loc[common]
        # calcular
        total = counts.sum(axis=1).astype(float)
        for p in phyla:
            mask = taxonomy.str.contains(p, case=False, na=False)
            if mask.sum() == 0:
                out[p + "_%"] = pd.NA
            else:
                subtotal = counts.loc[:, mask].sum(axis=1)
                perc = (subtotal / total.replace({0: np.nan})) * 100.0
                out.loc[perc.index, p + "_%"] = perc
        # F/B ratio
        try:
            f = out["Firmicutes_%"].astype(float)
            b = out["Bacteroidetes_%"].astype(float).replace({0: np.nan})
            out["F_B_ratio"] = (f / b).replace([np.inf, -np.inf], np.nan)
        except Exception:
            out["F_B_ratio"] = pd.NA
else:
    print("AG.txt no existe; valores de filo quedarán vacíos.")

# ---- crear columna 'presencia_enfermedades_cronicas' a partir de varias columnas ----
# si alguna de las columnas DIABETES, IBD, ASTHMA o similar está no-null/true, marcaremos como 1
diseases_cols = [c for c in ["DIABETES","IBD","ASTHMA","DIABETES_DIAGNOSE_DATE","DIABETES_MEDICATION"] if c in df.columns]
if diseases_cols:
    combined = df[diseases_cols].notna().any(axis=1)
    out["presencia_enfermedades_cronicas"] = combined.map({True: "yes", False: "no"})
else:
    out["presencia_enfermedades_cronicas"] = pd.NA

# Mantener columnas en el orden solicitado
final_cols = [
    "ID_Muestra","Edad","Sexo",
    "IMC","Tipo_de_dieta","Estado_de_salud","Uso_antibióticos_últimos_6_meses",
    "Actividad_física","Horas_de_sueño","Consumo_de_fibra","consumo_frutas_verduras",
    "ingesta_azucares_añadidos","sintomas_ansiedad_estres","presencia_enfermedades_cronicas",
    "consumo_alcohol","consumo_alcohol_categoria",
    "Diversidad_Shannon","Firmicutes_%","Bacteroidetes_%","Actinobacteria_%","Proteobacteria_%","F_B_ratio",
    "Bienestar_subjetivo"
]
for c in final_cols:
    if c not in out.columns:
        out[c] = pd.NA

out = out[final_cols]
out.to_csv(out_csv, index=False)
print("Archivo guardado en:", out_csv)
print("\nValores no nulos por columna en el CSV final:")
print(out.notna().sum())
