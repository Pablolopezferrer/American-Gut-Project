American Gut Project Scripts

Script 1: agp_analysis.py 

 Descarga del CSV con todas las variables
```python
# agp_analysis.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# bibliotecas para BIOM y distancias
from biom import load_table
from skbio.diversity import alpha as sk_alpha
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa


DATA_DIR = Path(r"C:\Users\pablo\Documents\MetadataAGP")

BIOM_FILE = DATA_DIR / "AG.biom"        # tu BIOM
TABLE_TXT = DATA_DIR / "AG.txt"        # (opcional) tabla en texto
META_FILE = DATA_DIR / "AG_full.txt"   # metadata

# ---------------- checks ----------------
for p in (BIOM_FILE, META_FILE):
    if not p.exists():
        raise FileNotFoundError(f"No encontrado: {p}. Asegura que el archivo existe.")

print("Archivos localizados. Cargando BIOM...")

# --- 1) Cargar BIOM ---
table = load_table(str(BIOM_FILE))  # biom table object

# Convertir BIOM a pandas DataFrame (features x samples)
# biom.Table.to_dataframe produces a dataframe with observations as rows; ensure orientation
df_counts = table.to_dataframe(dense=True).T  # ahora filas = muestras, columnas = features
# si necesitas filas=features, usa .T

print("Tabla de conteos cargada:", df_counts.shape)

# --- 2) Cargar metadata ---
# Muchos mapping files tienen una columna '#SampleID' o 'sample_name' — examinaremos la primera columna
meta = pd.read_csv(META_FILE, sep='\t', dtype=str, low_memory=False)
meta.index = meta.iloc[:,0].astype(str)   # establecer primer campo como índice (ajusta si es necesario)
meta.index.name = "sample_id"
meta = meta.drop(meta.columns[0], axis=1)  # quitar la columna que hemos pasado a índice
print("Metadata cargada:", meta.shape)

# --- 3) Alinear muestras entre metadata y counts ---
# algunas muestras en metadata pueden no estar en la tabla y viceversa
common = df_counts.index.intersection(meta.index)
print(f"Muestras en común: {len(common)}")

df_counts = df_counts.loc[common].copy()
meta = meta.loc[common].copy()

# guardar una versión unida para inspección
joined = meta.copy()
# añadir algunas columnas resumen (reads totales)
joined["total_counts"] = df_counts.sum(axis=1)
joined.to_csv(DATA_DIR / "AGP_joined_metadata_counts.csv")
print("Guardado: AGP_joined_metadata_counts.csv")

# ---------------- Analisis rápido ----------------

# 4) Alpha diversity (Shannon)
def shannon(row_counts):
    arr = np.asarray(row_counts, dtype=float)
    # skbio espera vector de counts
    return sk_alpha.shannon(arr, base=np.e)

joined["shannon"] = df_counts.apply(shannon, axis=1)
print("Alpha (Shannon) calculada. Estadísticas:")
print(joined["shannon"].describe())

# 5) Beta diversity (Bray-Curtis) + PCoA (usando skbio)
# Bray-Curtis requiere una matriz samples x features
counts_matrix = df_counts.values
sample_ids = df_counts.index.astype(str).tolist()
dm = beta_diversity(metric="braycurtis", counts=counts_matrix, ids=sample_ids)

pcoa_res = pcoa(dm)
# coger primeras 2 componentes
coords = pcoa_res.samples.iloc[:, :2]
coords.columns = ["PC1", "PC2"]

# unir con metadata y shannon
coords = coords.join(joined[["shannon"]], how="left")

# 6) Ploteo
plt.figure(figsize=(8,6))
sns.scatterplot(data=coords, x="PC1", y="PC2", hue="shannon", palette="viridis", s=40)
plt.title("PCoA (Bray-Curtis) coloreado por Shannon")
plt.legend(title="Shannon", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(DATA_DIR / "AGP_PCoA_shannon.png", dpi=150)
print("PCoA guardado en AGP_PCoA_shannon.png")

# 7) Guardar tabla de conteos subsetada y metadata final
df_counts.to_csv(DATA_DIR / "AG_counts_filtered_by_metadata.csv")
meta.to_csv(DATA_DIR / "AG_metadata_filtered.csv")
print("CSV de conteos y metadata guardados.")

# 8) Mostrar primeras líneas en consola
print("\nPrimeras filas del joined metadata:")
print(joined.head().to_string())

 ```


Script 2: inspect_meta.py 

 Saca todas las variables para poder verlas en un archivo de texto que descarga el script

```python
 from pathlib import Path
import pandas as pd

META = Path(r"C:\Users\pablo\Documents\MetadataAGP\AGP_joined_metadata_counts.csv")

df = pd.read_csv(META, index_col=0, dtype=str, low_memory=False)
print("Número filas:", df.shape[0], "Número columnas:", df.shape[1])
print("\nPrimeros 30 nombres de columnas (exactos):")
for i, c in enumerate(df.columns[:30], 1):
    print(f"{i:02d}. '{c}'")

print("\nResumen de valores no nulos por columna (top 50):")
nonulls = df.notna().sum().sort_values(ascending=False)
print(nonulls.head(50).to_string())
# guarda una lista con los 200 nombres para pegar aquí si quieres
df.columns.to_series().to_csv("column_names_list.txt", index=False)
print("\nHe guardado 'column_names_list.txt' con todos los nombres de columna.")
```


 Script 3: agp_extract_columns.py 

  Descarga del CSV con todas las varibales que nos interesan

```python
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

   ```

Script 4: diversity.py

 Los filos no aparecen n AG.txt, aparecen en AG.biom. Para calcular las diversidades y los porcentajes:
 ```python
 import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# ----------------------------
# 1. LEER ARCHIVO .BIOM
# ----------------------------

biom_path = "AG.biom"  # pon aquí la ruta a tu archivo

with h5py.File(biom_path, 'r') as f:
    
    # IDs de observación (OTUs / ASVs)
    obs_ids = [x.decode('utf-8') for x in f["observation"]["ids"][()]]
    
    # IDs de muestra
    sample_ids = [x.decode('utf-8') for x in f["sample"]["ids"][()]]
    
    # Matriz dispersa (CSR)
    data = f["observation"]["matrix"]["data"][()]
    indices = f["observation"]["matrix"]["indices"][()]
    indptr = f["observation"]["matrix"]["indptr"][()]
    
    n_obs = len(obs_ids)
    n_samples = len(sample_ids)
    
    csr_obs_by_sample = csr_matrix((data, indices, indptr), shape=(n_obs, n_samples))

    # Transponer → muestras x OTUs
    csr_sample_by_obs = csr_obs_by_sample.T.tocsr()

    # Taxonomía
    taxonomy = f["observation"]["metadata"]["taxonomy"][()]

# ----------------------------
# 2. EXTRAER PHYLUM
# ----------------------------

phylum_list = []

for entry in taxonomy:
    entry = [x.decode("utf-8") for x in entry]

    # Buscamos la columna que contiene p__Phylum
    ph = None
    for x in entry:
        if x.startswith("p__"):
            ph = x.replace("p__", "")
            break
    if ph is None:
        ph = "unassigned"

    phylum_list.append(ph)

phylum_series = pd.Series(phylum_list, index=obs_ids)

# ----------------------------
# 3. CALCULAR ÍNDICE DE SHANNON
# ----------------------------

def shannon_entropy(row_data):
    """Calcula Shannon para una fila dispersa."""
    d = row_data.data
    total = d.sum()
    
    if total == 0:
        return 0
    
    p = d / total
    return -np.sum(p * np.log(p))

shannon_values = []

for i in range(csr_sample_by_obs.shape[0]):
    row = csr_sample_by_obs.getrow(i)
    H = shannon_entropy(row)
    shannon_values.append(H)

# ----------------------------
# 4. CALCULAR % DE CADA PHYLUM
# ----------------------------

# Agrupamos columnas según phylum
obs_index = {obs: i for i, obs in enumerate(obs_ids)}
phylum_to_columns = {}

for ph in set(phylum_list):
    phylum_to_columns[ph] = [obs_index[o] for o in phylum_series[phylum_series == ph].index]

# Sumar por phylum sin densificar
phylum_counts = {}

for ph, col_idx in phylum_to_columns.items():
    sub = csr_sample_by_obs[:, col_idx]
    phylum_counts[ph] = np.asarray(sub.sum(axis=1)).flatten()

phylum_df = pd.DataFrame(phylum_counts, index=sample_ids)

# porcentajes
phylum_pct = phylum_df.div(phylum_df.sum(axis=1), axis=0) * 100

# ----------------------------
# 5. ARMAR COLUMNA FIRMICUTES, BACTEROIDETES, ETC.
# ----------------------------

def get_phylum_col(name):
    for col in phylum_pct.columns:
        if col.lower() == name.lower():
            return col
    return None

firm = get_phylum_col("Firmicutes")
bact = get_phylum_col("Bacteroidetes")
acti = get_phylum_col("Actinobacteria")
prot = get_phylum_col("Proteobacteria")

# ----------------------------
# 6. COMPILAR DATAFRAME FINAL
# ----------------------------

result = pd.DataFrame({
    "SampleID": sample_ids,
    "Diversidad_Shannon": shannon_values,
    "Firmicutes_%": phylum_pct[firm] if firm else 0,
    "Bacteroidetes_%": phylum_pct[bact] if bact else 0,
    "Actinobacteria_%": phylum_pct[acti] if acti else 0,
    "Proteobacteria_%": phylum_pct[prot] if prot else 0,
})

result["F_B_ratio"] = result["Firmicutes_%"] / result["Bacteroidetes_%"]

# ----------------------------
# 7. EXPORTAR A CSV
# ----------------------------

result.to_csv("AG_metrics.csv", index=False)

print("¡CSV generado con éxito!: AG_metrics.csv")
print(result.head())
 
 ```

 Para sacar de manera precisa la columna de enfermedades crónicas: 

 Script 5.1: select_conditions.py

 Sacamos del CSV total todas las enfermedades crónicas y las guardamos en otro CSV ordenado
  ```python
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
```
Script 5.2: enfermedadesCronicas.py

Para crear una columna llamada "Presencia de enfermedades crónicas" (SI/NO)
  ```python
import pandas as pd
import numpy as np

# Cargar el CSV original
df = pd.read_csv("CSV/AGP_selected_conditions.csv")

# Normalizamos todo a minúsculas para evitar errores
df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

# Definimos categorías
have_values = [
    "i have this condition",
    "yes",
    "diagnosed",
    "have"
]

no_values = [
    "i do not have this condition",
    "i do not have ibd",
    "no"
]

no_data_values = [
    "unknown",
    "no_data",
    np.nan
]

def evaluar_fila(row):
    valores = row.values

    # 1️⃣ Si hay al menos un "tiene"
    if any(v in have_values for v in valores if isinstance(v, str)):
        return "SI"

    # 2️⃣ Si todos son NO_DATA
    if all((v in no_data_values) or pd.isna(v) for v in valores):
        return "NO_DATA"

    # 3️⃣ Si no hay ningún "tiene" y hay al menos un "no"
    if any(v in no_values for v in valores if isinstance(v, str)):
        return "NO"

    # 4️⃣ Caso extremo (por seguridad)
    return "NO_DATA"

# Aplicar función por fila
resultado = df.apply(evaluar_fila, axis=1)

# Crear nuevo DataFrame con una sola columna
df_resultado = pd.DataFrame({
    "Presencia de enfermedades cronicas": resultado
})

# Guardar nuevo CSV
df_resultado.to_csv(
    "CSV/Presencia_enfermedades_cronicas.csv",
    index=False
)

print("CSV generado correctamente")
```
