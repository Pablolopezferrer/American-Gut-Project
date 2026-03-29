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
