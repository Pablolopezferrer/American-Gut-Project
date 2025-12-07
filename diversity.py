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
