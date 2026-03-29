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
