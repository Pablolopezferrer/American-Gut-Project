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
