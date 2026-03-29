import pandas as pd
import numpy as np
import json
df =pd.read_excel('AGP_EXCEL.xlsx')
df = df.replace([np.inf, -np.inf], None)
df = df.where(pd.notnull(df), None)
datos=df.to_dict(orient='records')
with open('datos.json', 'w') as f:
    json.dump(datos, f, indent=4)