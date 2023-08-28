import pandas as pd
import requests

df = pd.read_excel("./respuestas.xlsx", sheet_name="RESPUESTAS")
df = df['mensajes']
data = []
for idx in range (len(df)):
    # print(df.loc[idx])
    json_data = {}

    message = df.loc[idx]
    json_data["textMessage"] = message
    data.append(json_data)
    
# print(df)

json_req = {"data": data}

x = requests.post("http://localhost:5000/api/process-nlp", json=json_req)

val = x.json()

df = pd.DataFrame(val)

#df = pd.read_json(val, index = False)

df.to_csv("clasificacion.csv", index=False)