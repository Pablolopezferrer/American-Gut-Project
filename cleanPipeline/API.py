from fastapi import FastAPI
from pymongo import MongoClient

app = FastAPI()

client = MongoClient("mongodb://localhost:27017/")
db = client["agp_db"]
collection = db["mediciones"]

@app.get("/paciente/{id}")
def get_paciente(id: str):
    return list(collection.find({"id_paciente": id}, {"_id": 0}))