from fastapi import FastAPI
from pydantic import BaseModel
from model import GAT, Predictions
from fastapi.exceptions import HTTPException
from service import Service
from typing import List

API = FastAPI()
service = Service()
model = GAT(in_channels=8, hidden_channels=64, num_classes=2)

class Friends(BaseModel):
    users_id: List[int]

@API.get("/user/{user_id}")
def findById(user_id: int):
    try:
        user = service.findById(user_id)
    except IndexError:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@API.get("/friends/{user_id}")
def findFriends(user_id: int):
    try:
        friends = service.findFriends(user_id)
    except IndexError:
        raise HTTPException(status_code=404, detail="Friend not found")
    return friends

@API.post("/predict")
def predict(friends: Friends):
    graph = service.build_graph(friends.users_id)
    prediction = Predictions(model, 'model.pth', graph).predict()[0]
    label = prediction.argmax()
    confidence = prediction.max().item()
    return {
        "prediction": "Fake" if label == 1 else "Real",
        "confidence": f'{confidence * 100:.2f}%'
    }

# uvicorn main:API --reload