from fastapi import FastAPI
from pydantic import BaseModel
from GNN import GCN, Predictions
from processing import load_users, FriendGraphBuilder

API = FastAPI()

class UserInput(BaseModel):
    statuses_count: int
    followers_count: int
    friends_count: int
    favourites_count: int
    listed_count: int
    lang: str
    time_zone: str
    location: str

model = GCN(in_channels=8, hidden_channels=64, num_classes=2)

@API.post("/predict")
def predict(user: UserInput):
    users = load_users(input=user.dict())
    graph = FriendGraphBuilder(users).build_graph()
    prediction = Predictions(model, './model/GCN.pth', graph).predict()[-1]
    label = prediction.argmax()
    confidence = prediction.max().item()
    return {
        "prediction": "Fake" if label == 1 else "Real",
        "confidence": confidence
    }

# uvicorn API:API --reload
