import pandas as pd
import torch
from processing import FRIENDS_PATHS, dataEncoder, USEFUL_COLS, USER_PATHS
from torch_geometric.data import Data

users_raw = pd.concat([pd.read_csv(path) for path in USER_PATHS], ignore_index=True)
users_raw = users_raw[USEFUL_COLS]
relations = pd.concat([pd.read_csv(path) for path in FRIENDS_PATHS], ignore_index=True)

class Service:
    def __init__(self):
        pass

    def findById(user_id: int):
        user = users_raw[users_raw["id"] == user_id].iloc[0]
        return {
            "id": int(user["id"]),
            "statuses_count": int(user["statuses_count"]),
            "followers_count": int(user["followers_count"]),
            "friends_count": int(user["friends_count"]),
            "favourites_count": int(user["favourites_count"]),
            "listed_count": int(user["listed_count"]),
            "lang": user["lang"] if pd.notnull(user["lang"]) else "NaN",
            "time_zone": user["time_zone"] if pd.notnull(user["time_zone"]) else "NaN",
            "location": user["location"] if pd.notnull(user["location"]) else "NaN"
        }

    def findFriends(user_id: int):
        target_friends = relations.query("source_id == @user_id")["target_id"]
        source_friends = relations.query("target_id == @user_id")["source_id"]
        friends = set(pd.concat([target_friends, source_friends]))
        friends.discard(user_id)
        for friend in friends.copy():
            if friend not in users_raw["id"].values:
                friends.remove(friend)
        return {"friends": list(friends) if friends else []}

    def predict(users_id: list[int]):
        friends = []
        edges = []
        for index, user_id in enumerate(users_id):
            user = users_raw[users_raw["id"] == user_id].iloc[0]
            friends.append(user)
            if index == 0:
                edges.append((0, index))
            else:
                edges.append((0, index))
                edges.append((index, 0))
        friends = dataEncoder(pd.DataFrame(friends))
        print(friends)
        edges = torch.tensor(edges).T
        x = torch.tensor(friends.drop(['id'], axis=1).values, dtype=torch.float)
        return Data(x = x, edge_index = edges)