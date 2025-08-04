import pandas as pd
import torch
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

USER_PATHS = [
    "./cresci-2015/E13/users.csv", 
    "./cresci-2015/INT/users.csv",
    "./cresci-2015/TFP/users.csv", 
    "./cresci-2015/FSF/users.csv",
    "./cresci-2015/TWT/users.csv"
]

FRIENDS_PATHS = [
    "./cresci-2015/E13/friends.csv", 
    "./cresci-2015/INT/friends.csv",
    "./cresci-2015/TFP/friends.csv", 
    "./cresci-2015/FSF/friends.csv",
    "./cresci-2015/TWT/friends.csv"
]

USEFUL_COLS = [
    'id',
    'statuses_count', 
    'followers_count', 
    'friends_count',
    'favourites_count', 
    'listed_count', 
    'lang', 
    'time_zone',
    'location'
]

def dataEncoder(data: pd.DataFrame):
    data = data.copy()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = LabelEncoder().fit_transform(data[col].fillna('NaN'))
        else:
            data[col] = data[col].fillna(0)
    return data

class FriendGraphBuilder:
    def __init__(self, users_encoder: pd.DataFrame):
        self.users = users_encoder
        self.edges = None

    def load_edges(self):
        friends = pd.concat([pd.read_csv(path) for path in FRIENDS_PATHS], ignore_index=True)
        users_index = {id_: idx for idx, id_ in enumerate(self.users['id'])}
        edges_forward = pd.DataFrame({
            'source_index': friends['source_id'].map(users_index),
            'target_index': friends['target_id'].map(users_index)
        })
        edges_backward = pd.DataFrame({
            'source_index': friends['target_id'].map(users_index),
            'target_index': friends['source_id'].map(users_index)
        })
        edges = pd.concat([edges_forward, edges_backward])
        edges = edges.dropna().drop_duplicates().query("source_index != target_index").reset_index(drop=True)
        self.edges = edges.astype({'source_index': 'long', 'target_index': 'long'})
        return self.edges

    def build_graph(self):
        self.load_edges()
        x = torch.tensor(self.users.drop(['id'], axis=1).values, dtype=torch.float)
        edge_index = torch.tensor(self.edges[['source_index', 'target_index']].values, dtype=torch.long).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return Data(x=x, edge_index=edge_index)

class KNNGraphBuilder:
    def __init__(self, users_encoder: pd.DataFrame, n_neighbors = 11):
        self.users = users_encoder
        self.n_neighbors = n_neighbors

    def build_graph(self):
        features = self.users.drop(['id', 'lang', 'time_zone', 'location'], axis=1).values
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean').fit(features)
        _, indices = knn.kneighbors(features)
        row, col = [], []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                row.append(i)
                col.append(j)
        x = torch.tensor(self.users.drop(['id'], axis=1).values, dtype=torch.float)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)