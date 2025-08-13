import pandas as pd
import torch
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from model import FRIENDS_PATHS

class Graph:
    def __init__(self, encoded_users: pd.DataFrame):
        self.users = encoded_users
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

    def build(self):
        self.load_edges()
        x = torch.tensor(self.users.drop(['id'], axis=1).values, dtype=torch.float)
        edge_index = torch.tensor(self.edges[['source_index', 'target_index']].values, dtype=torch.long).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return Data(x=x, edge_index=edge_index)