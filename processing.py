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
    'location', 
    'dataset'
]

# Hàm tải dữ liệu người dùng từ các tệp CSV
def load_users(input = None, user_paths = USER_PATHS, useful_cols = USEFUL_COLS):
    users = pd.concat([pd.read_csv(path) for path in user_paths], ignore_index=True)
    users = users[useful_cols]
    users['dataset'] = users['dataset'].apply(lambda x: 0 if x in ['TFP', 'E13', 0] else 1)
    if input:
        user = pd.DataFrame([input])
        user["dataset"] = 0  # dummy label
        user["id"] = 999999999  # id giả
        users = pd.concat([users, user], ignore_index=True)
    for col in users.columns:
        if users[col].dtype == 'object':
            users[col] = LabelEncoder().fit_transform(users[col].fillna('NaN'))
        else:
            users[col] = users[col].fillna(0)
    return users

# Lớp để xây dựng đồ thị từ dữ liệu người dùng và cạnh
class FriendGraphBuilder:
    def __init__(self, users: pd.DataFrame, friends_paths=FRIENDS_PATHS):
        self.users = users
        self.friends_paths = friends_paths
        self.edges = None
        
    # Phương thức tải dữ liệu cạnh từ các tệp CSV và tạo chỉ mục cạnh
    def load_edges(self):
        friends = pd.concat([pd.read_csv(path) for path in self.friends_paths], ignore_index=True)
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
        edges = edges.astype({'source_index': 'long', 'target_index': 'long'})
        self.edges = edges
        return edges
        
    # Phương thức xây dựng đồ thị từ dữ liệu người dùng và cạnh
    def build_graph(self):
        self.load_edges()
        x = torch.tensor(self.users.drop(['id', 'dataset'], axis=1).values, dtype=torch.float)
        edge_index = torch.tensor(self.edges[['source_index', 'target_index']].values, dtype=torch.long).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        y = torch.tensor(self.users['dataset'].values, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

# Lớp để xây dựng đồ thị KNN từ dữ liệu người dùng
class KNNGraphBuilder:
    def __init__(self, users: pd.DataFrame, n_neighbors=11):
        self.users = users
        self.n_neighbors = n_neighbors
        
    # Phương thức xây dựng đồ thị KNN từ dữ liệu người dùng
    def build_graph(self):
        features = self.users.drop(['id', 'lang', 'time_zone', 'location', 'dataset'], axis=1).values
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean').fit(features)
        _, indices = knn.kneighbors(features)
        row, col = [], []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                row.append(i)
                col.append(j)
        x = torch.tensor(self.users.drop(['id', 'dataset'], axis=1).values, dtype=torch.float)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        y = torch.tensor(self.users['dataset'].values, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)