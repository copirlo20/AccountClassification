import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

def encode_users(data: pd.DataFrame):
    data = data.copy()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = LabelEncoder().fit_transform(data[col].fillna('NaN'))
        else:
            data[col] = data[col].fillna(0)
    return data

# Lớp GAT cho các nhiệm vụ phân loại đồ thị
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Lớp GCN cho các nhiệm vụ phân loại đồ thị
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)
        return x

# Lớp GraphSAGE cho các nhiệm vụ phân loại đồ thị
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Lớp dự đoán kết quả từ mô hình
class Predictions:
    def __init__(self, model, file, graph):
        self.model = model
        self.file = file
        self.graph = graph
    
    # Phương thức tải trọng số và đặt mô hình ở chế độ đánh giá
    def load_model(self):
        self.model.load_state_dict(torch.load(self.file))
        self.model.eval()
    
    # Phương thức dự đoán kết quả
    def predict(self):
        self.load_model()
        with torch.no_grad():
            out = self.model(self.graph.x, self.graph.edge_index)
            return torch.softmax(out, dim=1)