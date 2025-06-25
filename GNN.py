import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

# Hàm xử lý dữ liệu
# Chuyển đổi các cột thành dạng số và điền giá trị NaN bằng 0
def Data_processing(data):
    useful_cols = [
        'id',
        'name',
        'screen_name',
        'statuses_count',
        'followers_count',
        'friends_count',
        'favourites_count', 
        'listed_count',
        'lang',
        'time_zone',
        'location',
        'geo_enabled',
        'default_profile',
        'default_profile_image',
        'updated'
    ]
    data = data[useful_cols].copy()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = LabelEncoder().fit_transform(data[col].fillna('NaN'))
        else:
            data[col] = data[col].fillna(0)
    return data

# Hàm tạo đặc trưng của cạnh từ dữ liệu
# Sử dụng StandardScaler để chuẩn hóa các cột có khoảng giá trị khác nhau
# Các cột 'lang' và 'time_zone' được giữ nguyên dưới dạng số nguyên
def Edge_features(data):
    scaler_cols = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count']
    scaler = StandardScaler()
    x_scaler = scaler.fit_transform(data[scaler_cols])
    other_cols = ['lang', 'time_zone']
    x_other = data[other_cols]
    return np.hstack([x_scaler, x_other])

# Tạo ma trận cạnh từ các đặc trưng của cạnh
# Sử dụng NearestNeighbors để tìm k láng giềng gần nhất
def Edges(edge_features, k):
    neighbors = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(edge_features)
    distances, indices = neighbors.kneighbors(edge_features)
    row, col = [], []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            row.append(i)
            col.append(j)
    return torch.tensor([row, col], dtype=torch.long)

# Vẽ đồ thị từ ma trận cạnh và nhãn
# Sử dụng NetworkX để tạo đồ thị và matplotlib để vẽ
# Màu sắc của các nút được xác định theo nhãn của chúng
def Draw_graph(edges, labels, num_nodes):
    G = nx.Graph()
    edges = edges.t().tolist()
    G.add_edges_from(edges)

    cmap = plt.colormaps['tab10']
    color_map = [cmap(l % 10) for l in labels]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(num_nodes))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=70, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.8)
    plt.axis('off')
    plt.show()

# Hàm tạo đối tượng Data cho PyTorch Geometric
# Trả về đối tượng Data chứa các đặc trưng của nút, ma trận cạnh và nhãn
def Graph(data, edges, labels):
    x = data.drop(['id', 'name', 'screen_name'], axis=1).values
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    return Data(x=x, edge_index=edges, y=y)

# Mô hình GCN sử dụng GCNConv từ PyTorch Geometric
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

# Mô hình GAT sử dụng GATConv từ PyTorch Geometric    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=10):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.softmax(x, dim=1)
        return x