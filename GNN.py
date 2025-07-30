import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

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
            return out.argmax(dim=1).numpy()