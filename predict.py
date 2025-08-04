import pandas as pd
import matplotlib.pyplot as plt
from processing import FriendGraphBuilder, KNNGraphBuilder, dataEncoder, USER_PATHS, USEFUL_COLS
from GNN import GAT, GCN, GraphSAGE, Predictions
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Đọc dữ liệu người dùng và lấy các cột cần thiết
users_raw = pd.concat([pd.read_csv(path) for path in USER_PATHS], ignore_index=True)
labels = users_raw['dataset'].apply(lambda x: 0 if x in ['TFP', 'E13', 0] else 1)
users_raw = users_raw[USEFUL_COLS]

# Hàm tính độ chính xác của dự đoán
def accuracy(predict):
    return (predict == labels.to_numpy()).sum() / labels.shape[0]

# Hàm vẽ biểu đồ confusion matrix
def chart(predictions, name, color, index):
    results = predictions.predict().argmax(dim=1).numpy()
    cm = confusion_matrix(labels.to_numpy(), results)
    class_names = ['Real', 'Fake']
    plt.subplot(2, 3, index)
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.ylabel('True Label')
    plt.title(f'{name} {accuracy(results)*100:.2f}%\n', fontsize=11)

# Tải dữ liệu người dùng và xây dựng đồ thị
users = dataEncoder(users_raw)
graph = FriendGraphBuilder(users).build_graph()
graph_knn = KNNGraphBuilder(users).build_graph()

# Khởi tạo các mô hình
gcnModel = GCN(graph.num_node_features, 64, 2)
gatModel = GAT(graph.num_node_features, 64, 2)
graphSAGEModel = GraphSAGE(graph.num_node_features, 64, 2)

# Dự đoán kết quả từ mô hình theo quan hệ bạn bè
gcn = Predictions(gcnModel, './model/GCN.pth', graph)
gat = Predictions(gatModel, './model/GAT.pth', graph)
graphSAGE = Predictions(graphSAGEModel, './model/GraphSAGE.pth', graph)

# Dự đoán kết quả từ mô hình theo KNN
gcn_knn = Predictions(gcnModel, './model/GCN-KNN.pth', graph_knn)
gat_knn = Predictions(gatModel, './model/GAT-KNN.pth', graph_knn)
graphSAGE_knn = Predictions(graphSAGEModel, './model/GraphSAGE-KNN.pth', graph_knn)

# Biểu đồ kết quả dự đoán
fig = plt.figure(figsize=(8, 8))
manager = plt.get_current_fig_manager()
manager.window.state('zoomed')
chart(gcn, 'GCN', 'Blues', 1)
chart(gat, 'GAT', 'Reds', 2)
chart(graphSAGE, 'GraphSAGE', 'Greens', 3)
chart(gcn_knn, 'GCN-KNN', 'Blues', 4)
chart(gat_knn, 'GAT-KNN', 'Reds', 5)
chart(graphSAGE_knn, 'GraphSAGE-KNN', 'Greens', 6)
plt.suptitle('Model Evaluation on CRESCI-2015 Dataset', fontsize=16)
plt.tight_layout()
plt.show()